"""Utility functions for the audio annotation pipeline."""

import json
import re

import librosa
import numpy as np


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    if result.startswith("<think>"):
        return ""
    return result


def extract_json(text: str):
    """Robustly extract a JSON array from model output text.

    Handles raw JSON, markdown code blocks, and embedded objects.
    Returns a list of dicts or None if no valid JSON found.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    objects = []
    for m in re.finditer(r'\{[^{}]*\}', text):
        try:
            objects.append(json.loads(m.group()))
        except Exception:
            pass
    return objects if objects else None


def _coerce_time(x):
    """MOSS occasionally emits timestamps as strings (e.g. "9.80") or null; make them floats."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def dedup_events(events: list) -> list:
    """Merge overlapping events with 70% temporal overlap threshold.

    For speech events, also checks transcription identity.
    For sound events, checks description prefix similarity.
    """
    if not events:
        return events
    # Normalise timestamps to floats so downstream consumers never see string/None times.
    for evt in events:
        if isinstance(evt, dict):
            if "start_time" in evt:
                evt["start_time"] = _coerce_time(evt.get("start_time"))
            if "end_time" in evt:
                evt["end_time"] = _coerce_time(evt.get("end_time"))
    seen = []
    for evt in events:
        is_dup = False
        for s in seen:
            if evt.get("type") != s.get("type"):
                continue
            s1 = float(evt.get("start_time", 0))
            e1 = float(evt.get("end_time", 0))
            s2 = float(s.get("start_time", 0))
            e2 = float(s.get("end_time", 0))
            overlap = max(0, min(e1, e2) - max(s1, s2))
            duration = max(e1 - s1, e2 - s2, 0.01)
            if overlap / duration < 0.7:
                continue
            if evt.get("type") == "speech":
                if (evt.get("transcription") or "").strip() == (s.get("transcription") or "").strip():
                    is_dup = True
                    break
            elif evt.get("type") == "sound_event":
                d1 = evt.get("description", "").lower()[:60]
                d2 = s.get("description", "").lower()[:60]
                if d1 == d2 or (len(d1) > 20 and d1[:20] == d2[:20]):
                    is_dup = True
                    break
            else:
                is_dup = True
                break
        if not is_dup:
            seen.append(evt)
    return seen


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target rate.

    Args:
        path: Path to audio file.
        sr: Target sample rate (default 16000).

    Returns:
        Mono audio as numpy array.
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


# Words that indicate speech / a speaker / what was said. SFX-derived fallback
# fillers must never carry these — speaker identity and spoken content are decided
# only by the ASR models, the Whisper experts and the MOSS annotator.
# Indicators of articulated speech / a specific speaker saying something. Indistinct
# crowd ambience ("restaurant chatter", "murmur of a crowd", "background voices") is
# deliberately NOT matched — that is legitimate non-speech background to describe.
_SPEECH_RE = re.compile(
    r"\b(speak\w*|spoke|spoken|says?|said|saying|talk\w*|conversation\w*|"
    r"dialogue|dialog|utters?|uttering|narrat\w*|recit\w*|monologue|"
    r"enunciat\w*|articulat\w*|exclaim\w*|replies|replied|asks|asked|answers|answered|"
    r"voice(?!s)|voiced)\b",   # singular "voice" = a specific speaker; plural "voices"/crowd kept
    re.IGNORECASE,
)


def _sanitize_nonspeech(text: str) -> str:
    """Strip any clause that describes speech/a speaker, keeping only non-speech
    (sound effects, music, ambience) description. Returns '' if nothing remains."""
    if not text:
        return ""
    # Split into clauses on sentence enders and a few clause connectors that commonly
    # separate a "someone speaks" clause from a "sound happens" clause.
    clauses = re.split(r"(?<=[.!?;])\s+|\s*,?\s+(?:while|whilst|as)\s+", text)
    kept = [c.strip() for c in clauses if c.strip() and not _SPEECH_RE.search(c)]
    out = re.sub(r"\s+", " ", " ".join(kept)).strip(" ,;.")
    return (out[0].upper() + out[1:]) if out else ""


def fill_timeline_gaps(annotations: list, duration: float,
                       sfx_predictions: list = None, min_gap: float = 0.5) -> list:
    """Guarantee full-timeline coverage of [0, duration].

    The MOSS prompt asks the model to cover every instant of audio, but the model
    does not always comply. This deterministic backstop finds any uncovered span and
    fills it: preferring the upstream SFX LoRA predictions overlapping that span
    (clipped to the gap), and otherwise inserting a quiet "near silence / room tone"
    sound_event. Filler segments are tagged ``"source": "timeline_fill"``.

    Args:
        annotations: MOSS output segments (each with start_time/end_time).
        duration: Clip duration in seconds.
        sfx_predictions: Upstream SFX LoRA predictions (start_time/end_time/caption).
        min_gap: Ignore gaps shorter than this (seconds).

    Returns:
        annotations + any filler segments needed for full coverage.
    """
    sfx_predictions = sfx_predictions or []

    def _intervals(items):
        out = []
        for a in items:
            try:
                s, e = float(a.get("start_time")), float(a.get("end_time"))
            except (TypeError, ValueError):
                continue
            if e > s:
                out.append((s, e))
        return sorted(out)

    # Merge covered intervals from the real annotations.
    merged = []
    for s, e in _intervals(annotations):
        if merged and s <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Collect the gaps in [0, duration].
    gaps, cur = [], 0.0
    for s, e in merged:
        if s - cur > min_gap:
            gaps.append((cur, s))
        cur = max(cur, e)
    if duration - cur > min_gap:
        gaps.append((cur, duration))

    fillers = []
    for gs, ge in gaps:
        # SFX predictions overlapping this gap, clipped to it.
        clipped = []
        for p in sfx_predictions:
            try:
                ps, pe = float(p.get("start_time")), float(p.get("end_time"))
            except (TypeError, ValueError):
                continue
            cs, ce = max(gs, ps), min(ge, pe)
            if ce - cs >= min_gap:
                clipped.append((cs, ce, p.get("caption", p.get("description", "")) or ""))
        clipped.sort()
        # Walk the gap, emitting SFX fills where available and silence elsewhere.
        pos = gs
        for cs, ce, desc in clipped:
            if cs - pos > min_gap:
                fillers.append(_silence(pos, cs))
            # Fallback fillers may only describe NON-SPEECH background/sound effects —
            # never a speaker or what was said. Strip any speech content from the SFX
            # caption; if nothing non-speech remains, use a neutral background label.
            clean = _sanitize_nonspeech(desc) or "Indistinct background sound / ambience"
            fillers.append({"type": "sound_event", "start_time": round(max(cs, pos), 2),
                            "end_time": round(ce, 2), "description": clean,
                            "loudness": "moderate", "source": "timeline_fill"})
            pos = max(pos, ce)
        if ge - pos > min_gap:
            fillers.append(_silence(pos, ge))

    return annotations + fillers


def _silence(s: float, e: float) -> dict:
    return {"type": "sound_event", "start_time": round(s, 2), "end_time": round(e, 2),
            "description": "near silence with faint room tone", "loudness": "quiet",
            "source": "timeline_fill"}


def find_dominant_speaker(start: float, end: float, diar_segs: list) -> int:
    """Find which speaker is most active during [start, end].

    Args:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        diar_segs: List of diarization segments with 'start', 'end', 'speaker' keys.

    Returns:
        Speaker ID (int) with most overlap, or 0 if no overlap found.
    """
    if not diar_segs:
        return 0
    speaker_time = {}
    for seg in diar_segs:
        overlap_start = max(start, seg["start"])
        overlap_end = min(end, seg["end"])
        if overlap_start < overlap_end:
            spk = seg["speaker"]
            speaker_time[spk] = speaker_time.get(spk, 0) + (overlap_end - overlap_start)
    if not speaker_time:
        return 0
    return max(speaker_time, key=speaker_time.get)
