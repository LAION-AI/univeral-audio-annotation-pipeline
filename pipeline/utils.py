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


def dedup_events(events: list) -> list:
    """Merge overlapping events with 70% temporal overlap threshold.

    For speech events, also checks transcription identity.
    For sound events, checks description prefix similarity.
    """
    if not events:
        return events
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
                if evt.get("transcription", "").strip() == s.get("transcription", "").strip():
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
