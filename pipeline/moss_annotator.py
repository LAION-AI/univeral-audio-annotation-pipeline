"""MOSS-Audio-8B-Thinking final annotator.

Takes all upstream context (ASR transcriptions, voice analysis, SFX predictions)
and produces the final structured JSON annotation.
"""

import json
import time
from typing import List, Dict, Optional

import torch

from .utils import strip_thinking, extract_json, dedup_events


# ════════════════════════════════════════════════════════════
# Schema and prompt templates
# ════════════════════════════════════════════════════════════

SCHEMA_BLOCK = r"""## Output Schema

Return a JSON array of segment dictionaries. Each segment uses one of four schemas:

### Speech segment
```json
{
  "type": "speech",
  "start_time": 2.31,
  "end_time": 5.87,
  "transcription": "I can't believe you actually did that",
  "speaker_id": "speaker_1",
  "emotion": "clearly intense anger laced with a thread of wounded disappointment",
  "age": "adult_30s",
  "gender": "female",
  "voice_timbre": "alto, warm, slightly raspy",
  "speaking_style": "confrontational accusatory bark, voice raised and clipped, almost ranting",
  "language": "en",
  "accent": "American Midwest",
  "speaking_rate": "fast"
}
```

### Vocal burst segment
```json
{
  "type": "vocal_burst",
  "start_time": 5.87,
  "end_time": 6.14,
  "transcription": null,
  "speaker_id": "speaker_1",
  "vocal_burst": "scoff",
  "emotion": "slight, barely-veiled contempt with a flicker of amusement"
}
```

### Sound event segment
```json
{
  "type": "sound_event",
  "start_time": 3.10,
  "end_time": 5.20,
  "description": "Medium-sized dog barking aggressively in the background",
  "loudness": "moderate"
}
```

### Music segment  (USE THIS TYPE FOR MUSIC — do NOT label music as a sound_event)
```json
{
  "type": "music",
  "start_time": 9.80,
  "end_time": 15.10,
  "description": "Upbeat acoustic folk-pop: brightly strummed steel-string guitar and light tambourine, mid-tempo ~110 BPM, major key, warm and nostalgic mood, no vocals",
  "loudness": "moderate"
}
```

### Field details
- start_time/end_time: Seconds from audio start, 2 decimal places (always NUMBERS, never strings).
- speaker_id: Consistent label per unique voice (speaker_1, speaker_2, ...).
- emotion: A PRECISE EMOTION CAPTION (a few words up to ~10-12 words), NOT a single label and NOT a numeric intensity suffix. Describe the felt emotion using EmoNet voice-taxonomy emotion words (e.g. anger, contempt, disgust, fear, anxiety, sadness, disappointment, grief, joy, amusement, excitement, pride, relief, awe, tenderness, love, desire, embarrassment, guilt, shame, confusion, surprise, boredom, calmness, determination, etc.). Pin the intensity precisely with a modifier such as barely / slight / mild / moderate / clearly / strongly / intensely / extreme, and capture blends when two or more emotions co-occur (e.g. "clearly amused excitement edging into giddy joy", "barely contained anxiety under a calm surface", "intensely bitter disgust mixed with simmering anger"). Aim to nail the exact affect, not just name it.
- age: baby, toddler, child, teenager, young_adult_20s, adult_30s, adult_40s, middle_aged_50s, senior_60s, elderly_70s_plus.
- gender: male, female, nonbinary, unclear.
- speaking_rate: very_slow, slow, normal, fast, very_fast.
- voice_timbre: comma-separated descriptors.
- speaking_style: A VIVID DELIVERY CAPTION (a few words up to ~10-12 words) describing HOW it is spoken, not just the words. Capture the manner and register precisely, e.g. "low conspiratorial whisper, barely audible", "euphoric manic rant, words tumbling over each other", "slow, drowsy, slurring, barely staying awake", "booming drill-sergeant bark, clipped and commanding", "measured preacher-like cadence building to fervor", "flat deadpan mutter", "breathless pleading". Nail the specific speaking style rather than giving a generic description. SINGING: if what you hear is actually SUNG rather than spoken (melody, sustained pitches, rhythmic/musical phrasing), TRUST YOUR OWN LISTENING and say so explicitly here — even when the upstream voice-tagging / timbre experts report ordinary talking (those experts often mislabel singing as speech). Use clear wording such as "singing, melodic and sustained", "softly crooning a tune", "belting out a song", or "rhythmic, rap-like chanting". Most segments are plain speech, but whenever the delivery is sung or part-sung, flag it in this field.
- language: ISO 639-1 code.
- vocal_burst: Category label (chuckle, belly_laugh, gentle_sob, gasp, sigh, scoff, scream, etc.).
- description (sound_event / music): write a RICH, SPECIFIC caption. For sound_event name the concrete
  sound(s) (e.g. "sword being unsheathed", "ceramic mug set on wood", "high electronic beep") plus
  source/texture; avoid vague mood words. For MUSIC give genre/style, key instruments, tempo feel, mood,
  and whether it is vocal or instrumental — a bare "rock music" is not enough.
- loudness (sound_event / music only): quiet, moderate, loud, very_loud.
- Segments may overlap in time."""

SPEAKER_RULE = """## SPEAKER IDENTITY RULE (MANDATORY)

The number of distinct speakers, which speaker says what and when, and the speaker IDs
are determined SOLELY by the diarization built into the ASR sources. Follow this strict priority:

1. **If a VibeVoice-ASR transcript is present, it is the DEFINITIVE ground truth for the speaker
   count and for which speaker each utterance belongs to.** VibeVoice has the strongest built-in
   speaker diarization — trust its speaker labels. When VibeVoice is available, IGNORE the Sortformer
   speaker labels (used by Parakeet and Qwen3) for deciding HOW MANY speakers there are; use Parakeet/
   Qwen3 only to reconcile the words and their timing, not the speaker count.
   **This trust applies ONLY to speaker count / diarization — NOT to the words.** For the actual
   wording, VibeVoice carries no special weight; the Parakeet+Qwen3 majority decides the words (see
   TRIPLE-ASR RECONCILIATION). Do not copy VibeVoice's wording just because you trust its diarization.
2. **Only if there is NO VibeVoice transcript**, fall back to the Sortformer diarization speaker
   labels (shared by Parakeet and Qwen3).

Hard constraints:
- Do NOT invent speakers beyond what the chosen diarization source (VibeVoice, else Sortformer) reports.
- This rule limits the speaker COUNT, not whether speech exists: if an ASR transcribes a real line in a
  region the diarizer left empty (e.g. VibeVoice labelled it music), still include that speech and attach
  it to the most plausible existing speaker (default speaker_0) — never silently drop audible speech.
- The per-segment voice analysis and the sound-event (SFX) predictions are NOT diarization. They are
  descriptions of audio content and may mention "a second voice", "another speaker", "an older male
  voice", etc. Such phrases are NEVER evidence of an additional speaker and must NOT increase the
  speaker count. A sound event is never a speaker.
- Every speech segment must use one of the speaker IDs established by the diarization above; never
  create a new speaker_id from a sound-event caption or a voice description."""

TRIPLE_ASR_PROMPT = r"""You are an expert audio annotation model. Annotate every audible event.

{schema}

{speaker_rule}

## SOUND EVENT ANNOTATION

The upstream LoRA model provides MEDIUM-LENGTH overlapping sound event predictions below. These are broader windows that may contain multiple distinct sounds within each prediction.

Your job:
1. For each upstream prediction, listen to that time range
2. Break broad descriptions into separate, specific sound_event entries where appropriate
3. Refine timestamps to match what you actually hear
4. Add sounds the upstream missed (transitions, brief impacts, room tone shifts)
5. Continuous backgrounds (drones, music) = single entries spanning full duration

These sound-event captions describe audio content ONLY. If an SFX caption mentions a voice or
"another speaker", do NOT treat it as a new speaker or emit it as a speech segment — speaker identity
and count come exclusively from the diarization (see the SPEAKER IDENTITY RULE).

A separate SPECIALIST DETECTOR also proposes short, timestamped sound effects (listed in the
background information). Treat these as candidate sound events: verify each one against what you
actually hear at that timestamp. Integrate a candidate as a sound_event ONLY if it is genuinely
audible and real — these detections can be hallucinations or false positives, so silently ignore any
that you cannot actually hear. When a candidate is real, refine its description, timing and loudness
to match the audio.

## TRIPLE-ASR RECONCILIATION

You have THREE independent ASR transcriptions below. Two use Sortformer diarization (Parakeet, Qwen3-ASR), one uses its own built-in pipeline (VibeVoice-ASR).

Reconciliation strategy — three ASR sources vote on the WORDS/CONTENT. **MAJORITY WORDING WINS:**
- **When two or three ASR systems agree on the wording, you MUST use that majority wording — verbatim.**
  Do not substitute a different transcription. In particular:
  - **If Parakeet and Qwen3 agree on a word/phrase and VibeVoice differs, the Parakeet+Qwen3 wording
    WINS and overrides VibeVoice.** Two votes beat one. Do NOT keep VibeVoice's wording in this case.
    > Worked example — sources say:
    > • VibeVoice: "My maker told his tale. Then I will turn you behind."
    > • Parakeet:  "My maker told his tale, and I will tell you mine."
    > • Qwen3:     "My maker told his tale and I will tell you mine."
    > Parakeet and Qwen3 agree → output **"My maker told his tale, and I will tell you mine."**
    > (NOT VibeVoice's "Then I will turn you behind").
  - If **Qwen3 and VibeVoice agree** (Parakeet differs/empty) → use that wording.
  - If **VibeVoice and Parakeet agree** → use that wording.
- **VibeVoice does NOT get wording priority.** VibeVoice is authoritative ONLY for the SPEAKER COUNT and
  diarization (see the SPEAKER IDENTITY RULE) — never for choosing words. When VibeVoice is the lone
  dissenter on wording, discard its wording and follow the two-vote majority.
- **Only one source has content the others missed → ASSUME IT IS REAL and INCLUDE it.**
  e.g. if only Qwen3 transcribes a line while VibeVoice and Parakeet caught nothing there, still
  transcribe that speech — do not drop it just because the other two missed it.
- **All three differ → listen carefully and pick the most plausible version.**
- **Timestamps**: average or prefer the source most consistent with what you hear.
- **Speaker count / IDs** are governed by the SPEAKER IDENTITY RULE above (VibeVoice's built-in
  diarization when present, otherwise Sortformer). Word-level voting decides WHAT is said and WHETHER
  speech exists there; it does NOT change the number of speakers. Speech that only one ASR caught is
  still included and attributed to the most plausible existing speaker (default speaker_0).

## SPEECH ANNOTATION

Use the reconciled transcripts. Fill speaker attributes from voice analysis + your listening.

### SPEECH COMPLETENESS (MANDATORY — DO NOT DROP TRANSCRIBED SPEECH)

Transcribe EVERY utterance that the ASR systems found. The combined ASR output is the FLOOR for
speech content — your annotation must contain **at least as much speech as the ASR sources provide**,
never less. Specifically:
- Do NOT drop, skip, omit, shorten, summarise, or truncate any words, phrases or sentences that appear
  in the ASR transcripts. If the ASR systems heard a full sentence, output the full sentence.
- If a word or phrase appears in TWO or THREE of the ASR transcripts, it is high-confidence real speech
  and MUST appear in your output — there is no excuse to leave it out.
- Cover the speech across the WHOLE clip, not just the beginning. A common failure is transcribing the
  first utterance and dropping later ones — do not do this; walk through every ASR utterance to the end.
- You may fix wording, merge fragments into natural sentences, and split by speaker, but the full spoken
  content must survive. When unsure whether to include a phrase the ASRs reported, INCLUDE it.

SINGING vs SPEECH: the upstream Whisper voice-tagging/timbre experts tend to report "talking" even when
a segment is actually being sung. Rely on your own audio judgment: if you hear melody, sustained pitches
or musical rhythm, treat it as singing and state that explicitly in the segment's `speaking_style`
(e.g. "singing, melodic and sustained"). Speech is the common case, but never hide singing.

## COMPLETENESS (FULL-TIMELINE COVERAGE — MANDATORY)

This clip is {duration} seconds long. Your annotation MUST cover the ENTIRE span from 0.00s to
{duration}s with NO uncovered gaps: every instant of audio must fall inside at least one segment's
[start_time, end_time].
- Do not stop early. The last segment must reach (≈) {duration}s; never let the annotation end while
  audio remains.
- Fill every gap between speech/events: if a stretch is essentially silent, emit a sound_event
  describing it (e.g. "near silence with faint room tone", loudness "quiet") covering that span.
- If you are UNSURE what occupies some stretch of time, fall back to the upstream SFX LoRA predictions
  for that time range and use them to cover it rather than leaving a hole.
- Segments may overlap; continuous backgrounds (music, drone, room tone) are single entries spanning
  their full duration.
- BACKGROUND AT ALL TIMES: at every moment there is either some background sound or silence. For every
  span of the timeline, emit a sound_event that describes what is in the background there (e.g. "city
  traffic hum", "soft restaurant chatter and clinking glasses", "tense orchestral underscore", "wind and
  distant birdsong"). If a span genuinely has no audible background, emit a sound_event explicitly marking
  it as silence/room tone (e.g. "near silence with faint room tone", loudness "quiet"). Speech segments
  may overlap these background sound_events — the background coverage is in addition to the speech.

## Background Information

{context}

Be thorough. Output ONLY the JSON array."""

ENSEMBLE_ASR_PROMPT = r"""You are an expert audio annotation model with ACCESS TO TWO INDEPENDENT ASR SYSTEMS. Produce a structured JSON annotation of every audible event.

{schema}

{speaker_rule}

## SOUND EVENT ANNOTATION

The upstream LoRA model provides MEDIUM-LENGTH overlapping sound event predictions below.

Your job:
1. For each upstream prediction, listen to that time range
2. Break broad descriptions into separate, specific sound_event entries where appropriate
3. Refine timestamps to match what you actually hear
4. Add sounds the upstream missed

## DUAL-ASR RECONCILIATION

You have TWO ASR sources. Compare them carefully:
- Where they AGREE: high confidence — use those words and timestamps
- Where they DISAGREE: listen to the audio yourself and pick the more plausible transcription
- Where ONE has words the OTHER missed: likely real speech that was missed — INCLUDE it
- For timestamps: prefer the source that aligns better with what you hear

## COMPLETENESS

Every second of audio should be covered by at least one annotation.

## Background Information

{context}

Be thorough. Output ONLY the JSON array."""


# ════════════════════════════════════════════════════════════
# Default (recommended) prompt: Nemotron-3.5 words + VibeVoice/Sortformer diarization,
# detailed sound-event AND music captions. This is the configuration that scores best on
# SoundScape-Bench (see README / docs/evaluation_results.md).
# ════════════════════════════════════════════════════════════
NEMOTRON_PROMPT = r"""You are an expert audio annotation model. Annotate every audible event.

{schema}

{speaker_rule}

## SOUND EVENT & MUSIC ANNOTATION — WRITE DETAILED CAPTIONS

The upstream LoRA model provides MEDIUM-LENGTH overlapping sound-event windows below (one caption each).
For every window:
1. Listen to that time range yourself.
2. Write a DETAILED, SPECIFIC description — name the concrete sound(s) (e.g. "sword being unsheathed",
   "ceramic mug set on wood", "high electronic beep") plus source/texture/loudness. Do not settle for
   vague mood words; the more concrete and detailed, the better.
3. Break a broad window into separate sound_event entries where multiple distinct sounds occur.
4. Refine timestamps to what you actually hear; add sounds the upstream missed.
5. **MUSIC:** whenever music plays, emit it as a `music` segment (NOT a sound_event) and describe it in
   RICH DETAIL (genre/style, instrumentation, tempo, key/mode if discernible, mood, vocal vs instrumental).
   Continuous music = a single `music` entry spanning its full duration.
These captions describe audio CONTENT ONLY. If a caption mentions a voice or "another speaker", do NOT
treat it as a new speaker or a speech segment — speaker identity/count come from the diarization.

## ASR — words from Nemotron 3.5; timing/diarization from VibeVoice + Sortformer

- **Words / "what is being said" come from Nemotron 3.5** (shown below, Sortformer-diarized). Use
  Nemotron's wording as the transcription; do NOT invent words it did not report, do NOT drop words it did.
- **Timing + speaker diarization:** VibeVoice is the authority for the segment start/end times, the
  speaker count, and which speaker speaks when (see the SPEAKER IDENTITY RULE). The raw **Sortformer
  diarization** that Nemotron's transcript uses is shown as a secondary reference — use it to place
  Nemotron's words onto VibeVoice's speaker timeline, not to add speakers beyond VibeVoice's count.
- Map Nemotron's words onto VibeVoice's timeline: keep VibeVoice's boundaries/speaker assignment, fill
  the words from Nemotron.

### SPEECH COMPLETENESS (MANDATORY)
Transcribe EVERY utterance Nemotron found — it is the FLOOR for speech content; never output less. Cover
speech across the WHOLE clip, not just the start. When unsure whether to include a phrase Nemotron
reported, INCLUDE it. SINGING: if you hear melody/sustained pitches, mark it in `speaking_style`.

## COMPLETENESS (FULL-TIMELINE COVERAGE — MANDATORY)

This clip is {duration} seconds long. Cover the ENTIRE span 0.00s–{duration}s with no gaps; the last
segment must reach (≈){duration}s. Fill gaps with a sound_event (or "near silence with faint room tone",
loudness "quiet"). Segments may overlap; continuous backgrounds (music, drone, room tone) are single
entries spanning their full duration.

## Background Information

{context}

Be thorough. Output ONLY the JSON array."""


class MOSSAnnotator:
    """MOSS-Audio-8B-Thinking model for final structured annotation.

    Model: OpenMOSS-Team/MOSS-Audio-8B-Thinking
    GPU VRAM: ~18 GB

    Takes upstream context (ASR, voice analysis, SFX) and produces
    the final structured JSON annotation with all segment types.
    """

    MODEL_ID = "OpenMOSS-Team/MOSS-Audio-8B-Thinking"
    MAX_NEW_TOKENS = 16384

    def __init__(self, device: str = "cuda:0", moss_audio_path: Optional[str] = None):
        """Load MOSS-Audio-8B-Thinking.

        Args:
            device: CUDA device string.
            moss_audio_path: Path to MOSS-Audio source code directory.
        """
        import sys
        if moss_audio_path:
            sys.path.insert(0, moss_audio_path)

        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor

        print(f"Loading MOSS-Audio-8B-Thinking on {device}...")
        self.processor = MossAudioProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        self.model = MossAudioModel.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        self.device = device
        self.mel_sr = self.processor.config.mel_sr
        self.audio_token_id = self.processor.audio_token_id
        print("MOSS-Audio-8B-Thinking loaded.")

    def annotate(
        self,
        audio_path: str,
        context: str,
        prompt_mode: str = "triple",
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> List[Dict]:
        """Produce final structured annotation from audio + upstream context.

        Args:
            audio_path: Path to WAV audio file.
            context: Formatted context string from upstream models.
            prompt_mode: 'triple' for triple ASR or 'ensemble' for dual ASR.
            do_sample: Whether to use sampling (False = greedy).
            temperature: Sampling temperature (only used if do_sample=True).

        Returns:
            List of annotation segment dicts.
        """
        from src.audio_io import load_audio

        t0 = time.time()

        if prompt_mode == "nemotron":
            template = NEMOTRON_PROMPT
        elif prompt_mode == "triple":
            template = TRIPLE_ASR_PROMPT
        else:
            template = ENSEMBLE_ASR_PROMPT

        raw_audio = load_audio(str(audio_path), sample_rate=self.mel_sr)
        duration = len(raw_audio) / float(self.mel_sr)

        instruction = (
            template
            .replace("{context}", context)
            .replace("{schema}", SCHEMA_BLOCK)
            .replace("{speaker_rule}", SPEAKER_RULE)
            .replace("{duration}", f"{duration:.2f}")
        )

        inputs = self.processor(
            text=instruction, audios=[raw_audio], return_tensors="pt"
        ).to(self.device)

        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(torch.bfloat16)
        inputs["audio_input_mask"] = inputs["input_ids"] == self.audio_token_id

        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=do_sample,
                temperature=temperature,
                use_cache=True,
            )

        raw_text = self.processor.decode(
            gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        clean = strip_thinking(raw_text)
        parsed = extract_json(clean)
        if parsed:
            parsed = dedup_events(parsed)

        elapsed = time.time() - t0
        n_events = len(parsed) if parsed else 0
        print(f"  MOSS annotator: {n_events} events ({elapsed:.1f}s)")
        return parsed or []

    @staticmethod
    def build_nemotron_context(
        vibevoice_utts: List[Dict],
        nemotron_utts: List[Dict],
        sortformer_diar: Optional[List[Dict]],
        whisper_analysis: List[Dict],
        sfx_predictions: List[Dict],
        extra_detections: Optional[List[Dict]] = None,
    ) -> str:
        """Context for the DEFAULT (recommended) configuration.

        Words come from Nemotron 3.5 (Sortformer-diarized); VibeVoice is the timing/diarization
        authority; the raw Sortformer diarization is a secondary reference. Sound-event and music
        windows are presented for DETAILED captioning (music as its own `music` type).

        Args:
            vibevoice_utts: VibeVoice-ASR utterances (diarization / timing reference).
            nemotron_utts: Nemotron-3.5 utterances (the word source), Sortformer-diarized.
            sortformer_diar: Raw Sortformer diarization segments ({start,end,speaker}).
            whisper_analysis: Per-segment voice analysis.
            sfx_predictions: LoRA SFX sound-event windows (single caption each).
            extra_detections: Optional specialist-detected candidate sound effects to verify.
        """
        lines = []
        if nemotron_utts:
            lines.append("### ASR — Nemotron 3.5 (word source; Sortformer-diarized)")
            for s in nemotron_utts:
                lines.append(f'- Speaker {s.get("speaker_id","?")}: '
                             f'[{s.get("start_time","?")}s - {s.get("end_time","?")}s] '
                             f'"{s.get("content","")}"')
            lines.append("")
        if sortformer_diar:
            segs = ", ".join(f'spk{d.get("speaker","?")}[{d.get("start","?")}-{d.get("end","?")}s]'
                             for d in sortformer_diar)
            lines.append("### Sortformer diarization (secondary reference — the timeline Nemotron uses)")
            lines.append(segs); lines.append("")
        if vibevoice_utts:
            lines.append("### Diarization & TIMING authority — VibeVoice: use its timestamps & speaker "
                         "assignment (its WORDS are not used)")
            for s in vibevoice_utts:
                lines.append(f'- Speaker {s.get("speaker_id","?")}: '
                             f'[{s.get("start_time","?")}s - {s.get("end_time","?")}s] '
                             f'"{s.get("content","")}"')
            lines.append("")
        if whisper_analysis:
            lines.append("### Per-Segment Voice Analysis (emotion / timbre / style)")
            for seg in whisper_analysis:
                lines.append(f"**Speaker {seg.get('speaker_id','?')} "
                             f"[{seg.get('start_time','?')}s - {seg.get('end_time','?')}s]**:")
                for k in ["emotion", "timbre", "style"]:
                    v = seg.get(k)
                    if v:
                        lines.append(f"- {k.title()}: {str(v)[:400]}")
            lines.append("")
        if sfx_predictions:
            lines.append("### Sound Event / Music Windows (single LoRA caption each — verify, then DESCRIBE IN DETAIL)")
            lines.append("**Use type `music` for music, `sound_event` for everything else. Write rich, "
                         "specific descriptions; refine timings; add missed sounds.**")
            lines.append("")
            for i, e in enumerate(sfx_predictions):
                lines.append(f"**Window {i+1}** [{e.get('start_time','?')}s - {e.get('end_time','?')}s]: "
                             f"{e.get('caption', e.get('description',''))}")
                lines.append("")
        else:
            lines.append("### Sound Event Predictions: Not available for this scene"); lines.append("")
        if extra_detections:
            lines.append("### Additional Sound Effects Detected by a Specialist Detector")
            lines.append("**Candidates only — integrate one ONLY if you can actually hear it at that time; "
                         "ignore false positives.**")
            lines.append("")
            for i, e in enumerate(extra_detections):
                lines.append(f"**Detection {i+1}** [{e.get('start','?')}s - {e.get('end','?')}s]: "
                             f"{e.get('caption', e.get('description',''))}")
                lines.append("")
        return "\n".join(lines)

    @staticmethod
    def build_triple_context(
        vibevoice_utts: List[Dict],
        parakeet_utts: List[Dict],
        qwen3_utts: List[Dict],
        whisper_analysis: List[Dict],
        sfx_predictions: List[Dict],
        extra_detections: Optional[List[Dict]] = None,
    ) -> str:
        """Build context block for triple ASR reconciliation.

        Args:
            vibevoice_utts: Utterances from VibeVoice-ASR.
            parakeet_utts: Utterances from Parakeet + Sortformer.
            qwen3_utts: Utterances from Qwen3-ASR + Sortformer.
            whisper_analysis: Per-segment voice analysis results.
            sfx_predictions: LoRA SFX sound event predictions.
            extra_detections: Optional specialist-detected candidate sound effects,
                each {start, end, confidence?, caption}. Presented to MOSS neutrally as
                candidates to verify (they may be hallucinations).

        Returns:
            Formatted context string for the MOSS prompt.
        """
        lines = []

        # Sources are ordered so the two independent word-level ASRs (Parakeet, Qwen3)
        # come FIRST as the primary wording reference; VibeVoice is last and is only a
        # diarization reference for wording purposes (it must not win wording votes).

        # ASR Source 1: Parakeet + Sortformer (primary wording reference)
        if parakeet_utts:
            lines.append("### ASR Source 1: Parakeet TDT v3 + Sortformer diarization (primary wording reference)")
            for seg in parakeet_utts:
                spk = f"Speaker {seg.get('speaker_id', '?')}"
                st = seg.get("start_time", "?")
                et = seg.get("end_time", "?")
                content = seg.get("content", "")
                lines.append(f'- {spk}: [{st}s - {et}s] "{content}"')
            lines.append("")

        # ASR Source 2: Qwen3-ASR + Sortformer (primary wording reference)
        if qwen3_utts:
            lines.append("### ASR Source 2: Qwen3-ASR-1.7B + Sortformer diarization (primary wording reference)")
            for seg in qwen3_utts:
                spk = f"Speaker {seg.get('speaker_id', '?')}"
                st = seg.get("start_time", "?")
                et = seg.get("end_time", "?")
                content = seg.get("content", "")
                lines.append(f'- {spk}: [{st}s - {et}s] "{content}"')
            lines.append("")
        if parakeet_utts and qwen3_utts:
            lines.append("> NOTE: Where Source 1 (Parakeet) and Source 2 (Qwen3) agree on the wording, "
                         "that agreed wording is the MAJORITY and must be used verbatim, overriding any "
                         "different wording in Source 3 (VibeVoice) below.")
            lines.append("")

        # ASR Source 3: VibeVoice (diarization reference only — NOT a wording authority)
        if vibevoice_utts:
            lines.append("### ASR Source 3: VibeVoice-ASR (end-to-end; speaker-diarization reference — do NOT prefer its wording)")
            for seg in vibevoice_utts:
                spk = f"Speaker {seg.get('speaker_id', '?')}"
                st = seg.get("start_time", "?")
                et = seg.get("end_time", "?")
                content = seg.get("content", "")
                lines.append(f'- {spk}: [{st}s - {et}s] "{content}"')
            lines.append("")

        # Whisper voice analysis
        if whisper_analysis:
            lines.append("### Per-Segment Voice Analysis (emotion / timbre / style)")
            for seg in whisper_analysis:
                st = seg.get("start_time", "?")
                et = seg.get("end_time", "?")
                lines.append(f"**Speaker {seg.get('speaker_id', '?')} [{st}s - {et}s]**:")
                for k in ["emotion", "timbre", "style"]:
                    v = seg.get(k, "N/A")
                    if v:
                        v = str(v)[:400]
                        lines.append(f"- {k.title()}: {v}")
            lines.append("")

        # LoRA SFX predictions
        if sfx_predictions:
            lines.append("### Sound Event Predictions (fine-tuned LoRA model, medium overlap)")
            lines.append("**IMPORTANT: Verify each prediction against what you hear. Refine timestamps. Add sounds the model missed.**")
            lines.append("")
            for i, e in enumerate(sfx_predictions):
                lines.append(
                    f"**Event {i+1}** [{e.get('start_time','?')}s - {e.get('end_time','?')}s]: "
                    f"{e.get('caption', e.get('description', ''))}"
                )
                lines.append("")
        else:
            lines.append("### Sound Event Predictions: Not available for this scene")
            lines.append("")

        # Specialist detector: extra candidate sound effects (verify before keeping)
        if extra_detections:
            lines.append("### Additional Sound Effects Detected by a Specialist Detector")
            lines.append("**Candidates only — integrate a sound_event for one ONLY if you can actually "
                         "hear it at that time. These may be false positives; ignore any you cannot hear.**")
            lines.append("")
            for i, e in enumerate(extra_detections):
                lines.append(
                    f"**Detection {i+1}** [{e.get('start','?')}s - {e.get('end','?')}s]: "
                    f"{e.get('caption', e.get('description', ''))}"
                )
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def build_ensemble_context(
        parakeet_utts: List[Dict],
        qwen3_utts: List[Dict],
        whisper_analysis: List[Dict],
        sfx_predictions: List[Dict],
    ) -> str:
        """Build context block for dual ASR reconciliation.

        Args:
            parakeet_utts: Utterances from Parakeet + Sortformer.
            qwen3_utts: Utterances from Qwen3-ASR + Sortformer.
            whisper_analysis: Per-segment voice analysis results.
            sfx_predictions: LoRA SFX sound event predictions.

        Returns:
            Formatted context string for the MOSS prompt.
        """
        lines = []

        if parakeet_utts:
            lines.append("### ASR Source 1: Parakeet TDT v3 + Sortformer diarization")
            for seg in parakeet_utts:
                spk = f"Speaker {seg.get('speaker_id', '?')}"
                st = seg.get("start_time", "?")
                et = seg.get("end_time", "?")
                content = seg.get("content", "")
                lines.append(f'- {spk}: [{st}s - {et}s] "{content}"')
            lines.append("")

        if qwen3_utts:
            lines.append("### ASR Source 2: Qwen3-ASR-1.7B + Sortformer diarization")
            for seg in qwen3_utts:
                spk = f"Speaker {seg.get('speaker_id', '?')}"
                st = seg.get("start_time", "?")
                et = seg.get("end_time", "?")
                content = seg.get("content", "")
                lines.append(f'- {spk}: [{st}s - {et}s] "{content}"')
            lines.append("")

        if whisper_analysis:
            lines.append("### Per-Segment Voice Analysis (emotion / timbre / style)")
            for seg in whisper_analysis:
                st = seg.get("start_time", "?")
                et = seg.get("end_time", "?")
                lines.append(f"**Speaker {seg.get('speaker_id', '?')} [{st}s - {et}s]**:")
                for k in ["emotion", "timbre", "style"]:
                    v = seg.get(k, "N/A")
                    if v:
                        lines.append(f"- {k.title()}: {v}")
            lines.append("")

        if sfx_predictions:
            lines.append("### Sound Event Predictions (fine-tuned LoRA model)")
            for i, e in enumerate(sfx_predictions):
                lines.append(
                    f"**Event {i+1}** [{e.get('start_time','?')}s - {e.get('end_time','?')}s]: "
                    f"{e.get('caption', e.get('description', ''))}"
                )
            lines.append("")

        return "\n".join(lines)

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.processor
        torch.cuda.empty_cache()
