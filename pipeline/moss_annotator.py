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

Return a JSON array of segment dictionaries. Each segment uses one of three schemas:

### Speech segment
```json
{
  "type": "speech",
  "start_time": 2.31,
  "end_time": 5.87,
  "transcription": "I can't believe you actually did that",
  "speaker_id": "speaker_1",
  "emotion": "anger_3",
  "age": "adult_30s",
  "gender": "female",
  "voice_timbre": "alto, warm, slightly raspy",
  "speaking_style": "confrontational, accusatory, raised voice",
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
  "emotion": "contempt_2"
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

### Field details
- start_time/end_time: Seconds from audio start, 2 decimal places.
- speaker_id: Consistent label per unique voice (speaker_1, speaker_2, ...).
- emotion: EmoNet taxonomy label with intensity suffix 1-4 (anger_3, joy_2, sadness_4, etc.).
- age: baby, toddler, child, teenager, young_adult_20s, adult_30s, adult_40s, middle_aged_50s, senior_60s, elderly_70s_plus.
- gender: male, female, nonbinary, unclear.
- speaking_rate: very_slow, slow, normal, fast, very_fast.
- voice_timbre: comma-separated descriptors.
- speaking_style: Free description of delivery manner.
- language: ISO 639-1 code.
- vocal_burst: Category label (chuckle, belly_laugh, gentle_sob, gasp, sigh, scoff, scream, etc.).
- loudness (sound events only): quiet, moderate, loud, very_loud.
- Segments may overlap in time."""

SPEAKER_RULE = """## SPEAKER IDENTITY RULE (MANDATORY)

The ASR pipelines are the DEFINITIVE ground truth for:
- How many speakers exist in this audio
- Which speaker says what and when
- Speaker IDs (speaker_0, speaker_1, etc.)

Do NOT create additional speakers beyond what the ASR systems identify."""

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

## TRIPLE-ASR RECONCILIATION

You have THREE independent ASR transcriptions below. Two use Sortformer diarization (Parakeet, Qwen3-ASR), one uses its own built-in pipeline (VibeVoice-ASR).

Reconciliation strategy:
- **Majority vote**: If 2 or 3 ASR systems agree on a word → HIGH CONFIDENCE, use it
- **Single disagreement**: If one system differs from two others → trust the majority
- **All three differ**: Listen carefully to the audio and pick the most plausible version
- **Extra words**: If any ASR has words the others missed → likely real speech, INCLUDE it
- **Timestamps**: Average or prefer the source most consistent with what you hear
- **Speaker IDs**: Trust Sortformer diarization (used by Parakeet and Qwen3); cross-check with VibeVoice

## SPEECH ANNOTATION

Use the reconciled transcripts. Fill speaker attributes from voice analysis + your listening.

## COMPLETENESS

Every second of audio should be covered by at least one annotation (speech, sound event, or both). Gaps between speech always contain at least ambient sound or room tone.

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

        if prompt_mode == "triple":
            template = TRIPLE_ASR_PROMPT
        else:
            template = ENSEMBLE_ASR_PROMPT

        instruction = (
            template
            .replace("{context}", context)
            .replace("{schema}", SCHEMA_BLOCK)
            .replace("{speaker_rule}", SPEAKER_RULE)
        )

        raw_audio = load_audio(str(audio_path), sample_rate=self.mel_sr)
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
    def build_triple_context(
        vibevoice_utts: List[Dict],
        parakeet_utts: List[Dict],
        qwen3_utts: List[Dict],
        whisper_analysis: List[Dict],
        sfx_predictions: List[Dict],
    ) -> str:
        """Build context block for triple ASR reconciliation.

        Args:
            vibevoice_utts: Utterances from VibeVoice-ASR.
            parakeet_utts: Utterances from Parakeet + Sortformer.
            qwen3_utts: Utterances from Qwen3-ASR + Sortformer.
            whisper_analysis: Per-segment voice analysis results.
            sfx_predictions: LoRA SFX sound event predictions.

        Returns:
            Formatted context string for the MOSS prompt.
        """
        lines = []

        # ASR Source 1: VibeVoice
        if vibevoice_utts:
            lines.append("### ASR Source 1: VibeVoice-ASR (end-to-end, built-in diarization)")
            for seg in vibevoice_utts:
                spk = f"Speaker {seg.get('speaker_id', '?')}"
                st = seg.get("start_time", "?")
                et = seg.get("end_time", "?")
                content = seg.get("content", "")
                lines.append(f'- {spk}: [{st}s - {et}s] "{content}"')
            lines.append("")

        # ASR Source 2: Parakeet + Sortformer
        if parakeet_utts:
            lines.append("### ASR Source 2: Parakeet TDT v3 + Sortformer diarization")
            for seg in parakeet_utts:
                spk = f"Speaker {seg.get('speaker_id', '?')}"
                st = seg.get("start_time", "?")
                et = seg.get("end_time", "?")
                content = seg.get("content", "")
                lines.append(f'- {spk}: [{st}s - {et}s] "{content}"')
            lines.append("")

        # ASR Source 3: Qwen3-ASR + Sortformer
        if qwen3_utts:
            lines.append("### ASR Source 3: Qwen3-ASR-1.7B + Sortformer diarization")
            for seg in qwen3_utts:
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
