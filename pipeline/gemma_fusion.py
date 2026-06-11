"""Text-only LLM fusion with Gemma (GGUF via llama.cpp).

The DEFAULT final annotation stage. Unlike the MOSS-Audio annotator, this model does NOT listen to
the audio — it fuses ONLY the upstream experts' text outputs (Nemotron ASR, VibeVoice/Sortformer
diarization, DiCoW overlap-aware per-speaker transcripts, Whisper voice analysis, SFX/vocal-burst
captions) into the final structured JSON. On SoundScape-Bench, Gemma-4-12B + DiCoW scores the highest
Reward of any pipeline configuration (faithful copying of the experts → best timing/transcription);
it trades precision (more hallucinated events) for that recall, since it cannot verify candidates by
listening. See docs/evaluation_results.md.
"""
import json
from typing import List, Dict, Optional

from .moss_annotator import SCHEMA_BLOCK, SPEAKER_RULE
from .utils import extract_json, dedup_events, fill_timeline_gaps


FUSION_PROMPT = r"""You are an expert audio-annotation FUSION model. You do NOT have access to the audio
itself. Produce the final structured annotation SOLELY by fusing the upstream expert-model outputs under
"Background Information" (ASR transcripts, speaker diarization, DiCoW overlap-aware per-speaker
transcripts, per-segment voice analysis, sound-event / vocal-burst captions). Trust those expert
outputs; do not invent events no expert reported, and never drop speech the ASR transcribed.

{schema}

{speaker_rule}

## FUSION RULES (no audio — rely on the experts)
- Speech: use the Nemotron 3.5 wording; place it on the VibeVoice timeline/speakers (see SPEAKER RULE).
- ⚠️ SPEECH TIMESTAMPS — READ CAREFULLY. Each `speech` segment's `start_time` and `end_time` MUST be taken
  from the **ASR / diarization timestamps** — i.e. the per-utterance `[start s - end s]` ranges given in the
  ASR sources (VibeVoice and the Nemotron word-level ASR), and the DiCoW per-speaker turns. Use ONE speech
  segment per ASR utterance, with exactly that utterance's start/end. **NEVER take a speech segment's timing
  from the "Sound Event / Music caption windows" (the SFX-LoRA predictions)** — those windows are sound
  events, not speech, and their spans must NOT become speech boundaries. If an ASR utterance has no
  timestamp, estimate it from the surrounding utterances, never from an SFX window.
- Do NOT emit a `speech` segment with an empty/blank transcription. Do NOT duplicate the same utterance as
  two segments, and do NOT make two different utterances share identical start/end times unless they are
  genuinely overlapping speech from DIFFERENT speakers.
- OVERLAPPING speech: when the DiCoW per-speaker transcripts show a speaker saying something (especially
  a DIFFERENT language) that the full-clip Nemotron pass missed, emit it as its OWN `speech` segment
  overlapping in time with the other speaker — two simultaneous voices = TWO speech segments (each with its
  own ASR timestamps).
- Sound events / music: turn the SFX-LoRA caption windows into sound_event entries (THESE windows supply the
  timing for sound_event/`music` ONLY); emit a `music` segment (not sound_event) whenever a caption
  describes music, with a detailed description.
- Vocal bursts: use the specialist detections when present. Emotion/style: from the voice analysis.
- Cover the full timeline ({duration:.1f}s); fill gaps with a quiet room-tone sound_event.

## Background Information
{context}

Output ONLY the JSON array of segments."""


def build_fusion_context(vibevoice=None, nemotron=None, sortformer_diar=None, dicow=None,
                         whisper=None, sfx=None, vocalburst=None) -> str:
    """Assemble the text context block from whatever expert outputs are available."""
    L = []
    if nemotron:
        L.append("### ASR — Nemotron 3.5 (word source; Sortformer-diarized)")
        for s in nemotron:
            L.append(f'- Speaker {s.get("speaker_id","?")}: [{s.get("start_time","?")}s-{s.get("end_time","?")}s] "{s.get("content","")}"')
        L.append("")
    if dicow:
        L.append("### DiCoW per-speaker overlap-aware transcripts (recover overlapped/simultaneous speech)")
        for e in dicow:
            L.append(f'- DiCoW speaker {e.get("speaker")}: "{e.get("text","")}"')
        L.append("")
    if vibevoice:
        L.append("### Diarization & TIMING authority — VibeVoice (use its timestamps/speakers, not its words)")
        for s in vibevoice:
            L.append(f'- Speaker {s.get("speaker_id","?")}: [{s.get("start_time","?")}s-{s.get("end_time","?")}s] "{s.get("content","")}"')
        L.append("")
    if sortformer_diar:
        L.append("### Sortformer diarization (secondary reference)")
        L.append(", ".join(f'spk{d.get("speaker","?")}[{d.get("start","?")}-{d.get("end","?")}s]' for d in sortformer_diar))
        L.append("")
    if whisper:
        L.append("### Per-Segment Voice Analysis (emotion / timbre / style)")
        for s in whisper:
            L.append(f"**Speaker {s.get('speaker_id','?')} [{s.get('start_time','?')}s-{s.get('end_time','?')}s]**:")
            for k in ["emotion", "timbre", "style"]:
                if s.get(k):
                    L.append(f"- {k.title()}: {str(s[k])[:300]}")
        L.append("")
    if sfx:
        L.append("### Sound Event / Music caption windows (LoRA) — turn into sound_event / `music`")
        for i, e in enumerate(sfx):
            L.append(f'**Window {i+1}** [{e.get("start_time","?")}s-{e.get("end_time","?")}s]: {e.get("caption", e.get("description",""))}')
        L.append("")
    if vocalburst:
        L.append("### Vocal-burst / sound-effect detections")
        for i, e in enumerate(vocalburst):
            L.append(f'**Detection {i+1}** [{e.get("start","?")}s-{e.get("end","?")}s]: {e.get("caption", e.get("description",""))}')
        L.append("")
    return "\n".join(L)


class GemmaFuser:
    """Gemma GGUF text-only fusion model (llama.cpp backend)."""

    def __init__(self, repo_id: str = "unsloth/gemma-4-12b-it-GGUF",
                 filename: str = "gemma-4-12b-it-Q8_0.gguf",
                 n_ctx: int = 16384, n_gpu_layers: int = -1, n_threads: int = 8):
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        gguf = hf_hub_download(repo_id, filename)
        self.llm = Llama(model_path=gguf, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                         verbose=False, n_threads=n_threads)
        print(f"Gemma fuser loaded: {filename}")

    def fuse(self, duration: float, context: str, sfx_predictions: Optional[List[Dict]] = None,
             max_tokens: int = 3072) -> List[Dict]:
        prompt = (FUSION_PROMPT.replace("{schema}", SCHEMA_BLOCK)
                  .replace("{speaker_rule}", SPEAKER_RULE)
                  .replace("{duration:.1f}", f"{duration:.1f}").replace("{context}", context))
        r = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=max_tokens)
        ann = dedup_events(extract_json(r["choices"][0]["message"]["content"]) or [])
        # Drop phantom speech segments with no words (these come from the model copying an SFX-LoRA
        # window's span as if it were speech). A speech event must carry a transcription.
        ann = [e for e in ann
               if not (e.get("type") == "speech" and not str(e.get("transcription") or "").strip())]
        return fill_timeline_gaps(ann, duration, sfx_predictions=sfx_predictions or [])
