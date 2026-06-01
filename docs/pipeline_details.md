# Pipeline Component Details

Detailed documentation for each component in the Universal Audio Annotation Pipeline.

## Component Overview

| # | Component | Model | Purpose |
|---|-----------|-------|---------|
| 1a | VibeVoice-ASR | `microsoft/VibeVoice-ASR` | End-to-end ASR + diarization |
| 1b | Parakeet TDT v3 | `nvidia/parakeet-tdt-0.6b-v3` | Word-level ASR timestamps |
| 1c | Sortformer | `nvidia/diar_sortformer_4spk-v1` | Speaker diarization (up to 4 speakers) |
| 1d | Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B` + `Qwen/Qwen3-ForcedAligner-0.6B` | ASR + forced alignment |
| 2 | Whisper Experts | `laion/BUD-E-Whisper`, `laion/timbre-whisper`, `laion/voice-tagging-whisper` | Voice analysis |
| 3 | SFX LoRA | `OpenMOSS-Team/MOSS-Audio-8B-Instruct` + LoRA | Sound event detection |
| 4 | MOSS Annotator | `OpenMOSS-Team/MOSS-Audio-8B-Thinking` | Final structured annotation |

## 1a. VibeVoice-ASR

**Model**: `microsoft/VibeVoice-ASR`
**GPU VRAM**: ~16 GB
**Input**: WAV audio file
**Output**: Speaker-attributed utterances with timestamps

VibeVoice is an end-to-end ASR model with built-in speaker diarization. It uses Qwen2.5-7B as its language model backbone.

```python
from pipeline.asr_vibevoice import VibeVoiceASR

vv = VibeVoiceASR(device="cuda:0")
utterances = vv.run("audio.wav")
# Returns: [{"start_time": 0.0, "end_time": 5.2, "speaker_id": 0, "content": "Hello world"}]
vv.cleanup()
```

## 1b. Parakeet TDT v3 + Sortformer

**Models**: `nvidia/parakeet-tdt-0.6b-v3` (~4 GB) + `nvidia/diar_sortformer_4spk-v1` (~2 GB)
**Input**: WAV audio file
**Output**: Speaker-attributed utterances with word-level timestamps

The pipeline:
1. **Sortformer** produces speaker diarization segments (start, end, speaker_id)
2. **Parakeet TDT v3** produces word-level transcription with timestamps
3. **Merge**: Each word is assigned to the speaker with most temporal overlap, then consecutive same-speaker words are grouped into utterances (split on 1.0s gaps)

```python
from pipeline.asr_parakeet import ParakeetSortformerASR

parakeet = ParakeetSortformerASR(asr_device="cuda:0", diar_device="cuda:1")
utterances = parakeet.run("audio.wav")
parakeet.cleanup()
```

### Merge Algorithm

```
find_dominant_speaker(word_start, word_end, diar_segments):
    For each diarization segment overlapping the word:
        Accumulate overlap duration per speaker
    Return speaker with maximum overlap

group_words_to_utterances(words, gap_threshold=1.0):
    For each word:
        If speaker changed or gap > threshold:
            Start new utterance
        Else:
            Append to current utterance
```

## 1c. Qwen3-ASR

**Models**: `Qwen/Qwen3-ASR-1.7B` (~8 GB) + `Qwen/Qwen3-ForcedAligner-0.6B` (~3 GB)
**Input**: WAV audio file + optional Sortformer diarization segments
**Output**: Speaker-attributed utterances with word-level timestamps

Uses the same Sortformer diarization results as Parakeet for speaker assignment.

```python
from pipeline.asr_qwen3 import Qwen3ASR

qwen3 = Qwen3ASR(device="cuda:0")
utterances = qwen3.run("audio.wav", language="English", diar_segs=diar_segments)
qwen3.cleanup()
```

## 2. Whisper Experts

**Models**: 3 fine-tuned Whisper-small models (~600 MB each)
**GPU VRAM**: ~2 GB total
**Input**: Audio file + utterance segments (for time slicing)
**Output**: Per-utterance emotion, timbre, and style labels

| Expert | Model ID | Output |
|--------|----------|--------|
| Emotion | `laion/BUD-E-Whisper` | Emotion classification |
| Timbre | `laion/timbre-whisper` | Voice timbre description |
| Style | `laion/voice-tagging-whisper` | Speaking style tags |

All three use `openai/whisper-small` as the base processor. Each utterance segment is extracted from the full audio and run through all three models.

```python
from pipeline.whisper_experts import WhisperExperts

whisper = WhisperExperts(device="cuda:0")
analysis = whisper.analyze("audio.wav", utterances, min_duration=0.3)
# Returns: [{"start_time": 0.0, "end_time": 5.2, "speaker_id": 0,
#            "emotion": "angry", "timbre": "deep, warm", "style": "assertive"}]
whisper.cleanup()
```

## 3. SFX LoRA

**Base Model**: `OpenMOSS-Team/MOSS-Audio-8B-Instruct` (~18 GB)
**LoRA Adapter**: `LAION-AI/moss-audio-sfx-lora-v4`
**Input**: WAV audio file
**Output**: Sound event predictions with timestamps and captions

The LoRA adapter was fine-tuned on 10,998 LAION soundscapes annotated by Gemini 2.5 Pro.

```python
from pipeline.sfx_lora import SFXDetector

sfx = SFXDetector(device="cuda:0")
events = sfx.run("audio.wav", segment_duration="medium", overlapping=True)
# Returns: [{"start_time": 0.0, "end_time": 3.5, "caption": "Birds chirping"}]
sfx.cleanup()
```

**Prompt format**: "Please describe all audio events in this audio together with start time, end time, and caption for {medium} segments that are {overlapping}."

**Best configuration**: `medium` segments, `overlapping` predictions. This outperformed short/non-overlapping variants in prompt optimization (Phase 1).

## 4. MOSS Annotator

**Model**: `OpenMOSS-Team/MOSS-Audio-8B-Thinking` (~18 GB)
**Input**: WAV audio + formatted context from all upstream components
**Output**: Final structured JSON annotation

The MOSS-Audio-8B-Thinking model receives:
- Three independent ASR transcriptions (VibeVoice, Parakeet, Qwen3)
- Per-utterance voice analysis (emotion, timbre, style)
- Sound event predictions (from SFX LoRA)
- The audio itself

It then produces a comprehensive structured annotation using majority-vote ASR reconciliation.

```python
from pipeline.moss_annotator import MOSSAnnotator

moss = MOSSAnnotator(device="cuda:0")
context = moss.build_triple_context(
    vibevoice_utts, parakeet_utts, qwen3_utts,
    whisper_analysis, sfx_predictions,
)
annotations = moss.annotate("audio.wav", context, prompt_mode="triple", do_sample=False)
moss.cleanup()
```

## GPU Placement Strategy

### Single GPU (40+ GB VRAM, e.g., A100)

Load and unload models sequentially:
```
VibeVoice → cleanup → Parakeet+Sortformer → cleanup → Qwen3 → cleanup
→ Whisper → cleanup → SFX LoRA → cleanup → MOSS → cleanup
```

### Multi-GPU (e.g., 4x A100)

Parallel placement:
```
GPU 0: VibeVoice-ASR (16 GB)
GPU 1: Parakeet TDT v3 (4 GB) + Sortformer (2 GB) + Whisper experts (2 GB)
GPU 2: Qwen3-ASR (8 GB)
GPU 3: SFX LoRA (18 GB) → then MOSS (18 GB)
```

### Inference Optimization

- All models use `torch.bfloat16` precision
- `use_cache=True` for autoregressive generation
- Greedy decoding (`do_sample=False`) is recommended for the triple ASR config
- Models are loaded/unloaded sequentially to minimize peak VRAM
