# Universal Audio Annotation Pipeline

Produces structured JSON annotations from any audio file, covering speech transcription, speaker diarization, emotions, vocal bursts, sound effects, and music.

**Best configuration: Triple ASR greedy** (combined equal-weight score: **4.13/5.00**)

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Audio File (WAV)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │VibeVoice │     │Parakeet  │     │ Qwen3    │
   │  ASR     │     │TDT v3 + │     │ASR 1.7B +│
   │          │     │Sortformer│     │Aligner   │
   └────┬─────┘     └────┬─────┘     └────┬─────┘
        │                │                │
        │         ┌──────┴──────┐         │
        │         │  Diarized   │         │
        │         │ Transcripts │         │
        │         └──────┬──────┘         │
        │                │                │
        │    ┌───────────┼───────────┐    │
        │    ▼           ▼           ▼    │
        │ ┌────────┐ ┌────────┐ ┌────────┐│
        │ │Emotion │ │Timbre  │ │Style   ││
        │ │Whisper │ │Whisper │ │Whisper ││
        │ └───┬────┘ └───┬────┘ └───┬────┘│
        │     └───────────┼─────────┘     │
        │                 │               │
        │    ┌────────────┴────────┐      │
        │    │ Voice Analysis Per  │      │
        │    │ Utterance           │      │
        │    └────────────┬────────┘      │
        │                 │               │
        │         ┌───────┴────────┐      │
        │         │ SFX LoRA       │      │
        │         │ (MOSS-8B-Inst +│      │
        │         │  LoRA r=128)   │      │
        │         └───────┬────────┘      │
        │                 │               │
        └────────┬────────┼───────┬───────┘
                 │        │       │
                 ▼        ▼       ▼
        ┌────────────────────────────────┐
        │   MOSS-Audio-8B-Thinking       │
        │   (Triple ASR reconciliation   │
        │    + structured annotation)    │
        └───────────────┬────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │   OUTPUT: Structured JSON      │
        │   [speech, vocal_burst,        │
        │    sound_event segments]        │
        └────────────────────────────────┘
```

## Output Format

The pipeline produces a JSON array of segment dictionaries. Three segment types:

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

### Field Reference

| Field | Values |
|-------|--------|
| `emotion` | EmoNet taxonomy + intensity 1-4 (e.g. `anger_3`, `joy_2`, `sadness_4`) |
| `age` | `baby`, `toddler`, `child`, `teenager`, `young_adult_20s`, `adult_30s`, `adult_40s`, `middle_aged_50s`, `senior_60s`, `elderly_70s_plus` |
| `gender` | `male`, `female`, `nonbinary`, `unclear` |
| `speaking_rate` | `very_slow`, `slow`, `normal`, `fast`, `very_fast` |
| `vocal_burst` | `chuckle`, `belly_laugh`, `gentle_sob`, `gasp`, `sigh`, `scoff`, `scream`, etc. |
| `loudness` | `quiet`, `moderate`, `loud`, `very_loud` |
| `language` | ISO 639-1 code |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python -m pipeline.run_pipeline \
  --audio input.wav \
  --output output.json \
  --config triple_greedy \
  --gpus 0
```

### Configurations

| Config | ASR Systems | Decoding | Combined Score |
|--------|-------------|----------|----------------|
| `triple_greedy` | VibeVoice + Parakeet + Qwen3 | Greedy | **4.13** |
| `ensemble_greedy` | Parakeet + Qwen3 | Greedy | 3.91 |
| `vibevoice` | VibeVoice only | Greedy | 3.80 |

## Model Requirements

| Model | HuggingFace ID | GPU VRAM |
|-------|---------------|----------|
| VibeVoice-ASR | `microsoft/VibeVoice-ASR` | ~16 GB |
| Parakeet TDT v3 | `nvidia/parakeet-tdt-0.6b-v3` | ~4 GB |
| Sortformer | `nvidia/diar_sortformer_4spk-v1` | ~2 GB |
| Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B` + `Qwen/Qwen3-ForcedAligner-0.6B` | ~8 GB |
| Whisper experts (x3) | `laion/BUD-E-Whisper`, `laion/timbre-whisper`, `laion/voice-tagging-whisper` | ~2 GB total |
| SFX LoRA | `OpenMOSS-Team/MOSS-Audio-8B-Instruct` + `LAION-AI/moss-audio-sfx-lora-v4` | ~18 GB |
| MOSS Annotator | `OpenMOSS-Team/MOSS-Audio-8B-Thinking` | ~18 GB |

**Total for triple_greedy**: ~40-50 GB VRAM (models loaded/unloaded sequentially on a single GPU).

With multi-GPU, components can run in parallel on separate GPUs.

## Evaluation Results

Tested 8 configurations across 60 scenes (50 synthetic TTS + 10 YouTube), scored by Gemini 3.1 Pro on 9 dimensions (0-5 scale).

| # | ASR | Decoding | Synthetic | YouTube | Combined |
|---|-----|----------|-----------|---------|----------|
| 1 | **Triple** | **greedy** | 3.70 | **4.56** | **4.13** |
| 2 | Triple | temp=0.5 | **3.74** | 4.23 | 3.99 |
| 3 | Ensemble | greedy | 3.43 | 4.39 | 3.91 |
| 4 | VibeVoice | greedy | 3.53 | 4.08 | 3.80 |
| 5 | VibeVoice | temp=0.5 | 3.70 | 3.65 | 3.67 |
| 6 | Ensemble | temp=0.5 | 3.50 | 3.81 | 3.66 |

**Combined** = equal-weight average of Synthetic and YouTube scores.

For detailed evaluation results, see [docs/evaluation_results.md](docs/evaluation_results.md).

Interactive evaluation grid: [GitHub Pages](https://laion-ai.github.io/univeral-audio-annotation-pipeline/eval_grid/)

## Repository Structure

```
pipeline/
  run_pipeline.py        # Main entry point
  asr_vibevoice.py       # VibeVoice-ASR component
  asr_parakeet.py        # Parakeet TDT v3 + Sortformer
  asr_qwen3.py           # Qwen3-ASR-1.7B + ForcedAligner
  whisper_experts.py     # Emotion/timbre/style Whisper models
  sfx_lora.py            # LoRA SFX sound event detection
  moss_annotator.py      # MOSS-Audio-8B-Thinking annotation
  utils.py               # Shared utilities

evaluation/
  eval_triple_asr.py     # Full evaluation script
  gemini_judge.py        # Gemini evaluation scorer
  build_html_report.py   # HTML report generator

docs/
  pipeline_details.md    # Per-component details
  evaluation_results.md  # Full evaluation results
  training_lora.md       # LoRA training details
  eval_grid/index.html   # Interactive evaluation grid

examples/
  sample_output.json     # Example pipeline output
  sample_predictions/    # Sample prediction JSONs
```

## Links

- **LoRA Weights**: [LAION-AI/moss-audio-sfx-lora-v4](https://huggingface.co/LAION-AI/moss-audio-sfx-lora-v4)
- **MOSS-Audio**: [OpenMOSS-Team/MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Thinking)
- **Evaluation Grid**: [GitHub Pages](https://laion-ai.github.io/univeral-audio-annotation-pipeline/eval_grid/)

## License

Apache 2.0
