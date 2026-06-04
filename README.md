# Universal Audio Annotation Pipeline

Produces structured JSON annotations from any audio file, covering speech transcription, speaker diarization, emotions, vocal bursts, sound effects, and music.

**Best configuration: Triple ASR greedy** (combined equal-weight score: **4.13/5.00**)

> ### ⭐ Recommended: the default pipeline
> A turnkey, end-to-end implementation of this best configuration lives in
> **[`default_pipeline/`](default_pipeline/)** — the **default, suggested configuration**.
> It runs the full triple-ASR ensemble (VibeVoice + Parakeet + Qwen3, greedy) with the
> Whisper experts, the SFX LoRA sound-event detector and MOSS-Audio-8B-Thinking, adds
> **expressive emotion & speaking-style captions**, strict diarization-only speaker
> identity, and explicit **singing** detection, and produces per-clip JSON plus a
> self-contained HTML report. See **[docs/default_pipeline.md](docs/default_pipeline.md)**
> for model weights, setup and run instructions.
>
> ```bash
> cd default_pipeline && bash setup_environments.sh ./envs
> export UAAP_MOSS_SRC="$(pwd)/envs/MOSS-Audio"
> bash run_all.sh --audio /path/to/clips --workdir ./uaap_work --envs ./envs
> ```

## Pipeline Architecture

The **default ensemble** fuses three ASR systems, the Whisper voice experts, and two
specialist sound-event pre-passes into a single MOSS-Audio reasoning stage:

```
┌───────────────────────────────────────────────────────────────────────┐
│                    INPUT: Audio File (any length)                      │
└───────────────────────────────────┬───────────────────────────────────┘
                                     │
        ┌──────────────┬─────────────┴──────────────┐
        ▼              ▼                             ▼
  ┌──────────┐   ┌──────────┐                 ┌──────────┐
  │VibeVoice │   │Parakeet  │                 │ Qwen3    │
  │  ASR     │   │TDT v3 +  │                 │ASR 1.7B +│
  │(diarize) │   │Sortformer│                 │ Aligner  │
  └────┬─────┘   └────┬─────┘                 └────┬─────┘
       │              │  Diarized transcripts      │
       └──────┬───────┴───────────────┬────────────┘
              ▼                        ▼
  ┌──────────────────────┐  ┌──────────────────────────────┐
  │ Whisper experts (x3) │  │ Specialist sound-event prepass│
  │ emotion · timbre ·   │  │ • SFX LoRA (MOSS-8B-Instruct  │
  │ speaking-style        │  │   + laion sfx-lora r=128)     │
  │ (per utterance)      │  │ • Vocal-burst locator @0.7    │
  └──────────┬───────────┘  │   + sound-effect captioner    │
             │              └───────────────┬──────────────┘
             └───────────────┬──────────────┘
                             ▼
        ┌──────────────────────────────────────────────┐
        │            MOSS-Audio-8B-Thinking             │
        │  triple-ASR vote · diarization-only speakers  │
        │  expressive emotion & speaking-style captions │
        │  singing flag · full-timeline coverage hint   │
        └───────────────────────┬──────────────────────┘
                                │  + deterministic gap-fill
                                ▼     (non-speech background only)
        ┌──────────────────────────────────────────────┐
        │   OUTPUT: Structured JSON (covers full clip)  │
        │   [speech · vocal_burst · sound_event]        │
        └──────────────────────────────────────────────┘
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
| `triple_greedy` ⭐ **default ensemble** | VibeVoice + Parakeet + Qwen3 | Greedy | **4.13** |
| `ensemble_greedy` | Parakeet + Qwen3 | Greedy | 3.91 |
| `vibevoice` | VibeVoice only | Greedy | 3.80 |

⭐ **`triple_greedy` is the default ensemble.** The turnkey
[`default_pipeline/`](default_pipeline/) runs it with the SFX LoRA sound-event detector,
the vocal-burst locator (threshold 0.7) + sound-effect captioner, expressive emotion &
speaking-style captions, diarization-only speaker identity, singing detection, and
guaranteed full-timeline coverage. See [docs/default_pipeline.md](docs/default_pipeline.md).

## Model Requirements

| Model | HuggingFace ID | GPU VRAM |
|-------|---------------|----------|
| VibeVoice-ASR | `microsoft/VibeVoice-ASR` | ~16 GB |
| Parakeet TDT v3 | `nvidia/parakeet-tdt-0.6b-v3` | ~4 GB |
| Sortformer | `nvidia/diar_sortformer_4spk-v1` | ~2 GB |
| Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B` + `Qwen/Qwen3-ForcedAligner-0.6B` | ~8 GB |
| Whisper experts (x3) | `laion/BUD-E-Whisper`, `laion/timbre-whisper`, `laion/voice-tagging-whisper` | ~2 GB total |
| SFX LoRA | `OpenMOSS-Team/MOSS-Audio-8B-Instruct` + `laion/moss-audio-sfx-lora-v4` (gated) | ~18 GB |
| Vocal-burst locator | `laion/vocalburst-locator` (threshold 0.7) | ~1 GB |
| Sound-effect captioner | `laion/sound-effect-captioning-whisper` | ~1 GB |
| MOSS Annotator | `OpenMOSS-Team/MOSS-Audio-8B-Thinking` | ~18 GB |

**Total for triple_greedy**: ~40-50 GB VRAM (models loaded/unloaded sequentially; VibeVoice-ASR
is sharded across two GPUs). With multi-GPU, components can run in parallel on separate GPUs.

> The SFX LoRA `laion/moss-audio-sfx-lora-v4` is gated — request access and `huggingface-cli login`,
> or run with `--no-sfx`. Full model table with links: [docs/default_pipeline.md](docs/default_pipeline.md).

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
default_pipeline/        # ⭐ Default recommended configuration (turnkey scripts)
  setup_environments.sh  # Build the per-component virtual-envs
  run_all.sh             # Run all stages end-to-end
  prepare_audio.py       # Stage 0: decode + index
  workers/               # One script per stage:
    stage1a_vibevoice.py #   VibeVoice-ASR
    stage1b_parakeet.py  #   Parakeet TDT v3 + Sortformer
    stage1c_qwen3.py     #   Qwen3-ASR + ForcedAligner
    stage2_whisper_experts.py  # emotion/timbre/style
    stage3_sfx_lora.py   #   SFX LoRA sound events
    stage3b_vocalburst.py#   Vocal-burst locator + captioner
    stage4_moss_annotator.py   # MOSS final annotation (greedy)
  build_report.py        # Self-contained HTML report

pipeline/
  run_pipeline.py        # Main entry point (single-process reference implementation)
  asr_vibevoice.py       # VibeVoice-ASR component
  asr_parakeet.py        # Parakeet TDT v3 + Sortformer
  asr_qwen3.py           # Qwen3-ASR-1.7B + ForcedAligner
  whisper_experts.py     # Emotion/timbre/style Whisper models
  sfx_lora.py            # LoRA SFX sound event detection
  vocalburst_locator.py  # Vocal-burst locator + sound-effect captioner
  moss_annotator.py      # MOSS-Audio-8B-Thinking annotation (customized prompt)
  utils.py               # Shared utilities (incl. full-timeline gap-fill)

evaluation/
  eval_triple_asr.py     # Full evaluation script
  gemini_judge.py        # Gemini evaluation scorer
  build_html_report.py   # HTML report generator

docs/
  default_pipeline.md    # ⭐ Default configuration: models, links, setup, run guide
  pipeline_details.md    # Per-component details
  evaluation_results.md  # Full evaluation results
  training_lora.md       # LoRA training details
  eval_grid/index.html   # Interactive evaluation grid

examples/
  sample_output.json     # Example pipeline output
  sample_predictions/    # Sample prediction JSONs
```

## Links

**ASR**
- VibeVoice-ASR: [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) · [code](https://github.com/microsoft/VibeVoice)
- Parakeet TDT v3: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) · Sortformer: [nvidia/diar_sortformer_4spk-v1](https://huggingface.co/nvidia/diar_sortformer_4spk-v1)
- Qwen3-ASR: [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)

**Voice experts**
- Emotion: [laion/BUD-E-Whisper](https://huggingface.co/laion/BUD-E-Whisper) · Timbre: [laion/timbre-whisper](https://huggingface.co/laion/timbre-whisper) · Style: [laion/voice-tagging-whisper](https://huggingface.co/laion/voice-tagging-whisper)

**Sound events**
- SFX LoRA: [laion/moss-audio-sfx-lora-v4](https://huggingface.co/laion/moss-audio-sfx-lora-v4) (gated) on [OpenMOSS-Team/MOSS-Audio-8B-Instruct](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct)
- Vocal-burst locator: [laion/vocalburst-locator](https://huggingface.co/laion/vocalburst-locator)
- Sound-effect captioner: [laion/sound-effect-captioning-whisper](https://huggingface.co/laion/sound-effect-captioning-whisper)

**Annotator & MOSS-Audio**
- [OpenMOSS-Team/MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Thinking) · source code: [github.com/OpenMOSS/MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio)

- **Evaluation Grid**: [GitHub Pages](https://laion-ai.github.io/univeral-audio-annotation-pipeline/eval_grid/)

## License

Apache 2.0
