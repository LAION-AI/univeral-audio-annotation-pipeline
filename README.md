# Universal Audio Annotation Pipeline

Produces structured JSON annotations from any audio file, covering speech transcription, speaker diarization, emotions, vocal bursts, sound effects, and music.

**Best configuration: Nemotron 3.5 words + VibeVoice/Sortformer diarization** (`nemotron_vibevoice`) — the top-scoring pipeline on [SoundScape-Bench](#evaluation-results) (Reward **0.236**), ahead of every other pipeline configuration and of Gemini 3 Flash.

> ### ⭐ Recommended: the default pipeline
> A turnkey, end-to-end implementation of this best configuration lives in
> **[`default_pipeline/`](default_pipeline/)** — the **default, suggested configuration**.
> It uses **Nemotron 3.5** for the words (what is said), **VibeVoice + Sortformer** for the
> diarization and timing, plus the Whisper experts, the SFX LoRA sound-event detector and
> MOSS-Audio-8B-Thinking. It adds **expressive emotion & speaking-style captions**, **detailed
> sound-effect captions and a dedicated `music` segment type** (genre / instrumentation / tempo /
> mood), strict diarization-only speaker identity, and explicit **singing** detection, and produces
> per-clip JSON plus a self-contained HTML report. See **[docs/default_pipeline.md](docs/default_pipeline.md)**
> for model weights, setup and run instructions. The legacy triple-ASR ensemble (VibeVoice +
> Parakeet + Qwen3) is still available as an option.
>
> ```bash
> cd default_pipeline && bash setup_environments.sh ./envs
> export UAAP_MOSS_SRC="$(pwd)/envs/MOSS-Audio"
> bash run_all.sh --audio /path/to/clips --workdir ./uaap_work --envs ./envs
> ```

### 🔗 Links

- **Live predictions (example report):** https://laion-ai.github.io/univeral-audio-annotation-pipeline/predictions/
- **Model mirror on Hugging Face** (all weights + code, self-contained): https://huggingface.co/laion/universal-audio-annotation-pipeline

## Pipeline Architecture

The **default configuration** pairs Nemotron 3.5 (words) with VibeVoice + Sortformer
(diarization/timing), the Whisper voice experts, and two specialist sound-event pre-passes,
fused by a single MOSS-Audio reasoning stage:

```
┌───────────────────────────────────────────────────────────────────────┐
│                    INPUT: Audio File (any length)                      │
└───────────────────────────────────┬───────────────────────────────────┘
                                     │
        ┌──────────────────────────┴──────────────────┐
        ▼                                              ▼
  ┌──────────────┐                          ┌────────────────────┐
  │ VibeVoice    │                          │ Nemotron 3.5 ASR + │
  │ ASR          │                          │ Sortformer         │
  │ (diarization │                          │ (words + secondary │
  │  & timing    │                          │  diarization)      │
  │  authority)  │                          │                    │
  └──────┬───────┘                          └─────────┬──────────┘
         │   diarization / timing      words / what is said
         └──────────────┬──────────────────────┬───────┘
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
        │  Nemotron words on VibeVoice timeline         │
        │  diarization-only speakers · emotion & style  │
        │  DETAILED sound-event + dedicated music caps  │
        │  singing flag · full-timeline coverage hint   │
        └───────────────────────┬──────────────────────┘
                                │  + deterministic gap-fill
                                ▼     (non-speech background only)
        ┌──────────────────────────────────────────────┐
        │   OUTPUT: Structured JSON (covers full clip)  │
        │   [speech · vocal_burst · sound_event · music]│
        └──────────────────────────────────────────────┘
```

## Output Format

The pipeline produces a JSON array of segment dictionaries. Four segment types:

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

### Music segment
```json
{
  "type": "music",
  "start_time": 9.80,
  "end_time": 15.10,
  "description": "Upbeat acoustic folk-pop: brightly strummed steel-string guitar and light tambourine, mid-tempo ~110 BPM, major key, warm and nostalgic mood, no vocals",
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
  --config nemotron_vibevoice \
  --gpus 0
```

### Configurations

| Config | ASR (words / diarization) | Notes |
|--------|---------------------------|-------|
| `nemotron_vibevoice` ⭐ **default** | Nemotron 3.5 words / VibeVoice + Sortformer diarization | Best on SoundScape-Bench (Reward **0.236**); detailed sound-event + `music` captions |
| `triple_greedy` (legacy ensemble) | VibeVoice + Parakeet + Qwen3 | Previous default; top on the older 60-scene LLM-judged eval (4.13) |
| `ensemble_greedy` | Parakeet + Qwen3 | Dual ASR |
| `vibevoice` | VibeVoice only | Single ASR |

⭐ **`nemotron_vibevoice` is the default.** The turnkey
[`default_pipeline/`](default_pipeline/) runs it with the SFX LoRA sound-event detector,
the vocal-burst locator (threshold 0.7) + sound-effect captioner, expressive emotion &
speaking-style captions, detailed sound-effect and dedicated `music` captions,
diarization-only speaker identity, singing detection, and guaranteed full-timeline coverage.
See [docs/default_pipeline.md](docs/default_pipeline.md).

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

The default `nemotron_vibevoice` config drops Parakeet and Qwen3 (using Nemotron 3.5 +
Sortformer for words instead), so it is **lighter** than the legacy triple ensemble: it needs
VibeVoice (~16 GB) + Nemotron 3.5/Sortformer (~5 GB) + Whisper experts + SFX LoRA + MOSS,
loaded/unloaded sequentially. With multi-GPU, components can run in parallel on separate GPUs.

> The SFX LoRA `laion/moss-audio-sfx-lora-v4` is gated — request access and `huggingface-cli login`,
> or run with `--no-sfx`. Full model table with links: [docs/default_pipeline.md](docs/default_pipeline.md).

## Evaluation Results

### SoundScape-Bench (200 multilingual soundscapes, automatic answer-key scoring)

The current default was selected on **[SoundScape-Bench](https://huggingface.co/datasets/laion/soundscape-bench)** — 200
held-out soundscapes (EN/ZH/FR/DE/ES/NL, ~25 % overlapping speech) built from understood pieces so
every event has an exact answer key. The headline **Reward** = IoU(timing) × content, where content is
a weighted mix of caption cosine and (1 − WER) for speech, averaged over all answer-key events.

| # | System | Reward | IoU | F1 | WER | sound/music cos |
|---|--------|--------|-----|-----|-----|-----------------|
| 1 | Gemini 3.1 Pro (omni) | 0.297 | 0.615 | 0.270 | 72 % | 0.385 |
| 2 | Gemini 3.5 Flash (omni) | 0.256 | 0.556 | 0.233 | 67 % | 0.310 |
| 3 | **UAAP `nemotron_vibevoice` ⭐ (this default)** | **0.236** | 0.457 | 0.191 | 65 % | 0.287 |
| 4 | Gemini 3 Flash (omni) | 0.212 | 0.450 | 0.172 | 66 % | 0.262 |
| 5 | UAAP triple-ASR ensemble (legacy default) | 0.196 | 0.388 | 0.145 | 66 % | 0.226 |
| 6 | UAAP Sortformer + Nemotron 3.5 (no VibeVoice diar) | 0.153 | 0.331 | 0.112 | 59 % | 0.223 |
| 7 | GPT-Audio 1.5 (omni) | 0.097 | 0.223 | 0.097 | 61 % | 0.152 |

The default pipeline is the **strongest non-Gemini system and the best UAAP configuration** — it beats
the legacy triple ensemble by +20 % Reward and edges out Gemini 3 Flash. Two changes drive the gain:
emitting a dedicated **`music`** type (the legacy configs emit `0` music events, so every music answer
scored zero) and asking MOSS for **detailed sound-effect captions**.

### Legacy LLM-judged eval (60 scenes, Gemini 3.1 Pro, 0–5 scale)

Earlier configuration sweep that selected the previous triple-ASR default:

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
    stage1a_vibevoice.py #   VibeVoice-ASR (diarization / timing authority)
    stage1_nemotron_sortformer.py #  ⭐ Nemotron 3.5 + Sortformer (default word source)
    stage1b_parakeet.py  #   Parakeet TDT v3 + Sortformer (legacy ensemble option)
    stage1c_qwen3.py     #   Qwen3-ASR + ForcedAligner (legacy ensemble option)
    stage2_whisper_experts.py  # emotion/timbre/style
    stage3_sfx_lora.py   #   SFX LoRA sound events
    stage3b_vocalburst.py#   Vocal-burst locator + captioner
    stage4_moss_annotator.py   # MOSS final annotation (greedy; auto-detects ASR JSONs)
  build_report.py        # Self-contained HTML report

pipeline/
  run_pipeline.py        # Main entry point (single-process reference implementation)
  asr_vibevoice.py       # VibeVoice-ASR component
  asr_nemotron.py        # ⭐ Nemotron 3.5 + Sortformer (default word source)
  asr_parakeet.py        # Parakeet TDT v3 + Sortformer (legacy ensemble option)
  asr_qwen3.py           # Qwen3-ASR-1.7B + ForcedAligner (legacy ensemble option)
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
