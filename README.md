# Universal Audio Annotation Pipeline

Produces structured JSON annotations from any audio file, covering speech transcription, speaker diarization, emotions, vocal bursts, sound effects, and music.

**Best configuration: Gemma-12B + DiCoW** — Nemotron 3.5 words + VibeVoice/Sortformer diarization + **DiCoW** overlap-aware ASR, fused by a **text-only Gemma-4-12B** LLM (no audio in the final step). It is the **highest-Reward pipeline on [SoundScape-Bench](#evaluation-results) (0.253)** — rank 3 of all systems, nearly matching Gemini 3.5 Flash (0.256) and ahead of every other pipeline. (It trades precision for that recall: see the [tradeoff note](#evaluation-results).)

> ### ⭐ Recommended: the default pipeline
> A turnkey, end-to-end implementation lives in **[`default_pipeline/`](default_pipeline/)** — the
> **default, suggested configuration**. It uses **Nemotron 3.5** for the words, **VibeVoice + Sortformer**
> for diarization/timing, **pyannote + DiCoW** to transcribe overlapping speakers separately, the Whisper
> experts and the SFX LoRA sound-event detector — all fused by a **text-only Gemma-4-12B** model (it reads
> only the experts' outputs, not the audio). It produces **expressive emotion & speaking-style captions**,
> **detailed sound-effect captions and a dedicated `music` segment type**, strict diarization-only speaker
> identity, **overlapping-speech** segments, and explicit **singing** detection, plus per-clip JSON and a
> self-contained HTML report. The legacy audio **MOSS-Audio-8B** annotator (higher precision/F1) remains
> available via `--fusion moss`, as does the triple-ASR ensemble. See
> **[docs/default_pipeline.md](docs/default_pipeline.md)** for models, setup and run instructions.
>
> ```bash
> cd default_pipeline && bash setup_environments.sh ./envs
> export HF_TOKEN=...        # token with gated pyannote + SFX-LoRA access accepted
> bash run_all.sh --audio /path/to/clips --workdir ./uaap_work --envs ./envs      # --fusion gemma (default)
> ```
> ⚡ With ≥2 GPUs the heavy stages (Gemma fusion, SFX, ASR…) are **auto-sharded across GPUs over
> disjoint clips** (≈N× faster, identical output; `--gpus 0,1`). Optional quality-neutral lower-VRAM
> fuser: `export GEMMA_FILE=gemma-4-12b-it-UD-Q6_K_XL.gguf`. See [efficiency notes](default_pipeline/README.md#performance--efficiency).
>
> **🎧 Live demo (20 samples — audio + predictions vs ground truth):**
> [demo](https://laion-ai.github.io/univeral-audio-annotation-pipeline/gemma12_dicow_demo.html) ·
> **📊 Full model comparison:**
> [comparison](https://laion-ai.github.io/univeral-audio-annotation-pipeline/soundscape_comparison.html)

### 🔗 Links

- **Live predictions (example report):** https://laion-ai.github.io/univeral-audio-annotation-pipeline/predictions/
- **Model mirror on Hugging Face** (all weights + code, self-contained): https://huggingface.co/laion/universal-audio-annotation-pipeline

## Pipeline Architecture

The **default configuration** combines Nemotron 3.5 (words), VibeVoice + Sortformer
(diarization/timing), pyannote + DiCoW (overlap-aware per-speaker ASR), the Whisper voice experts
and the specialist sound-event pre-passes — all fused by a **text-only Gemma-4-12B** stage that reads
the experts' outputs (not the audio):

```
┌───────────────────────────────────────────────────────────────────────┐
│                    INPUT: Audio File (any length)                      │
└───────────────────────────────────┬───────────────────────────────────┘
        ┌──────────────┬─────────────┴───────────┬───────────────────┐
        ▼              ▼                          ▼                   ▼
  ┌──────────┐  ┌────────────────┐     ┌──────────────┐   ┌────────────────────┐
  │ VibeVoice│  │ Nemotron 3.5 + │     │ pyannote     │   │ DiCoW              │
  │ ASR      │  │ Sortformer     │     │ diarization  │──▶│ diarization-cond.  │
  │ (diar/   │  │ (words)        │     │ + overlap    │   │ Whisper (per-spk,  │
  │  timing) │  │                │     │ detection    │   │ overlap-aware ASR) │
  └────┬─────┘  └───────┬────────┘     └──────┬───────┘   └─────────┬──────────┘
       │ timing/spk     │ words               │ overlaps            │ overlapped speech
       └────────┬───────┴─────────────┬───────┴──────────┬─────────┘
            ▼                      ▼                  ▼
  ┌──────────────────────┐  ┌──────────────────────────────┐
  │ Whisper experts (x3) │  │ Specialist sound-event prepass│
  │ emotion · timbre ·   │  │ • SFX LoRA (MOSS-8B-Instruct  │
  │ speaking-style        │  │   + laion sfx-lora r=128)     │
  │ (per utterance)      │  │ • Vocal-burst locator @0.7    │
  └──────────┬───────────┘  └───────────────┬──────────────┘
             └───────────────┬──────────────┘
                             ▼
        ┌──────────────────────────────────────────────┐
        │     Gemma-4-12B — TEXT-ONLY fusion (no audio) │
        │  fuses all expert text → final annotation     │
        │  Nemotron words on VibeVoice timeline         │
        │  DiCoW recovers overlapping/simultaneous spk  │
        │  detailed sound-event + dedicated music caps  │
        │  (legacy: MOSS-Audio-8B, audio, --fusion moss)│
        └───────────────────────┬──────────────────────┘
                                │  + deterministic gap-fill
                                ▼
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
# DEFAULT pipeline (Gemma-12B + DiCoW, text-only fusion): use the staged runner
cd default_pipeline
bash run_all.sh --audio input.wav --workdir ./uaap_work --envs ./envs       # --fusion gemma (default)
```

The default **Gemma-12B + DiCoW** configuration runs as the staged
[`default_pipeline/`](default_pipeline/) (each stage in its own venv — llama.cpp, pyannote, DiCoW and
the ASR toolkits pin incompatible deps, so they can't share one process). The single-process
`pipeline/run_pipeline.py` below is a **legacy reference** for the MOSS-Audio configurations only.

### Configurations

| Config | Final fusion · ASR | SoundScape-Bench Reward | Notes |
|--------|--------------------|------------------------:|-------|
| **Gemma-12B + DiCoW** ⭐ **default** | Gemma-4-12B *text-only* · Nemotron+VibeVoice/Sortformer + DiCoW overlap ASR | **0.253** | Best Reward of any pipeline; overlap-aware; trades precision (see eval) |
| `nemotron_vibevoice` (MOSS, `--fusion moss`) | MOSS-Audio-8B *audio* · Nemotron+VibeVoice/Sortformer | 0.236 | Most precise (best F1/lowest hallucination); audio-grounded |
| `triple_greedy` (legacy ensemble) | MOSS-Audio-8B · VibeVoice+Parakeet+Qwen3 | 0.196 | Top on the older 60-scene LLM-judged eval (4.13) |
| `ensemble_greedy` / `vibevoice` | MOSS-Audio-8B · dual / single ASR | — | Reduced ASR variants |

⭐ **Gemma-12B + DiCoW is the default** (`bash run_all.sh ... --fusion gemma`). It adds pyannote
diarization + DiCoW overlap-aware ASR, then fuses every expert's *text* output with Gemma-4-12B (no
audio in the final step) — plus the SFX LoRA detector, vocal-burst locator (0.7) + captioner, emotion/
style captions, dedicated `music` segments, overlapping-speech segments and full-timeline coverage.
Use `--fusion moss` for the audio MOSS-Audio-8B annotator (higher precision). See
[docs/default_pipeline.md](docs/default_pipeline.md).

## Model Requirements

| Model | HuggingFace ID | GPU VRAM | Role |
|-------|---------------|----------|------|
| VibeVoice-ASR | `microsoft/VibeVoice-ASR` | ~16 GB | diarization/timing (default) |
| Nemotron 3.5 ASR | `nvidia/nemotron-3.5-asr-streaming-0.6b` | ~3 GB | words (default) |
| Sortformer | `nvidia/diar_sortformer_4spk-v1` | ~2 GB | diarization (default) |
| **pyannote** | `pyannote/speaker-diarization-3.1` (+ `segmentation-3.0`, gated) | ~2 GB | overlap detection (default) |
| **DiCoW** | `BUT-FIT/DiCoW_v3_3` | ~3 GB | overlap-aware per-speaker ASR (default) |
| **Gemma fuser** | `unsloth/gemma-4-12b-it-GGUF` (Q8, llama.cpp) | ~14 GB | TEXT-only final fusion (default) |
| Whisper experts (x3) | `laion/BUD-E-Whisper`, `laion/timbre-whisper`, `laion/voice-tagging-whisper` | ~2 GB | emotion/timbre/style |
| SFX LoRA | `OpenMOSS-Team/MOSS-Audio-8B-Instruct` + `laion/moss-audio-sfx-lora-v4` (gated) | ~18 GB | sound events |
| Vocal-burst locator + captioner | `laion/vocalburst-locator`, `laion/sound-effect-captioning-whisper` | ~2 GB | vocal bursts |
| Parakeet / Qwen3 | `nvidia/parakeet-tdt-0.6b-v3`, `Qwen/Qwen3-ASR-1.7B` (+aligner) | ~12 GB | legacy ensemble only |
| MOSS Annotator | `OpenMOSS-Team/MOSS-Audio-8B-Thinking` | ~18 GB | legacy final stage (`--fusion moss`) |

Models load/unload sequentially (each stage in its own venv), so peak VRAM ≈ the largest single stage
(~16–18 GB); the default runs comfortably on one 24 GB GPU, faster on two.

> **Gated models** (request access, then `export HF_TOKEN=...`): `pyannote/segmentation-3.0`,
> `pyannote/speaker-diarization-3.1`, and `laion/moss-audio-sfx-lora-v4` (or run `--no-sfx`).
> The Gemma GGUF downloads automatically. Full model table with links:
> [docs/default_pipeline.md](docs/default_pipeline.md).

## Evaluation Results

### SoundScape-Bench (200 multilingual soundscapes, automatic answer-key scoring)

The current default was selected on **[SoundScape-Bench](https://huggingface.co/datasets/laion/soundscape-bench)** — 200
held-out soundscapes (EN/ZH/FR/DE/ES/NL, ~25 % overlapping speech) built from understood pieces so
every event has an exact answer key. The headline **Reward** = IoU(timing) × content, where content is
a weighted mix of caption cosine and (1 − WER) for speech, averaged over all answer-key events.

| # | System | Reward | IoU | F1 | WER | snd | halluc |
|---|--------|-------:|----:|---:|----:|----:|-------:|
| 1 | Gemini 3.1 Pro (omni) | 0.297 | 0.615 | 0.270 | 72 % | 0.385 | 23 % |
| 2 | Gemini 3.5 Flash (omni) | 0.256 | 0.556 | 0.233 | 67 % | 0.310 | 23 % |
| 3 | **UAAP Gemma-12B + DiCoW ⭐ (this default, text-only fusion)** | **0.253** | 0.515 | 0.149 | 59 % | 0.273 | 43 % |
| 4 | UAAP Gemma-12B (text-only, no DiCoW) | 0.248 | 0.512 | 0.144 | 56 % | 0.278 | 44 % |
| 5 | UAAP Gemma-4B + DiCoW (text-only) | 0.244 | 0.490 | 0.151 | 59 % | 0.269 | 44 % |
| 6 | UAAP `nemotron_vibevoice` (MOSS-Audio-8B, `--fusion moss`) | 0.236 | 0.457 | **0.191** | 65 % | 0.287 | **27 %** |
| 7 | Gemini 3 Flash (omni) | 0.212 | 0.450 | 0.172 | 66 % | 0.262 | 33 % |
| 8 | UAAP triple-ASR ensemble (legacy) | 0.196 | 0.388 | 0.145 | 66 % | 0.226 | 32 % |
| 9 | GPT-Audio 1.5 (omni) | 0.097 | 0.223 | 0.097 | 61 % | 0.152 | 36 % |

The default **Gemma-12B + DiCoW** is the **highest-Reward pipeline of all** (rank 3 overall, nearly
matching Gemini 3.5 Flash and well above the audio MOSS configs). A *text-only* LLM that faithfully
fuses strong experts wins on **timing (IoU) and transcription (WER)**, and **DiCoW** recovers
overlapping/simultaneous speakers (cleanly separating e.g. an English and a Mandarin speaker talking
at once).

> **⚖️ Precision tradeoff (read before deploying):** because the Gemma fuser has **no audio**, it
> cannot *reject* candidate events it can't hear, so it over-generates — **hallucination ~43 % and
> F1 ~0.15**, versus the audio **MOSS-Audio-8B** config's **27 % / 0.191**. If you need maximum
> precision / fewest spurious events, run `--fusion moss`. The default optimizes the recall-oriented
> Reward at lower cost (a ~12B text model, no audio in the final step).

📊 **Interactive comparison:** [soundscape_comparison.html](https://laion-ai.github.io/univeral-audio-annotation-pipeline/soundscape_comparison.html)
· 🎧 **20-sample audio demo (predictions vs ground truth):** [gemma12_dicow_demo.html](https://laion-ai.github.io/univeral-audio-annotation-pipeline/gemma12_dicow_demo.html)

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
    stage1_nemotron_sortformer.py #  Nemotron 3.5 + Sortformer (default word source)
    stage_pyannote_diar.py #  ⭐ pyannote diarization + overlap detection (default)
    stage_dicow.py       #   ⭐ DiCoW diarization-conditioned overlap ASR (default)
    stage5_gemma_fusion.py #  ⭐ Gemma-12B TEXT-only fusion — DEFAULT final stage
    stage1b_parakeet.py  #   Parakeet TDT v3 + Sortformer (legacy ensemble option)
    stage1c_qwen3.py     #   Qwen3-ASR + ForcedAligner (legacy ensemble option)
    stage2_whisper_experts.py  # emotion/timbre/style
    stage3_sfx_lora.py   #   SFX LoRA sound events
    stage3b_vocalburst.py#   Vocal-burst locator + captioner
    stage4_moss_annotator.py   # MOSS-Audio final annotation (legacy, --fusion moss)
  build_report.py        # Self-contained HTML report

pipeline/
  run_pipeline.py        # Single-process reference runner (MOSS configs only)
  asr_vibevoice.py       # VibeVoice-ASR component
  asr_nemotron.py        # Nemotron 3.5 + Sortformer (default word source)
  diarize_pyannote.py    # ⭐ pyannote diarization + overlap detection
  asr_dicow.py           # ⭐ DiCoW diarization-conditioned overlap ASR
  gemma_fusion.py        # ⭐ Gemma text-only LLM fusion (default final stage)
  asr_parakeet.py        # Parakeet TDT v3 + Sortformer (legacy ensemble option)
  asr_qwen3.py           # Qwen3-ASR-1.7B + ForcedAligner (legacy ensemble option)
  whisper_experts.py     # Emotion/timbre/style Whisper models
  sfx_lora.py            # LoRA SFX sound event detection
  vocalburst_locator.py  # Vocal-burst locator + sound-effect captioner
  moss_annotator.py      # MOSS-Audio-8B-Thinking annotation (legacy final stage)
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
