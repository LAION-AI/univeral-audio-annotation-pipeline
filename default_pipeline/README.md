# `default_pipeline/` — the default, recommended configuration

Driver scripts that run the default configuration end-to-end and produce structured JSON
annotations + a self-contained HTML report.

**Pipeline (`nemotron_vibevoice`, default):** VibeVoice-ASR (diarization / timing authority)
+ Nemotron 3.5/Sortformer (words) → Whisper experts (emotion/timbre/style) → SFX LoRA sound
events → MOSS-Audio-8B-Thinking final annotation, with **expressive emotion & speaking-style
captions**, **detailed sound-effect captions and a dedicated `music` segment type**, strict
diarization-only speaker identity, and explicit **singing** detection. This is the
best-scoring configuration on [SoundScape-Bench](../README.md#evaluation-results).

The legacy **triple-ASR ensemble** (VibeVoice + Parakeet + Qwen3) remains available — just run
`workers/stage1b_parakeet.py` and `workers/stage1c_qwen3.py` instead of the Nemotron stage;
`stage4` auto-detects which ASR JSONs are present (`nemotron.json` → default path, else triple).

📖 Full documentation, model weights and links: [`../docs/default_pipeline.md`](../docs/default_pipeline.md)

## Quickstart

```bash
bash setup_environments.sh ./envs                 # build the 4 venvs + clone sources
export UAAP_MOSS_SRC="$(pwd)/envs/MOSS-Audio"
huggingface-cli login                             # for the gated SFX LoRA (or use --no-sfx)

bash run_all.sh --audio /path/to/clips --workdir ./uaap_work --envs ./envs
```

Results: `<audio>_pred.json` next to each input, all intermediates in `uaap_work/<stem>/`,
and `uaap_work/report.html`.

## Files

| File | Purpose |
|------|---------|
| `setup_environments.sh` | Create the 4 isolated venvs (ASR packages need conflicting `transformers`) |
| `run_all.sh` | Orchestrate all stages in order, each in its own venv |
| `prepare_audio.py` | Stage 0 — decode inputs to 24 kHz mono WAV, build `index.json` |
| `workers/stage1a_vibevoice.py` | VibeVoice-ASR — diarization / timing authority (sharded across GPUs) |
| `workers/stage1_nemotron_sortformer.py` | ⭐ Nemotron 3.5 + Sortformer — **default word source** |
| `workers/stage1b_parakeet.py` | Parakeet TDT v3 + Sortformer (legacy ensemble option) |
| `workers/stage1c_qwen3.py` | Qwen3-ASR + ForcedAligner (legacy ensemble option) |
| `workers/stage2_whisper_experts.py` | Emotion / timbre / style Whisper models |
| `workers/stage3_sfx_lora.py` | SFX LoRA sound-event detection (optional, gated) |
| `workers/stage3b_vocalburst.py` | Vocal-burst locator (thr 0.7) + sound-effect captioner |
| `workers/stage4_moss_annotator.py` | MOSS-Audio-8B-Thinking final annotation (greedy) |
| `build_report.py` | Self-contained HTML report (base64 audio + predictions) |
| `workers/_common.py` | Shared helpers (flash-attn guard, IO, repo path) |

Each stage reads/writes JSON in the per-clip work dir, so stages can be re-run
individually (e.g. re-run `stage4` after editing the MOSS prompt).
