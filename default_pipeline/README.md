# `default_pipeline/` — the default, recommended configuration

Driver scripts that run the default configuration end-to-end and produce structured JSON
annotations + a self-contained HTML report.

**Pipeline (default, `--fusion gemma`):** VibeVoice-ASR (diarization/timing) + Nemotron 3.5/Sortformer
(words) + **pyannote** (overlap detection) + **DiCoW** (overlap-aware per-speaker ASR) → Whisper experts
(emotion/timbre/style) → SFX LoRA sound events → **Gemma-4-12B text-only fusion** (reads only the
experts' outputs, no audio). Produces **expressive emotion & speaking-style captions**, **detailed
sound-effect captions + a dedicated `music` segment type**, diarization-only speaker identity,
**overlapping-speech** segments, and **singing** detection. Highest-Reward configuration on
[SoundScape-Bench](../README.md#evaluation-results) (0.253).

**Final-stage choice:** `--fusion gemma` (default, best Reward, text-only) or `--fusion moss` (legacy
MOSS-Audio-8B, audio-grounded, higher precision/F1 — `--fusion moss` skips the pyannote/DiCoW/Gemma
stages and runs the MOSS annotator). The legacy **triple-ASR ensemble** (VibeVoice + Parakeet + Qwen3)
is also available by running `stage1b_parakeet.py` + `stage1c_qwen3.py` instead of the Nemotron stage.

📖 Full documentation, model weights and links: [`../docs/default_pipeline.md`](../docs/default_pipeline.md)

## Quickstart

```bash
bash setup_environments.sh ./envs                 # build all venvs + clone sources
export HF_TOKEN=...                               # gated: pyannote/* + laion sfx-lora (or --no-sfx)

bash run_all.sh --audio /path/to/clips --workdir ./uaap_work --envs ./envs   # --fusion gemma (default)
#   --fusion moss   # legacy audio MOSS-Audio-8B annotator (also: export UAAP_MOSS_SRC=.../MOSS-Audio)
```

Results: `<audio>_pred.json` next to each input, all intermediates in `uaap_work/<stem>/`,
and `uaap_work/report.html`.

## Files

| File | Purpose |
|------|---------|
| `setup_environments.sh` | Create the isolated venvs (ASR/llama.cpp/pyannote pin conflicting deps) |
| `run_all.sh` | Orchestrate all stages in order, each in its own venv (`--fusion gemma`/`moss`) |
| `prepare_audio.py` | Stage 0 — decode inputs to 24 kHz mono WAV, build `index.json` |
| `workers/stage1a_vibevoice.py` | VibeVoice-ASR — diarization / timing authority (sharded across GPUs) |
| `workers/stage1_nemotron_sortformer.py` | Nemotron 3.5 + Sortformer — **default word source** |
| `workers/stage_pyannote_diar.py` | ⭐ pyannote diarization + overlap detection (default) |
| `workers/stage_dicow.py` | ⭐ DiCoW diarization-conditioned overlap-aware per-speaker ASR (default) |
| `workers/stage5_gemma_fusion.py` | ⭐ Gemma-12B **TEXT-only** fusion — **default final stage** |
| `workers/stage1b_parakeet.py` | Parakeet TDT v3 + Sortformer (legacy ensemble option) |
| `workers/stage1c_qwen3.py` | Qwen3-ASR + ForcedAligner (legacy ensemble option) |
| `workers/stage2_whisper_experts.py` | Emotion / timbre / style Whisper models |
| `workers/stage3_sfx_lora.py` | SFX LoRA sound-event detection (optional, gated) |
| `workers/stage3b_vocalburst.py` | Vocal-burst locator (thr 0.7) + sound-effect captioner |
| `workers/stage4_moss_annotator.py` | MOSS-Audio-8B-Thinking final annotation (legacy, `--fusion moss`) |
| `build_report.py` | Self-contained HTML report (base64 audio + predictions) |
| `workers/_common.py` | Shared helpers (flash-attn guard, IO, repo path) |

Each stage reads/writes JSON in the per-clip work dir, so stages can be re-run
individually (e.g. re-run `stage4` after editing the MOSS prompt).
