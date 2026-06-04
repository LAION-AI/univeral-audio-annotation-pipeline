# Default Pipeline — Triple-ASR Ensemble (recommended configuration)

This is the **default, recommended configuration** of the Universal Audio Annotation
Pipeline. It runs the full triple-ASR ensemble with greedy decoding, the Whisper
expert voice models, the SFX LoRA sound-event detector, and the MOSS-Audio-8B-Thinking
annotator, and emits one structured JSON annotation per clip plus a self-contained HTML
report. All driver scripts live in [`default_pipeline/`](../default_pipeline).

It corresponds to the **Triple-ASR greedy** row in the evaluation table (combined
score **4.13/5.00**) and additionally enables the SFX LoRA stage.

---

## Models used (with weights)

| Stage | Model | HuggingFace | ~Disk | Runtime VRAM |
|-------|-------|-------------|-------|--------------|
| ASR 1 | VibeVoice-ASR | [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) · [code](https://github.com/microsoft/VibeVoice) | 17 GB | ~23 GB (sharded over 2 GPUs) |
| ASR 2 | Parakeet TDT v3 | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 2.4 GB | ~4 GB |
| ASR 2 | Sortformer diarizer | [nvidia/diar_sortformer_4spk-v1](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) | 0.5 GB | ~2 GB |
| ASR 3 | Qwen3-ASR-1.7B | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 3.4 GB | ~8 GB |
| ASR 3 | Qwen3 ForcedAligner | [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | 1.3 GB | ~3 GB |
| Voice | BUD-E-Whisper (emotion) | [laion/BUD-E-Whisper](https://huggingface.co/laion/BUD-E-Whisper) | 0.9 GB | ~0.7 GB |
| Voice | Timbre Whisper | [laion/timbre-whisper](https://huggingface.co/laion/timbre-whisper) | 0.9 GB | ~0.7 GB |
| Voice | Voice-tagging Whisper | [laion/voice-tagging-whisper](https://huggingface.co/laion/voice-tagging-whisper) | 0.9 GB | ~0.7 GB |
| SFX | MOSS-Audio-8B-Instruct | [OpenMOSS-Team/MOSS-Audio-8B-Instruct](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct) | 18 GB | ~18 GB |
| SFX | SFX LoRA adapter ⚠️ gated | [laion/moss-audio-sfx-lora-v4](https://huggingface.co/laion/moss-audio-sfx-lora-v4) | 0.7 GB | (merged into base) |
| VB | Vocal-burst locator | [laion/vocalburst-locator](https://huggingface.co/laion/vocalburst-locator) | 1.0 GB | ~1 GB |
| VB | Sound-effect captioner | [laion/sound-effect-captioning-whisper](https://huggingface.co/laion/sound-effect-captioning-whisper) | 1.0 GB | ~1 GB |
| Final | MOSS-Audio-8B-Thinking | [OpenMOSS-Team/MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Thinking) | 18 GB | ~18 GB |
| Source | MOSS-Audio code (`src.*`) | [github.com/OpenMOSS/MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio) | — | — |

> ⚠️ **`laion/moss-audio-sfx-lora-v4` is gated.** Request access on the model page and
> `huggingface-cli login` with a token that has access. Without it, run with `--no-sfx`
> (MOSS still detects sound events directly from the audio).

Models load and unload **sequentially**, so peak disk/VRAM is roughly one large model
at a time (~18–23 GB). Two 24 GB GPUs are recommended (VibeVoice-ASR is sharded across
both); the rest fit on a single 24 GB GPU.

---

## Why several virtual-envs?

The three ASR packages pin **mutually incompatible** `transformers` versions:

| venv | key package | transformers |
|------|-------------|--------------|
| `venv`     | base (Whisper experts, SFX, MOSS) | 4.57.x |
| `venv_vv`  | `vibevoice` (+ GitHub ASR source) | 4.51.3 |
| `venv_qwen`| `qwen-asr`  | 4.57.6 |
| `venv_nemo`| `nemo-toolkit[asr]` | its own pins |

So each stage runs as its own process in its own venv and the stages exchange small JSON
files in a per-clip work directory. `setup_environments.sh` builds all four.

---

## Setup

```bash
cd default_pipeline
bash setup_environments.sh ./envs          # creates envs/ + clones VibeVoice & MOSS-Audio source
export UAAP_MOSS_SRC="$(pwd)/envs/MOSS-Audio"
huggingface-cli login                      # needed for the gated SFX LoRA
```

If you already have a known-good CUDA `torch`, reuse it instead of reinstalling:
```bash
BASE_TORCH_SITE=/path/to/your/site-packages bash setup_environments.sh ./envs
```

## Run

```bash
# a directory of clips (or a single file)
bash run_all.sh --audio /path/to/clips --workdir ./uaap_work --envs ./envs

# without the gated SFX stage
bash run_all.sh --audio /path/to/clips --workdir ./uaap_work --envs ./envs --no-sfx
```

Outputs:
- `<audio>_pred.json` next to every input file — the final structured annotation.
- `uaap_work/<stem>/*.json` — every intermediate stage output.
- `uaap_work/report.html` — self-contained report (base64 audio + predictions + explanation).

Run a single stage manually (e.g. to re-annotate after a prompt change):
```bash
./envs/venv/bin/python workers/stage4_moss_annotator.py ./uaap_work
```

---

## What this configuration changes vs. the stock prompt

The MOSS annotator prompt (`pipeline/moss_annotator.py`) was tuned for this configuration:

1. **Expressive emotion captions.** `emotion` is a short, precise *caption* (a few words
   up to ~10–12) built from EmoNet voice-taxonomy words, with intensity modifiers
   (*barely / slight / clearly / intensely / extreme*) and emotion blends — e.g.
   *"clearly intense anger laced with wounded disappointment"* — instead of a single
   label + numeric intensity.
2. **Expressive speaking-style captions.** `speaking_style` is captioned just as vividly
   (*"low conspiratorial whisper"*, *"euphoric manic rant"*, *"booming drill-sergeant bark"*).
3. **Strict speaker identity.** The number of speakers and speaker IDs come **only** from
   the diarization, with priority: **VibeVoice's built-in labels first**, Sortformer
   (Parakeet/Qwen3) as fallback. Voice-analysis notes and SFX captions that mention
   "another voice" must **never** create an extra speaker.
4. **Singing detection.** If MOSS hears singing (melody / sustained pitches / musical
   rhythm) it states so explicitly in `speaking_style`, even when the Whisper voice-tagging
   expert reports ordinary talking.
5. **Explicit ASR voting.** The three ASR transcripts vote on the words: when two agree they
   outvote the third (with VibeVoice's rendering preferred when it is in the majority); content
   that only one ASR caught is still assumed real and included.
6. **Full-timeline coverage + background at all times.** The clip duration is given to MOSS, which
   must cover the entire span `[0, duration]` with no gaps and emit a background `sound_event` for
   every moment (describing the background, or explicit silence/room-tone when quiet). A deterministic
   backstop (`utils.fill_timeline_gaps`) then fills any span MOSS still left uncovered using the SFX
   LoRA predictions (else silence), tagged `source: "timeline_fill"`. **These fallback fillers are
   sanitized to NON-SPEECH only** — any clause describing a speaker or what was said is stripped — so a
   fallback can never invent a speaker or speech; speaker identity and spoken content come solely from
   the ASR models, the Whisper experts and MOSS.
7. **Vocal-burst candidates.** A specialist pre-pass (the vocal-burst locator at threshold 0.7,
   with >30 s clips windowed and merged, plus the sound-effect captioner) proposes extra
   timestamped sound effects. They are handed to MOSS **as candidates to verify** — MOSS keeps
   one only if it can actually hear it (they may be false positives), and is *not* told they are
   "vocal bursts".

## Things to consider

- **flash-attn / GLIBC.** If a host has a `flash_attn` wheel built against a newer GLIBC
  than the system, transformers crashes on import. The workers neutralise this
  (`sys.modules["flash_attn"] = None`) and fall back to SDPA attention.
- **VibeVoice size.** VibeVoice-ASR (~23 GB) does not fit on one 24 GB GPU; it is loaded
  with `device_map="auto"` and the modeling code was patched to keep its speech-feature
  insertion device-safe when sharded.
- **Decoding is greedy** (`do_sample=False`) for ASR reconciliation and the final
  annotation. The SFX LoRA detection step samples (as in the original component).
- **Languages.** Qwen3-ASR is invoked with `language="English"` by default; change it in
  `workers/stage1c_qwen3.py` for other languages.

See also [pipeline_details.md](pipeline_details.md) and [evaluation_results.md](evaluation_results.md).
