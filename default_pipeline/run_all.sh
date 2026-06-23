#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the full default pipeline over a file or directory of audio clips.
#
# Stages run sequentially (each model is loaded once, processes all clips, then
# frees the GPU before the next stage). Every stage uses its dedicated venv.
#
# Usage:
#   export HF_TOKEN=...                              # for gated pyannote (default fusion=gemma)
#   export UAAP_MOSS_SRC=/path/to/MOSS-Audio        # only needed for --fusion moss
#   bash run_all.sh --audio /path/to/clips --workdir ./uaap_work --envs ./envs [--no-sfx] [--fusion gemma|moss]
#
# Final-stage fusion:
#   --fusion gemma    (DEFAULT) Gemma-4-12B text-only fusion + DiCoW overlap ASR (best Reward on
#                     SoundScape-Bench; no audio in the final step). Adds pyannote + DiCoW + Gemma stages.
#   --fusion moss     legacy MOSS-Audio-8B-Thinking annotator (audio-grounded; higher precision/F1).
#   --fusion dramabox Gemma-4-E4B-it DramaBox prompt generator (outputs DramaBox prompts, not JSON).
#                     Adds speaker embedding stage + DramaBox fusion. Requires stage 2b.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AUDIO=""; WORKDIR="./uaap_work"; ENVS="./envs"; SFX=1; FUSION="gemma"
while [[ $# -gt 0 ]]; do case "$1" in
  --audio) AUDIO="$2"; shift 2;;
  --workdir) WORKDIR="$2"; shift 2;;
  --envs) ENVS="$2"; shift 2;;
  --no-sfx) SFX=0; shift;;
  --fusion) FUSION="$2"; shift 2;;
  --gpus) GPUS="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 1;;
esac; done
[[ -n "$AUDIO" ]] || { echo "need --audio"; exit 1; }

cd "$HERE"
export PYTHONPATH="$HERE/workers:${PYTHONPATH:-}"
PY_BASE="$ENVS/venv/bin/python"

# GPUs to use (default: all visible). Single-GPU stages are DATA-PARALLEL sharded across them
# (disjoint clips, identical per-clip output → ≈N× throughput on the heavy fusion/SFX/ASR stages).
GPUS="${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; GPUS="${GPUS:-1}"
IFS=',' read -ra GPU_ARR <<< "$([[ "$GPUS" =~ , ]] && echo "$GPUS" || seq -s, 0 $((GPUS-1)))"
NGPU="${#GPU_ARR[@]}"
echo "Using $NGPU GPU(s): ${GPU_ARR[*]}"

run () { echo; echo ">>> $1"; shift; "$@"; }      # one stage, as-is (e.g. VibeVoice uses all GPUs)
runp () {                                          # one single-GPU stage, sharded across GPUs
  echo; echo ">>> $1 (sharded ×$NGPU)"; shift; local py="$1" wk="$2"
  if [[ "$NGPU" -le 1 ]]; then "$py" "$wk" "$WORKDIR"; return; fi
  local pids=() i
  for i in "${!GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" UAAP_SHARD="$i/$NGPU" "$py" "$wk" "$WORKDIR" & pids+=($!)
  done
  wait "${pids[@]}"
}

# Stage 0: decode to canonical wav + build index.json
run "stage 0: prepare audio" "$PY_BASE" prepare_audio.py --audio "$AUDIO" --workdir "$WORKDIR"

# Stage 1: word ASR + diarization (DEFAULT = Nemotron 3.5 words + VibeVoice/Sortformer diarization).
#   VibeVoice provides the diarization / timing authority; Nemotron 3.5 + Sortformer provide the words.
#   The legacy triple-ASR ensemble (stage1b_parakeet.py + stage1c_qwen3.py) is still available — the
#   stage-4 worker auto-detects which ASR JSONs are present. See docs/default_pipeline.md.
# VibeVoice-ASR (~23 GB) shards its OWN model across all GPUs, so run it whole (not clip-sharded).
run "stage 1a: VibeVoice-ASR (diarization/timing)" "$ENVS/venv_vv/bin/python" workers/stage1a_vibevoice.py "$WORKDIR"
runp "stage 1: Nemotron 3.5 + Sortformer (words)"  "$ENVS/venv_nemo/bin/python" workers/stage1_nemotron_sortformer.py

# Stage 1d/1e (default gemma fusion only): pyannote diarization+overlap, then DiCoW overlap-aware ASR
if [[ "$FUSION" == "gemma" ]]; then
  runp "stage 1d: pyannote diarization + overlap" "$ENVS/venv_pyannote/bin/python" workers/stage_pyannote_diar.py
  runp "stage 1e: DiCoW overlap-aware ASR"        "$ENVS/venv_dicow/bin/python"    workers/stage_dicow.py
fi

# Stage 2: Whisper experts
runp "stage 2: Whisper experts" "$PY_BASE" workers/stage2_whisper_experts.py

# Stage 2b: Speaker embeddings (for dramabox fusion, or standalone speaker verification)
if [[ "$FUSION" == "dramabox" ]]; then
  runp "stage 2b: Speaker embeddings" "$PY_BASE" workers/stage2b_speaker_embeddings.py
fi

# Stage 3: SFX LoRA (optional) + vocal-burst candidate pre-pass
if [[ "$SFX" == "1" ]]; then
  runp "stage 3: SFX LoRA" "$PY_BASE" workers/stage3_sfx_lora.py
fi
runp "stage 3b: vocal-burst candidates" "$PY_BASE" workers/stage3b_vocalburst.py

# Final fusion: Gemma-12B text-only (DEFAULT), DramaBox prompts, or legacy MOSS-Audio annotator
if [[ "$FUSION" == "gemma" ]]; then
  runp "stage 5: Gemma-12B text-only fusion (DEFAULT)" "$ENVS/venv_gemma/bin/python" workers/stage5_gemma_fusion.py
elif [[ "$FUSION" == "dramabox" ]]; then
  runp "stage 5: DramaBox prompt generation (Gemma 4 E4B-it)" "$PY_BASE" workers/stage5_dramabox_fusion.py
else
  runp "stage 4: MOSS-Audio annotator (legacy)" "$PY_BASE" workers/stage4_moss_annotator.py
fi

# Report
run "build HTML report" "$PY_BASE" build_report.py --workdir "$WORKDIR" --out "$WORKDIR/report.html"
echo; echo "All done. Predictions are next to each audio file (<name>_pred.json) and in $WORKDIR."
