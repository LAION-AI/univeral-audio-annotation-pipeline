#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the full default pipeline over a file or directory of audio clips.
#
# Stages run sequentially (each model is loaded once, processes all clips, then
# frees the GPU before the next stage). Every stage uses its dedicated venv.
#
# Usage:
#   export UAAP_MOSS_SRC=/path/to/MOSS-Audio       # MOSS source checkout
#   bash run_all.sh --audio /path/to/clips --workdir ./uaap_work --envs ./envs [--no-sfx]
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AUDIO=""; WORKDIR="./uaap_work"; ENVS="./envs"; SFX=1
while [[ $# -gt 0 ]]; do case "$1" in
  --audio) AUDIO="$2"; shift 2;;
  --workdir) WORKDIR="$2"; shift 2;;
  --envs) ENVS="$2"; shift 2;;
  --no-sfx) SFX=0; shift;;
  *) echo "unknown arg $1"; exit 1;;
esac; done
[[ -n "$AUDIO" ]] || { echo "need --audio"; exit 1; }

cd "$HERE"
export PYTHONPATH="$HERE/workers:${PYTHONPATH:-}"
PY_BASE="$ENVS/venv/bin/python"
run () { echo; echo ">>> $1"; shift; "$@"; }

# Stage 0: decode to canonical wav + build index.json
run "stage 0: prepare audio" "$PY_BASE" prepare_audio.py --audio "$AUDIO" --workdir "$WORKDIR"

# Stage 1: three ASR systems (order matters: Parakeet produces diar segs for Qwen3)
run "stage 1a: VibeVoice-ASR" "$ENVS/venv_vv/bin/python"   workers/stage1a_vibevoice.py "$WORKDIR"
run "stage 1b: Parakeet+Sortformer" "$ENVS/venv_nemo/bin/python" workers/stage1b_parakeet.py "$WORKDIR"
run "stage 1c: Qwen3-ASR"     "$ENVS/venv_qwen/bin/python" workers/stage1c_qwen3.py "$WORKDIR"

# Stage 2: Whisper experts
run "stage 2: Whisper experts" "$PY_BASE" workers/stage2_whisper_experts.py "$WORKDIR"

# Stage 3: SFX LoRA (optional) + vocal-burst candidate pre-pass
if [[ "$SFX" == "1" ]]; then
  run "stage 3: SFX LoRA" "$PY_BASE" workers/stage3_sfx_lora.py "$WORKDIR"
fi
run "stage 3b: vocal-burst candidates" "$PY_BASE" workers/stage3b_vocalburst.py "$WORKDIR"

# Stage 4: MOSS final annotation (greedy)
run "stage 4: MOSS annotator" "$PY_BASE" workers/stage4_moss_annotator.py "$WORKDIR"

# Report
run "build HTML report" "$PY_BASE" build_report.py --workdir "$WORKDIR" --out "$WORKDIR/report.html"
echo; echo "All done. Predictions are next to each audio file (<name>_pred.json) and in $WORKDIR."
