#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Create the isolated virtual-envs the default pipeline needs.
#
# The three ASR packages pin MUTUALLY INCOMPATIBLE transformers versions, so each
# ASR stage runs in its own venv; the base venv is shared by the Whisper experts,
# the SFX LoRA and the MOSS annotator.
#
#   venv        base       transformers 4.57.x  -> Whisper experts, SFX, MOSS
#   venv_vv     VibeVoice  transformers 4.51.3  -> VibeVoice-ASR (from GitHub source)
#   venv_qwen   Qwen3-ASR  transformers 4.57.6  -> Qwen3-ASR + ForcedAligner
#   venv_nemo   NeMo       (its own pins)       -> Parakeet TDT v3 + Sortformer
#
# Each venv must provide a CUDA-enabled torch>=2.1. The simplest portable option is
# to install torch into every venv; if you already have a known-good torch build,
# point $BASE_TORCH_SITE at its site-packages and it will be reused via a .pth file.
#
# Usage:  bash setup_environments.sh /path/to/envs
# ---------------------------------------------------------------------------
set -euo pipefail
ENV_DIR="${1:-$(pwd)/envs}"
PYTHON="${PYTHON:-python3}"
TORCH_SPEC="${TORCH_SPEC:-torch torchaudio}"        # set e.g. to a +cuXXX pin if needed
BASE_TORCH_SITE="${BASE_TORCH_SITE:-}"              # optional: reuse an existing torch

mkdir -p "$ENV_DIR"

make_venv () {            # $1 = name
  local v="$ENV_DIR/$1"
  "$PYTHON" -m venv "$v"
  "$v/bin/pip" install -q -U pip
  if [[ -n "$BASE_TORCH_SITE" ]]; then
    local sp; sp="$("$v/bin/python" -c 'import site;print(site.getsitepackages()[0])')"
    echo "$BASE_TORCH_SITE" > "$sp/_reuse_torch.pth"
  else
    "$v/bin/pip" install -q $TORCH_SPEC
  fi
  echo "$v"
}

echo "== base venv (Whisper experts + SFX LoRA + MOSS annotator) =="
VBASE="$(make_venv venv)"
"$VBASE/bin/pip" install -q "transformers>=4.57,<4.58" peft accelerate \
    librosa soundfile numpy huggingface-hub

echo "== venv_vv (VibeVoice-ASR) =="
VVV="$(make_venv venv_vv)"
"$VVV/bin/pip" install -q vibevoice                  # pulls transformers==4.51.3 + accelerate
# Replace the TTS-only wheel with the ASR source from GitHub:
if [[ ! -d "$ENV_DIR/VibeVoice-src" ]]; then
  git clone --depth 1 https://github.com/microsoft/VibeVoice.git "$ENV_DIR/VibeVoice-src"
fi
"$VVV/bin/pip" install -q --no-deps -e "$ENV_DIR/VibeVoice-src"

echo "== venv_qwen (Qwen3-ASR) =="
VQ="$(make_venv venv_qwen)"
"$VQ/bin/pip" install -q qwen-asr                    # pulls transformers==4.57.6

echo "== venv_nemo (Parakeet + Sortformer) =="
VN="$(make_venv venv_nemo)"
"$VN/bin/pip" install -q "nemo-toolkit[asr]"

echo "== MOSS-Audio source (provides src.* for the MOSS models) =="
if [[ ! -d "$ENV_DIR/MOSS-Audio" ]]; then
  git clone --depth 1 https://github.com/OpenMOSS/MOSS-Audio.git "$ENV_DIR/MOSS-Audio"
fi

cat <<EOF

Done. Environments created under: $ENV_DIR
Export this before running the pipeline:

    export UAAP_MOSS_SRC="$ENV_DIR/MOSS-Audio"

A gated model is required for the SFX stage — request access and log in:
    huggingface-cli login        # token with access to laion/moss-audio-sfx-lora-v4
EOF
