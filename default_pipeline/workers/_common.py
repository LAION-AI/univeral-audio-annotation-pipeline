"""Shared helpers for the default-pipeline stage workers.

Each stage runs as its own process (in its own virtual-env, because the ASR
packages pin mutually incompatible ``transformers`` versions) and communicates
with the other stages through small JSON files inside a per-clip work directory:

    <workdir>/index.json                 # list of clips: {stem, audio, wav, workdir}
    <workdir>/<stem>/vibevoice.json      # stage 1a
    <workdir>/<stem>/parakeet.json       # stage 1b  (+ parakeet_diar.json)
    <workdir>/<stem>/qwen3.json          # stage 1c
    <workdir>/<stem>/whisper.json        # stage 2
    <workdir>/<stem>/sfx.json            # stage 3
    <workdir>/<stem>/predictions.json    # stage 4  (final, also written next to the audio)

Run a stage with::

    python workers/<stage>.py /path/to/workdir
"""
import json
import os
import sys
from pathlib import Path

# Repo root (…/repo) so workers can ``import pipeline.*`` regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]


def block_flash_attn():
    """Neutralise a broken/incompatible ``flash_attn`` install.

    Some hosts have a ``flash_attn`` wheel compiled against a newer GLIBC than the
    running system. transformers imports it eagerly when it thinks it is available,
    which hard-crashes the process. Setting the module to ``None`` makes
    ``importlib.util.find_spec`` return ``None`` so transformers reports flash-attn
    as unavailable and cleanly falls back to the SDPA attention kernel.
    """
    sys.modules.setdefault("flash_attn", None)


def patch_config_json_dtype():
    """Make ``PretrainedConfig.to_json_string`` tolerant of torch.dtype values.

    transformers 4.51.x (used by the VibeVoice env) crashes when it merely *logs* a
    config that carries a ``torch.dtype`` object (not JSON-serialisable). We only need
    the repr to succeed, so fall back to an empty object on failure.
    """
    from transformers import PretrainedConfig
    _orig = PretrainedConfig.to_json_string

    def _safe(self, *a, **k):
        try:
            return _orig(self, *a, **k)
        except TypeError:
            return "{}\n"

    PretrainedConfig.to_json_string = _safe


def add_repo_to_path():
    """Allow ``import pipeline.*`` from the repo root."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def get_workdir() -> Path:
    """Resolve the work directory from argv[1] or $UAAP_WORKDIR (default ./uaap_work)."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return Path(sys.argv[1]).resolve()
    return Path(os.environ.get("UAAP_WORKDIR", "uaap_work")).resolve()


def load_index(workdir: Path) -> list:
    return json.loads((workdir / "index.json").read_text())


def load_json(path) -> list:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else []


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
