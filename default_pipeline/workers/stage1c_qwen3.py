#!/usr/bin/env python3
"""Stage 1c — Qwen3-ASR-1.7B + ForcedAligner (run in the ``venv_qwen`` environment).

Triple-ASR source #3. Uses the Sortformer diarization from stage 1b for speaker
assignment. Writes ``<workdir>/<stem>/qwen3.json``.
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from _common import (block_flash_attn, add_repo_to_path, get_workdir, load_index,
                     load_json, save_json)

block_flash_attn()
add_repo_to_path()


def main():
    from pipeline.asr_qwen3 import Qwen3ASR
    workdir = get_workdir()
    index = load_index(workdir)
    asr = Qwen3ASR(device="cuda:0")
    for it in index:
        t0 = time.time()
        diar = load_json(os.path.join(it["workdir"], "parakeet_diar.json")) or None
        utts = asr.run(it["wav"], language="English", diar_segs=diar)
        save_json(os.path.join(it["workdir"], "qwen3.json"), utts)
        print(f"  [{it['stem']}] {len(utts)} utts ({time.time()-t0:.1f}s)", flush=True)
    asr.cleanup()


if __name__ == "__main__":
    main()
