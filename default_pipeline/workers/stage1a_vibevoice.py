#!/usr/bin/env python3
"""Stage 1a — VibeVoice-ASR (run in the ``venv_vv`` environment).

Triple-ASR source #1: end-to-end ASR with built-in speaker diarization. The model
(~23 GB) is sharded across the available GPUs. Greedy decoding.
Writes ``<workdir>/<stem>/vibevoice.json``.
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from _common import (block_flash_attn, patch_config_json_dtype, add_repo_to_path,
                     get_workdir, load_index, save_json)

block_flash_attn()
patch_config_json_dtype()
add_repo_to_path()


def main():
    from pipeline.asr_vibevoice import VibeVoiceASR
    workdir = get_workdir()
    index = load_index(workdir)
    asr = VibeVoiceASR()
    for it in index:
        t0 = time.time()
        utts = asr.run(it["wav"])
        save_json(os.path.join(it["workdir"], "vibevoice.json"), utts)
        print(f"  [{it['stem']}] {len(utts)} utts ({time.time()-t0:.1f}s)", flush=True)
    asr.cleanup()


if __name__ == "__main__":
    main()
