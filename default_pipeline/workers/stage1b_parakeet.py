#!/usr/bin/env python3
"""Stage 1b — Parakeet TDT v3 + Sortformer (run in the ``venv_nemo`` environment).

Triple-ASR source #2. Word-level ASR (GPU0) + speaker diarization (GPU1 if present).
Writes ``<workdir>/<stem>/parakeet.json`` and ``parakeet_diar.json`` (the diarization
segments are reused by the Qwen3 stage for speaker assignment).
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from _common import block_flash_attn, add_repo_to_path, get_workdir, load_index, save_json

block_flash_attn()
add_repo_to_path()


def main():
    import torch
    from pipeline.asr_parakeet import ParakeetSortformerASR
    workdir = get_workdir()
    index = load_index(workdir)
    diar_dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    asr = ParakeetSortformerASR(asr_device="cuda:0", diar_device=diar_dev)
    for it in index:
        t0 = time.time()
        diar = asr._diarize(it["wav"])
        words = asr._transcribe(it["wav"])
        utts = asr._merge(words, diar)
        save_json(os.path.join(it["workdir"], "parakeet.json"), utts)
        save_json(os.path.join(it["workdir"], "parakeet_diar.json"), diar)
        print(f"  [{it['stem']}] {len(utts)} utts, {len(diar)} diar ({time.time()-t0:.1f}s)", flush=True)
    asr.cleanup()


if __name__ == "__main__":
    main()
