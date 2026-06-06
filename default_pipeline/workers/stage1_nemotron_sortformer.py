#!/usr/bin/env python3
"""Stage 1 (word ASR) — Nemotron 3.5 + Sortformer (run in the ``venv_nemo`` environment).

The DEFAULT word source: nvidia/nemotron-3.5-asr-streaming-0.6b for word-level ASR (GPU0)
plus nvidia/diar_sortformer_4spk-v1 diarization (GPU1 if present). Writes
``<workdir>/<stem>/nemotron.json`` and ``sortformer_diar.json``. VibeVoice (stage 1a) still
runs as the diarization / timing authority; this stage supplies the words.
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from _common import block_flash_attn, add_repo_to_path, get_workdir, load_index, save_json

block_flash_attn()
add_repo_to_path()


def main():
    import torch
    from pipeline.asr_nemotron import NemotronSortformerASR
    workdir = get_workdir()
    index = load_index(workdir)
    diar_dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    asr = NemotronSortformerASR(asr_device="cuda:0", diar_device=diar_dev)
    for it in index:
        if os.path.exists(os.path.join(it["workdir"], "nemotron.json")): continue   # resume
        t0 = time.time()
        diar = asr._diarize(it["wav"])
        words = asr._transcribe(it["wav"])
        utts = asr._merge(words, diar)
        save_json(os.path.join(it["workdir"], "nemotron.json"), utts)
        save_json(os.path.join(it["workdir"], "sortformer_diar.json"), diar)
        print(f"  [{it['stem']}] {len(utts)} utts, {len(diar)} diar ({time.time()-t0:.1f}s)", flush=True)
    asr.cleanup()


if __name__ == "__main__":
    main()
