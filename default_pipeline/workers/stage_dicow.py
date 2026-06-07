#!/usr/bin/env python3
"""Stage (overlap ASR) — DiCoW diarization-conditioned Whisper (run in ``venv_dicow``).

Transcribes each pyannote speaker separately (overlap-aware). Reads ``pyannote_diar.json``, writes
``<workdir>/<stem>/dicow.json`` ([{speaker,text,turns}]). Feeds the LLM fuser for overlapping speech.
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from _common import block_flash_attn, add_repo_to_path, get_workdir, load_index, load_json, save_json

block_flash_attn()
add_repo_to_path()


def main():
    from pipeline.asr_dicow import DiCoWASR
    workdir = get_workdir()
    index = load_index(workdir)
    asr = DiCoWASR(device="cuda:0")
    for it in index:
        wd = it["workdir"]
        if os.path.exists(f"{wd}/dicow.json"): continue   # resume
        t0 = time.time()
        out = asr.run(it["wav"], load_json(f"{wd}/pyannote_diar.json"))
        save_json(f"{wd}/dicow.json", out)
        print(f"  [{it['stem']}] {len(out)} speakers ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
