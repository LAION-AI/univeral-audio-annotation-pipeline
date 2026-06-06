#!/usr/bin/env python3
"""Stage 2 — Whisper expert models (run in the base ``venv`` environment).

Per-utterance raw voice attributes: emotion (laion/BUD-E-Whisper),
timbre (laion/timbre-whisper) and speaking style (laion/voice-tagging-whisper).
Segmentation comes from the best available ASR utterances (Parakeet, else VibeVoice,
else Qwen3). Writes ``<workdir>/<stem>/whisper.json``.
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from _common import (block_flash_attn, add_repo_to_path, get_workdir, load_index,
                     load_json, save_json)

block_flash_attn()
add_repo_to_path()


def main():
    from pipeline.whisper_experts import WhisperExperts
    workdir = get_workdir()
    index = load_index(workdir)
    w = WhisperExperts(device="cuda:0")
    for it in index:
        t0 = time.time()
        wd = it["workdir"]
        if os.path.exists(f"{wd}/whisper.json"): continue   # resume: skip done clips
        primary = (load_json(f"{wd}/parakeet.json") or load_json(f"{wd}/vibevoice.json")
                   or load_json(f"{wd}/qwen3.json"))
        analysis = w.analyze(it["wav"], primary) if primary else []
        save_json(f"{wd}/whisper.json", analysis)
        print(f"  [{it['stem']}] {len(analysis)} segs ({time.time()-t0:.1f}s)", flush=True)
    w.cleanup()


if __name__ == "__main__":
    main()
