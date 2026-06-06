#!/usr/bin/env python3
"""Stage 3 — SFX LoRA sound-event detector (run in the base ``venv`` environment).

MOSS-Audio-8B-Instruct + laion/moss-audio-sfx-lora-v4 (rank-128 adapter trained on
~11k LAION soundscapes). Produces overlapping medium-length sound events with
timestamps and captions, which the MOSS annotator (stage 4) then verifies, splits
and refines. Writes ``<workdir>/<stem>/sfx.json``.

Set ``UAAP_MOSS_SRC`` to the MOSS-Audio source checkout (provides ``src.*``).
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from _common import block_flash_attn, add_repo_to_path, get_workdir, load_index, save_json

block_flash_attn()
add_repo_to_path()

MOSS_SRC = os.environ.get("UAAP_MOSS_SRC", "")


def main():
    from pipeline.sfx_lora import SFXDetector
    workdir = get_workdir()
    index = load_index(workdir)
    sfx = SFXDetector(device="cuda:0", moss_audio_path=MOSS_SRC or None)
    for it in index:
        t0 = time.time()
        if os.path.exists(os.path.join(it["workdir"], "sfx.json")): continue   # resume
        preds = sfx.run(it["wav"], segment_duration="medium", overlapping=True)
        save_json(os.path.join(it["workdir"], "sfx.json"), preds)
        print(f"  [{it['stem']}] {len(preds)} events ({time.time()-t0:.1f}s)", flush=True)
    sfx.cleanup()


if __name__ == "__main__":
    main()
