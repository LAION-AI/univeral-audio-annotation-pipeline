#!/usr/bin/env python3
"""Stage (diarization) — pyannote speaker diarization + overlap detection (run in ``venv_pyannote``).

Writes ``<workdir>/<stem>/pyannote_diar.json`` ([{start,end,speaker}]) and ``overlaps.json``
([{start,end}]). Needs ``HF_TOKEN`` with the gated pyannote terms accepted. Feeds the DiCoW stage.
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from _common import add_repo_to_path, get_workdir, load_index, save_json

add_repo_to_path()


def main():
    from pipeline.diarize_pyannote import PyannoteDiarizer
    import torch
    workdir = get_workdir()
    index = load_index(workdir)
    dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    diar = PyannoteDiarizer(device=dev, token=os.environ.get("HF_TOKEN"))
    for it in index:
        if os.path.exists(os.path.join(it["workdir"], "pyannote_diar.json")): continue   # resume
        t0 = time.time()
        turns, overlaps = diar.run(it["wav"])
        save_json(os.path.join(it["workdir"], "pyannote_diar.json"), turns)
        save_json(os.path.join(it["workdir"], "overlaps.json"), overlaps)
        nspk = len({t["speaker"] for t in turns})
        print(f"  [{it['stem']}] {len(turns)} turns, {nspk} spk, {len(overlaps)} overlaps ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
