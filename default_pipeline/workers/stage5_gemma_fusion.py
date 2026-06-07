#!/usr/bin/env python3
"""Stage 5 (DEFAULT final annotator) — Gemma text-only LLM fusion (run in ``venv_gemma``).

Replaces the MOSS-Audio annotator as the default last step. Fuses ONLY the upstream experts' text
outputs (no audio) into the final JSON. Writes ``<workdir>/<stem>/predictions.json`` and
``<audio>_pred.json``. Model via env: GEMMA_REPO / GEMMA_FILE (default Gemma-4-12B Q8).
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from _common import add_repo_to_path, get_workdir, load_index, load_json, save_json

add_repo_to_path()


def main():
    import soundfile as sf
    from pipeline.gemma_fusion import GemmaFuser, build_fusion_context
    workdir = get_workdir()
    index = load_index(workdir)
    fuser = GemmaFuser(repo_id=os.environ.get("GEMMA_REPO", "unsloth/gemma-4-12b-it-GGUF"),
                       filename=os.environ.get("GEMMA_FILE", "gemma-4-12b-it-Q8_0.gguf"),
                       n_ctx=int(os.environ.get("GEMMA_NCTX", "16384")))
    for it in index:
        wd = it["workdir"]
        if os.path.exists(f"{wd}/predictions.json"): continue   # resume
        t0 = time.time()
        sfx = load_json(f"{wd}/sfx.json")
        ctx = build_fusion_context(
            vibevoice=load_json(f"{wd}/vibevoice.json"),
            nemotron=load_json(f"{wd}/nemotron.json"),
            sortformer_diar=load_json(f"{wd}/sortformer_diar.json"),
            dicow=load_json(f"{wd}/dicow.json"),
            whisper=load_json(f"{wd}/whisper.json"),
            sfx=sfx,
            vocalburst=load_json(f"{wd}/vocalburst.json"),
        )
        dur = sf.info(it["wav"]).duration
        ann = fuser.fuse(dur, ctx, sfx_predictions=sfx)
        save_json(f"{wd}/predictions.json", ann)
        save_json(os.path.splitext(it["audio"])[0] + "_pred.json", ann)
        print(f"  [{it['stem']}] {len(ann)} annotations ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
