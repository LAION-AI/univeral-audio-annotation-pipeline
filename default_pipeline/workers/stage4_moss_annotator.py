#!/usr/bin/env python3
"""Stage 4 — MOSS-Audio-8B-Thinking final annotator (run in the base ``venv``).

Listens to the audio together with all upstream context (three ASR transcripts,
Whisper voice analysis, SFX sound events) and emits the final structured JSON:
triple-ASR reconciliation, speaker attribution, and the expressive emotion /
speaking-style captions. GREEDY decoding (do_sample=False).

Writes ``<workdir>/<stem>/predictions.json`` and ``<audio>_pred.json`` next to the
original audio file. If ``<workdir>/<stem>/sfx.json`` is absent the SFX context is
simply empty (MOSS still detects sound events directly from the audio).

Set ``UAAP_MOSS_SRC`` to the MOSS-Audio source checkout (provides ``src.*``).
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from _common import (block_flash_attn, add_repo_to_path, get_workdir, load_index,
                     load_json, save_json)

block_flash_attn()
add_repo_to_path()

MOSS_SRC = os.environ.get("UAAP_MOSS_SRC", "")


def main():
    import soundfile as sf
    from pipeline.moss_annotator import MOSSAnnotator
    from pipeline.utils import fill_timeline_gaps
    workdir = get_workdir()
    index = load_index(workdir)
    moss = MOSSAnnotator(device="cuda:0", moss_audio_path=MOSS_SRC or None)
    for it in index:
        t0 = time.time()
        wd = it["workdir"]
        sfx = load_json(f"{wd}/sfx.json")   # [] if SFX stage was not run
        context = moss.build_triple_context(
            load_json(f"{wd}/vibevoice.json"),
            load_json(f"{wd}/parakeet.json"),
            load_json(f"{wd}/qwen3.json"),
            load_json(f"{wd}/whisper.json"),
            sfx,
            extra_detections=load_json(f"{wd}/vocalburst.json"),  # [] if VB stage not run
        )
        ann = moss.annotate(it["wav"], context, prompt_mode="triple", do_sample=False)
        # Deterministic full-timeline coverage backstop.
        ann = fill_timeline_gaps(ann, sf.info(it["wav"]).duration, sfx_predictions=sfx)
        save_json(f"{wd}/predictions.json", ann)
        # also drop the prediction next to the original audio file
        audio = it["audio"]
        pred_next = os.path.splitext(audio)[0] + "_pred.json"
        save_json(pred_next, ann)
        print(f"  [{it['stem']}] {len(ann)} annotations ({time.time()-t0:.1f}s)", flush=True)
    moss.cleanup()


if __name__ == "__main__":
    main()
