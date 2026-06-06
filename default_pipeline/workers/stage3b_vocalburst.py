#!/usr/bin/env python3
"""Stage 3b — vocal-burst candidate pre-pass (run in the base ``venv``).

Runs the vocal-burst LOCATOR (laion/vocalburst-locator, confidence threshold 0.7,
>30 s audio split into <=30 s windows with cross-window merging and frame smoothing),
then captions each detected candidate with laion/sound-effect-captioning-whisper
(batched). Writes ``<workdir>/<stem>/vocalburst.json`` — a list of
``{start, end, confidence, caption}`` that stage 4 hands to MOSS as candidate sound
effects to verify (MOSS keeps only the ones it can actually hear).

Runs alongside the SFX LoRA stage, before the MOSS annotator.
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from _common import block_flash_attn, add_repo_to_path, get_workdir, load_index, save_json

block_flash_attn()
add_repo_to_path()

THRESHOLD = float(os.environ.get("UAAP_VB_THRESHOLD", "0.7"))


def main():
    import librosa
    from pipeline.vocalburst_locator import (VocalBurstLocator, SoundEffectCaptioner,
                                             SAMPLE_RATE)
    workdir = get_workdir()
    index = load_index(workdir)
    index = [it for it in index if not os.path.exists(os.path.join(it["workdir"], "vocalburst.json"))]  # resume
    if not index:
        print("all vocalburst.json present; skipping", flush=True); return

    # Stage 1: locate candidates across every clip (model loaded once).
    locator = VocalBurstLocator(device="cuda:0")
    wavs, cands = {}, {}
    for it in index:
        t0 = time.time()
        wavs[it["stem"]] = librosa.load(it["wav"], sr=SAMPLE_RATE, mono=True)[0]
        cands[it["stem"]] = locator.detect(wavs[it["stem"]], threshold=THRESHOLD)
        print(f"  [{it['stem']}] {len(cands[it['stem']])} candidates ({time.time()-t0:.1f}s)", flush=True)
    locator.cleanup()

    # Stage 2: caption each candidate (model loaded once, batched).
    captioner = SoundEffectCaptioner(device="cuda:0")
    for it in index:
        c = cands[it["stem"]]
        out = []
        if c:
            wav = wavs[it["stem"]]
            segs = [wav[int(x["start"]*SAMPLE_RATE):int(x["end"]*SAMPLE_RATE)] for x in c]
            caps = captioner.caption(segs)
            out = [{"start": x["start"], "end": x["end"], "confidence": x["confidence"],
                    "caption": cap} for x, cap in zip(c, caps)]
        save_json(os.path.join(it["workdir"], "vocalburst.json"), out)
        print(f"  [{it['stem']}] captioned {len(out)}", flush=True)
    captioner.cleanup()


if __name__ == "__main__":
    main()
