#!/usr/bin/env python3
"""Stage 5 (DramaBox alternative) — DramaBox prompt generation via Gemma 4 E4B-it.

Alternative to stage5_gemma_fusion.py. Instead of producing structured JSON
annotations, this stage generates DramaBox TTS prompts from the upstream
expert outputs (ASR, Whisper voice analysis, speaker embeddings).

Activated via ``--fusion dramabox`` in run_all.sh.

Writes ``<workdir>/<stem>/dramabox_prompts.json`` — a list of DramaBox prompts,
one per speaker-consistent group of utterances.
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from _common import (block_flash_attn, add_repo_to_path, get_workdir, load_index,
                     load_json, save_json)

block_flash_attn()
add_repo_to_path()


def main():
    from pipeline.dramabox_fusion import DramaBoxFuser

    workdir = get_workdir()
    index = load_index(workdir)

    model_id = os.environ.get("DRAMABOX_GEMMA_MODEL", "google/gemma-4-E4B-it")
    fuser = DramaBoxFuser(model_id=model_id, device="cuda:0")

    for it in index:
        wd = it["workdir"]
        out_path = f"{wd}/dramabox_prompts.json"
        if os.path.exists(out_path):
            continue  # resume

        t0 = time.time()

        # Load upstream data
        primary = (load_json(f"{wd}/nemotron.json")
                   or load_json(f"{wd}/vibevoice.json")
                   or load_json(f"{wd}/parakeet.json")
                   or load_json(f"{wd}/qwen3.json"))
        whisper = load_json(f"{wd}/whisper.json")

        # Speaker embeddings (optional, from stage 2b)
        spk_emb_path = f"{wd}/speaker_embeddings.json"
        spk_emb = None
        if os.path.exists(spk_emb_path):
            import json
            with open(spk_emb_path) as f:
                spk_emb = json.load(f)

        if not primary:
            save_json(out_path, [])
            continue

        prompts = fuser.fuse_clip(
            asr_utterances=primary,
            whisper_analyses=whisper,
            speaker_embeddings=spk_emb,
        )

        save_json(out_path, prompts)
        # Also write alongside the original audio
        save_json(os.path.splitext(it["audio"])[0] + "_dramabox.json", prompts)

        print(f"  [{it['stem']}] {len(prompts)} DramaBox prompt(s) ({time.time()-t0:.1f}s)",
              flush=True)

    fuser.cleanup()


if __name__ == "__main__":
    main()
