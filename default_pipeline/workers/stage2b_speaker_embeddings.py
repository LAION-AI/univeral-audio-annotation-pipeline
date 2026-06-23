#!/usr/bin/env python3
"""Stage 2b — Speaker embedding extraction (run in the base ``venv`` environment).

Extracts per-utterance 128-dim L2-normalized speaker embeddings using
Orange/Speaker-wavLM-tbr. Also performs greedy speaker clustering
(cosine threshold 0.42) and writes results to
``<workdir>/<stem>/speaker_embeddings.json``.

Segmentation comes from the best available ASR utterances (same source as
stage 2 whisper experts).
"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from _common import (block_flash_attn, add_repo_to_path, get_workdir, load_index,
                     load_json, save_json)

block_flash_attn()
add_repo_to_path()


def main():
    from pipeline.speaker_embeddings import (
        SpeakerEmbeddingExtractor, cluster_speakers, verify_two_part,
    )

    workdir = get_workdir()
    index = load_index(workdir)
    extractor = SpeakerEmbeddingExtractor(device="cuda:0")

    for it in index:
        t0 = time.time()
        wd = it["workdir"]
        out_path = f"{wd}/speaker_embeddings.json"
        if os.path.exists(out_path):
            continue  # resume: skip done clips

        # Use same primary ASR source as whisper experts
        primary = (load_json(f"{wd}/nemotron.json")
                   or load_json(f"{wd}/vibevoice.json")
                   or load_json(f"{wd}/parakeet.json")
                   or load_json(f"{wd}/qwen3.json"))

        if not primary:
            save_json(out_path, {"embeddings": [], "clusters": [], "speaker_count": 0})
            continue

        embeddings = extractor.extract(it["wav"], primary)
        clusters = cluster_speakers(embeddings)

        # Assign cluster-based speaker IDs
        cluster_map = {}
        for cluster_id, indices in enumerate(clusters):
            for idx in indices:
                cluster_map[idx] = cluster_id
                embeddings[idx]["cluster_id"] = cluster_id

        result = {
            "embeddings": embeddings,
            "clusters": [[idx for idx in c] for c in clusters],
            "speaker_count": len(clusters),
        }

        save_json(out_path, result)
        print(f"  [{it['stem']}] {len(embeddings)} embs, {len(clusters)} speakers "
              f"({time.time()-t0:.1f}s)", flush=True)

    extractor.cleanup()


if __name__ == "__main__":
    main()
