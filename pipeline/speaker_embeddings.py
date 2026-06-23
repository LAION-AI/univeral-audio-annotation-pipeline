"""Speaker embedding extraction and verification using Orange/Speaker-wavLM-tbr.

Extracts 128-dimensional L2-normalized speaker embeddings per utterance.
Cosine similarity between any two embeddings is simply dot(e1, e2)
since both are already normalized.

Speaker verification:
  - cosine_sim >= 0.42  ->  same speaker
  - cosine_sim <  0.42  ->  different speakers

GPU VRAM: ~300 MB
"""
import time
from typing import List, Dict, Tuple

import numpy as np
import torch
import torchaudio


SIMILARITY_THRESHOLD = 0.42


class SpeakerEmbeddingExtractor:
    """Extract per-utterance speaker embeddings using Speaker-wavLM-tbr.

    Model: Orange/Speaker-wavLM-tbr (128-dim, L2-normalized)
    Input: 16 kHz mono audio segments
    """

    MODEL_ID = "Orange/Speaker-wavLM-tbr"
    TARGET_SR = 16000
    MAX_DURATION_SEC = 30

    def __init__(self, device: str = "cuda:0"):
        from spk_embeddings import EmbeddingsModel

        self.device = device
        print(f"Loading speaker embedding model on {device}...")
        self.model = EmbeddingsModel.from_pretrained(self.MODEL_ID)
        self.model.to(device)
        self.model.eval()
        print("Speaker embedding model loaded.")

    def extract(
        self, audio_path: str, utterances: List[Dict], min_duration: float = 0.3
    ) -> List[Dict]:
        """Extract speaker embeddings for each utterance.

        Args:
            audio_path: Path to the full audio file.
            utterances: List of dicts with start_time, end_time, speaker_id.
            min_duration: Minimum segment duration in seconds.

        Returns:
            List of dicts with keys:
                - start_time (float)
                - end_time (float)
                - speaker_id (int/str)
                - embedding (list of 128 floats, L2-normalized)
        """
        t0 = time.time()

        # Load full audio
        wav, sr = torchaudio.load(str(audio_path))
        if sr != self.TARGET_SR:
            wav = torchaudio.transforms.Resample(sr, self.TARGET_SR)(wav)
            sr = self.TARGET_SR
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav_np = wav[0].numpy()

        max_samples = int(self.MAX_DURATION_SEC * sr)
        results = []

        for utt in utterances:
            st = utt["start_time"]
            et = utt["end_time"]

            if et - st < min_duration:
                continue

            start_sample = int(st * sr)
            end_sample = int(et * sr)
            segment = wav_np[start_sample:end_sample]

            if len(segment) < int(min_duration * sr):
                continue

            # Truncate very long segments
            if len(segment) > max_samples:
                segment = segment[:max_samples]

            # Model expects (batch, samples)
            seg_tensor = torch.from_numpy(segment).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.model(seg_tensor)  # (1, 128)

            emb_np = emb[0].cpu().numpy()
            # L2-normalize
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm

            results.append({
                "start_time": st,
                "end_time": et,
                "speaker_id": utt.get("speaker_id", 0),
                "embedding": emb_np.tolist(),
            })

        elapsed = time.time() - t0
        print(f"  Speaker embeddings: {len(results)} segments ({elapsed:.1f}s)")
        return results

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Speaker verification utilities
# ---------------------------------------------------------------------------

def cosine_similarity(emb1: list | np.ndarray, emb2: list | np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings (= dot product)."""
    a = np.asarray(emb1, dtype=np.float32)
    b = np.asarray(emb2, dtype=np.float32)
    return float(np.dot(a, b))


def same_speaker(emb1, emb2, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """Check if two embeddings belong to the same speaker."""
    return cosine_similarity(emb1, emb2) >= threshold


def cluster_speakers(
    embeddings: List[Dict], threshold: float = SIMILARITY_THRESHOLD
) -> List[List[int]]:
    """Greedy speaker clustering from embedding list.

    Each entry in embeddings must have an "embedding" key.
    Returns list of clusters, where each cluster is a list of indices.
    """
    n = len(embeddings)
    if n == 0:
        return []

    assigned = [-1] * n
    clusters = []

    for i in range(n):
        if assigned[i] >= 0:
            continue

        # Start new cluster
        cluster_id = len(clusters)
        cluster = [i]
        assigned[i] = cluster_id

        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue
            if same_speaker(embeddings[i]["embedding"], embeddings[j]["embedding"], threshold):
                cluster.append(j)
                assigned[j] = cluster_id

        clusters.append(cluster)

    return clusters


def verify_two_part(
    part1_embeddings: List[Dict],
    part2_embeddings: List[Dict],
    threshold: float = SIMILARITY_THRESHOLD,
) -> Tuple[bool, float]:
    """Verify that two audio parts contain the same speaker.

    Compares the mean embedding from each part.
    Returns (is_same_speaker, similarity_score).
    """
    if not part1_embeddings or not part2_embeddings:
        return False, 0.0

    # Average embeddings per part
    mean1 = np.mean([e["embedding"] for e in part1_embeddings], axis=0)
    mean2 = np.mean([e["embedding"] for e in part2_embeddings], axis=0)

    # Re-normalize
    norm1 = np.linalg.norm(mean1)
    norm2 = np.linalg.norm(mean2)
    if norm1 > 0:
        mean1 /= norm1
    if norm2 > 0:
        mean2 /= norm2

    sim = cosine_similarity(mean1, mean2)
    return sim >= threshold, sim
