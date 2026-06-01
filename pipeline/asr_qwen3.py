"""Qwen3-ASR-1.7B + Forced Aligner component.

Uses Qwen/Qwen3-ASR-1.7B for speech recognition with
Qwen/Qwen3-ForcedAligner-0.6B for word-level timestamps.
Speaker assignment uses Sortformer diarization results.
"""

import json
import time
from typing import List, Dict, Optional

import torch

from .utils import find_dominant_speaker


class Qwen3ASR:
    """Qwen3-ASR with forced alignment and Sortformer speaker assignment.

    Models:
        - Qwen/Qwen3-ASR-1.7B (ASR)
        - Qwen/Qwen3-ForcedAligner-0.6B (word-level timestamps)

    GPU VRAM: ~8 GB (ASR) + ~3 GB (aligner)
    """

    ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
    ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

    def __init__(self, device: str = "cuda:0"):
        """Load Qwen3-ASR and ForcedAligner.

        Args:
            device: CUDA device string.
        """
        from qwen_asr import Qwen3ASRModel

        print(f"Loading Qwen3-ASR on {device}...")
        self.model = Qwen3ASRModel.from_pretrained(
            self.ASR_MODEL,
            dtype=torch.bfloat16,
            device_map=device,
            max_inference_batch_size=32,
            max_new_tokens=512,
            forced_aligner=self.ALIGNER_MODEL,
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map=device,
            ),
        )
        self.device = device
        print("Qwen3-ASR loaded.")

    def run(
        self,
        audio_path: str,
        language: str = "English",
        diar_segs: Optional[List[Dict]] = None,
        gap_threshold: float = 1.0,
    ) -> List[Dict]:
        """Transcribe audio with word-level timestamps and optional speaker assignment.

        Args:
            audio_path: Path to WAV audio file.
            language: Language for transcription (default 'English').
            diar_segs: Optional Sortformer diarization segments for speaker assignment.
            gap_threshold: Max gap in seconds before splitting utterances.

        Returns:
            List of utterance dicts with keys:
                - start_time (float)
                - end_time (float)
                - speaker_id (int)
                - content (str)
        """
        t0 = time.time()
        results = self.model.transcribe(
            audio=str(audio_path),
            language=language,
            return_time_stamps=True,
        )

        words = []
        if results[0].time_stamps is not None:
            for item in results[0].time_stamps:
                words.append({
                    "word": item.text,
                    "start": round(item.start_time, 3),
                    "end": round(item.end_time, 3),
                })

        elapsed = time.time() - t0
        print(f"  Qwen3-ASR: {len(words)} words ({elapsed:.1f}s)")

        if not words:
            return [{
                "start_time": 0.0,
                "end_time": 30.0,
                "speaker_id": 0,
                "content": results[0].text,
            }]

        # Assign speakers if diarization segments provided
        if diar_segs:
            for word in words:
                word["speaker"] = find_dominant_speaker(
                    word["start"], word["end"], diar_segs
                )
        else:
            for word in words:
                word["speaker"] = 0

        # Group into utterances
        utterances = []
        current_utt = None
        for word in words:
            should_split = (
                current_utt is None
                or word["speaker"] != current_utt["speaker_id"]
                or (word["start"] - current_utt["end_time"]) > gap_threshold
            )
            if should_split:
                if current_utt:
                    utterances.append(current_utt)
                current_utt = {
                    "start_time": word["start"],
                    "end_time": word["end"],
                    "speaker_id": word["speaker"],
                    "content": word["word"],
                }
            else:
                current_utt["end_time"] = word["end"]
                current_utt["content"] += " " + word["word"]

        if current_utt:
            utterances.append(current_utt)

        return utterances

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        torch.cuda.empty_cache()
