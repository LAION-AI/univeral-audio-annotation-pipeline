"""VibeVoice-ASR component.

End-to-end ASR with built-in speaker diarization using microsoft/VibeVoice-ASR.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import torch


class VibeVoiceASR:
    """VibeVoice-ASR: end-to-end ASR with built-in diarization.

    Model: microsoft/VibeVoice-ASR
    Features: Automatic speech recognition + speaker diarization in a single pass.
    GPU VRAM: ~16 GB
    """

    MODEL_ID = "microsoft/VibeVoice-ASR"

    def __init__(self, device: str = "cuda:0"):
        """Load the VibeVoice-ASR model.

        Args:
            device: CUDA device string (e.g. 'cuda:0').
        """
        from vibevoice import VibeVoiceASRModel

        print(f"Loading VibeVoice-ASR on {device}...")
        self.model = VibeVoiceASRModel.from_pretrained(self.MODEL_ID)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        print("VibeVoice-ASR loaded.")

    def run(self, audio_path: str) -> List[Dict]:
        """Transcribe audio and return speaker-attributed utterances.

        Args:
            audio_path: Path to WAV audio file.

        Returns:
            List of utterance dicts with keys:
                - start_time (float)
                - end_time (float)
                - speaker_id (int)
                - content (str)
        """
        t0 = time.time()
        results = self.model.transcribe(audio_path)

        utterances = []
        for seg in results:
            utterances.append({
                "start_time": round(seg.get("start", seg.get("start_time", 0.0)), 3),
                "end_time": round(seg.get("end", seg.get("end_time", 0.0)), 3),
                "speaker_id": seg.get("speaker", seg.get("speaker_id", 0)),
                "content": seg.get("text", seg.get("content", "")),
            })

        elapsed = time.time() - t0
        print(f"  VibeVoice-ASR: {len(utterances)} utterances ({elapsed:.1f}s)")
        return utterances

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        torch.cuda.empty_cache()
