"""Parakeet TDT v3 + Sortformer diarization component.

Uses nvidia/parakeet-tdt-0.6b-v3 for word-level ASR and
nvidia/diar_sortformer_4spk-v1 for speaker diarization,
then merges the two for speaker-attributed utterances.
"""

import json
import time
from typing import List, Dict, Tuple

import torch

from .utils import find_dominant_speaker


class ParakeetSortformerASR:
    """Parakeet TDT v3 ASR with Sortformer speaker diarization.

    Models:
        - nvidia/parakeet-tdt-0.6b-v3 (ASR, word-level timestamps)
        - nvidia/diar_sortformer_4spk-v1 (diarization, up to 4 speakers)

    GPU VRAM: ~4 GB (Parakeet) + ~2 GB (Sortformer)
    """

    PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
    SORTFORMER_MODEL = "nvidia/diar_sortformer_4spk-v1"

    def __init__(self, asr_device: str = "cuda:0", diar_device: str = "cuda:0"):
        """Load Parakeet and Sortformer models.

        Args:
            asr_device: CUDA device for Parakeet ASR.
            diar_device: CUDA device for Sortformer diarization.
        """
        import nemo.collections.asr as nemo_asr

        print(f"Loading Sortformer on {diar_device}...")
        self.diar_model = nemo_asr.models.SortformerEncLabelModel.from_pretrained(
            self.SORTFORMER_MODEL
        )
        self.diar_model = self.diar_model.to(diar_device)
        self.diar_model.eval()
        self.diar_device = diar_device

        print(f"Loading Parakeet TDT v3 on {asr_device}...")
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(self.PARAKEET_MODEL)
        self.asr_model = self.asr_model.to(asr_device)
        self.asr_device = asr_device

        print("Parakeet + Sortformer loaded.")

    def run(self, audio_path: str) -> List[Dict]:
        """Transcribe and diarize audio.

        Args:
            audio_path: Path to WAV audio file.

        Returns:
            List of utterance dicts with keys:
                - start_time (float)
                - end_time (float)
                - speaker_id (int)
                - content (str)
        """
        diar_segs = self._diarize(audio_path)
        words = self._transcribe(audio_path)
        utterances = self._merge(words, diar_segs)

        print(f"  Parakeet+Sortformer: {len(utterances)} utterances")
        return utterances

    def _diarize(self, audio_path: str) -> List[Dict]:
        """Run Sortformer diarization."""
        t0 = time.time()
        predicted_segments = self.diar_model.diarize(
            audio=[str(audio_path)], batch_size=1
        )

        diar_results = []
        for seg in predicted_segments[0]:
            if hasattr(seg, "start"):
                diar_results.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "speaker": int(seg.speaker) if hasattr(seg, "speaker") else 0,
                })
            else:
                parts = str(seg).strip().split()
                if len(parts) >= 3:
                    diar_results.append({
                        "start": float(parts[0]),
                        "end": float(parts[1]),
                        "speaker": int(parts[2].replace("speaker_", "")),
                    })

        print(f"    Sortformer: {len(diar_results)} segments ({time.time()-t0:.1f}s)")
        return diar_results

    def _transcribe(self, audio_path: str) -> List[Dict]:
        """Run Parakeet TDT v3 ASR with word-level timestamps."""
        t0 = time.time()
        output = self.asr_model.transcribe([str(audio_path)], timestamps=True)

        words = []
        if hasattr(output[0], "timestamp") and output[0].timestamp:
            if "word" in output[0].timestamp:
                for w in output[0].timestamp["word"]:
                    words.append({
                        "word": w.get("word", w.get("char", "")),
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                    })

        print(f"    Parakeet: {len(words)} words ({time.time()-t0:.1f}s)")
        return words

    def _merge(
        self, words: List[Dict], diar_segs: List[Dict], gap_threshold: float = 1.0
    ) -> List[Dict]:
        """Merge word-level ASR with diarization segments.

        Assigns each word to a speaker based on time overlap,
        then groups consecutive same-speaker words into utterances,
        splitting on gaps > gap_threshold seconds.
        """
        if not words:
            return []

        for word in words:
            word["speaker"] = find_dominant_speaker(
                word["start"], word["end"], diar_segs
            )

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
        del self.asr_model
        del self.diar_model
        torch.cuda.empty_cache()
