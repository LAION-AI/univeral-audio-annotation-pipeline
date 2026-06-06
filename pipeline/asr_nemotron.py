"""Nemotron 3.5 ASR + Sortformer diarization component.

Uses nvidia/nemotron-3.5-asr-streaming-0.6b for word-level ASR and
nvidia/diar_sortformer_4spk-v1 for speaker diarization, then merges the two into
speaker-attributed utterances. This is the word source for the DEFAULT (recommended)
configuration (Nemotron words + VibeVoice/Sortformer diarization), which scores best on
SoundScape-Bench.
"""

import time
from typing import List, Dict

import torch

from .utils import find_dominant_speaker


class NemotronSortformerASR:
    """Nemotron 3.5 streaming ASR with Sortformer speaker diarization.

    Models:
        - nvidia/nemotron-3.5-asr-streaming-0.6b (ASR, word-level timestamps)
        - nvidia/diar_sortformer_4spk-v1 (diarization, up to 4 speakers)

    GPU VRAM: ~3 GB (Nemotron) + ~2 GB (Sortformer)
    """

    NEMOTRON_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    SORTFORMER_MODEL = "nvidia/diar_sortformer_4spk-v1"
    # Streaming attention context (left, right) chunks — matches the model card recipe.
    ATT_CONTEXT = [56, 13]

    def __init__(self, asr_device: str = "cuda:0", diar_device: str = "cuda:0"):
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.models import SortformerEncLabelModel

        # Nemotron's lhotse prompt-index dataset needs a tiny shim before construction.
        try:
            from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import (
                LhotseSpeechToTextBpeDatasetWithPromptIndex as _DS,
            )
            _DS._get_prompt_index_for_cut = lambda self, cut: self.auto_index
        except Exception:
            pass

        print(f"Loading Sortformer on {diar_device}...")
        self.diar_model = SortformerEncLabelModel.from_pretrained(self.SORTFORMER_MODEL)
        self.diar_model = self.diar_model.to(diar_device).eval()
        self.diar_device = diar_device

        print(f"Loading Nemotron 3.5 ASR on {asr_device}...")
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            self.NEMOTRON_MODEL, map_location=asr_device
        )
        self.asr_model.encoder.set_default_att_context_size(self.ATT_CONTEXT)
        self.asr_device = asr_device
        print("Nemotron 3.5 + Sortformer loaded.")

    def run(self, audio_path: str) -> List[Dict]:
        """Transcribe + diarize. Returns utterances with start_time/end_time/speaker_id/content."""
        diar_segs = self._diarize(audio_path)
        words = self._transcribe(audio_path)
        utterances = self._merge(words, diar_segs)
        print(f"  Nemotron+Sortformer: {len(utterances)} utterances")
        return utterances

    def _diarize(self, audio_path: str) -> List[Dict]:
        t0 = time.time()
        predicted_segments = self.diar_model.diarize(audio=[str(audio_path)], batch_size=1)
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
        """Run Nemotron 3.5 ASR with word-level timestamps."""
        t0 = time.time()
        output = self.asr_model.transcribe(
            [str(audio_path)], batch_size=1, timestamps=True, verbose=False
        )[0]
        words = []
        ts = getattr(output, "timestamp", None) or {}
        for w in ts.get("word", []):
            words.append({
                "word": w.get("word", w.get("char", "")),
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
            })
        print(f"    Nemotron: {len(words)} words ({time.time()-t0:.1f}s)")
        return words

    def _merge(self, words: List[Dict], diar_segs: List[Dict],
               gap_threshold: float = 1.0) -> List[Dict]:
        """Assign each word to a speaker, group consecutive same-speaker words into utterances."""
        if not words:
            return []
        for word in words:
            word["speaker"] = find_dominant_speaker(word["start"], word["end"], diar_segs)
        utterances, current_utt = [], None
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
                    "start_time": word["start"], "end_time": word["end"],
                    "speaker_id": word["speaker"], "content": word["word"],
                }
            else:
                current_utt["end_time"] = word["end"]
                current_utt["content"] += " " + word["word"]
        if current_utt:
            utterances.append(current_utt)
        return utterances

    def cleanup(self):
        del self.asr_model
        del self.diar_model
        torch.cuda.empty_cache()
