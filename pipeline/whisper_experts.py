"""Whisper expert models for per-utterance voice analysis.

Uses three fine-tuned Whisper models to extract emotion, timbre,
and speaking style from each speech segment.
"""

import time
from typing import List, Dict

import torch
import torchaudio
import numpy as np


class WhisperExperts:
    """Three Whisper-small expert models for voice attribute analysis.

    Models:
        - laion/BUD-E-Whisper (emotion classification)
        - laion/timbre-whisper (voice timbre description)
        - laion/voice-tagging-whisper (speaking style tags)

    GPU VRAM: ~2 GB total (3 x ~600 MB)
    """

    MODEL_IDS = {
        "emotion": "laion/BUD-E-Whisper",
        "timbre": "laion/timbre-whisper",
        "style": "laion/voice-tagging-whisper",
    }
    BASE_PROCESSOR = "openai/whisper-small"

    def __init__(self, device: str = "cuda:0"):
        """Load all three Whisper expert models.

        Args:
            device: CUDA device string.
        """
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        self.device = device
        print(f"Loading Whisper experts on {device}...")
        self.processor = WhisperProcessor.from_pretrained(self.BASE_PROCESSOR)
        self.models = {}
        for name, model_id in self.MODEL_IDS.items():
            print(f"  Loading {name} model...")
            model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
            model.eval()
            self.models[name] = model
        print("Whisper experts loaded.")

    def analyze(
        self, audio_path: str, utterances: List[Dict], min_duration: float = 0.3
    ) -> List[Dict]:
        """Run voice analysis on each utterance segment.

        Args:
            audio_path: Path to the full audio file.
            utterances: List of utterance dicts with start_time, end_time, speaker_id.
            min_duration: Minimum segment duration in seconds to analyze.

        Returns:
            List of dicts with keys:
                - start_time (float)
                - end_time (float)
                - speaker_id (int/str)
                - emotion (str)
                - timbre (str)
                - style (str)
        """
        t0 = time.time()

        # Load full audio
        wav, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            sr = 16000
        wav = wav[0].numpy()  # mono

        results = []
        for utt in utterances:
            st = utt["start_time"]
            et = utt["end_time"]

            if et - st < min_duration:
                continue

            start_sample = int(st * sr)
            end_sample = int(et * sr)
            segment = wav[start_sample:end_sample]

            if len(segment) < int(min_duration * sr):
                continue

            result = {
                "start_time": st,
                "end_time": et,
                "speaker_id": utt.get("speaker_id", 0),
            }

            for name, model in self.models.items():
                inputs = self.processor(
                    segment, sampling_rate=16000, return_tensors="pt"
                )
                input_features = inputs.input_features.to(self.device)
                with torch.no_grad():
                    gen_ids = model.generate(input_features, max_new_tokens=128)
                text = self.processor.batch_decode(
                    gen_ids, skip_special_tokens=True
                )[0].strip()
                result[name] = text

            results.append(result)

        elapsed = time.time() - t0
        print(f"  Whisper experts: {len(results)} segments analyzed ({elapsed:.1f}s)")
        return results

    def cleanup(self):
        """Free GPU memory."""
        del self.processor
        for model in self.models.values():
            del model
        self.models.clear()
        torch.cuda.empty_cache()
