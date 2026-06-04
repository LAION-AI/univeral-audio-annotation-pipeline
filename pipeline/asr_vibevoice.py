"""VibeVoice-ASR component.

End-to-end ASR with built-in speaker diarization using microsoft/VibeVoice-ASR.

The public PyPI ``vibevoice`` 0.0.1 wheel ships only the *TTS* stack; the ASR
modeling code (``VibeVoiceASRForConditionalGeneration`` /
``VibeVoiceASRProcessor``) lives in the GitHub repo https://github.com/microsoft/VibeVoice
and must be installed from source (``pip install -e .`` against that checkout).
See ``default_pipeline/setup_environments.sh``.
"""

import time
from typing import List, Dict

import torch


class VibeVoiceASR:
    """VibeVoice-ASR: end-to-end ASR with built-in diarization.

    Model: microsoft/VibeVoice-ASR (~23 GB in bf16)
    Features: speech recognition + speaker diarization + word timing in one pass.

    The model is built on Qwen2.5-7B plus acoustic/semantic speech encoders and is
    too large for a single 24 GB GPU, so it is loaded with ``device_map="auto"`` and
    sharded across the available GPUs.
    """

    MODEL_ID = "microsoft/VibeVoice-ASR"
    # The processor borrows the Qwen2.5 tokenizer for the text side.
    LM_TOKENIZER = "Qwen/Qwen2.5-7B"

    def __init__(self, device: str = "cuda:0", max_memory: dict | None = None):
        """Load the VibeVoice-ASR model and processor.

        Args:
            device: Unused when sharding (kept for API parity); inputs are placed on
                the model's embedding device automatically.
            max_memory: Optional per-GPU memory cap for ``device_map="auto"``
                (e.g. ``{0: "22GiB", 1: "22GiB"}``).
        """
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        print(f"Loading VibeVoice-ASR (device_map=auto)...")
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            self.MODEL_ID, language_model_pretrained_name=self.LM_TOKENIZER
        )
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            max_memory=max_memory or {0: "22GiB", 1: "22GiB"},
            trust_remote_code=True,
        ).eval()
        self.in_device = self.model.get_input_embeddings().weight.device
        print(f"VibeVoice-ASR loaded. input device={self.in_device}")

    def run(self, audio_path: str, max_new_tokens: int = 4096) -> List[Dict]:
        """Transcribe audio and return speaker-attributed utterances (greedy decode).

        Args:
            audio_path: Path to an audio file (wav/mp3/flac...).
            max_new_tokens: Generation cap.

        Returns:
            List of utterance dicts: start_time, end_time, speaker_id, content.
        """
        t0 = time.time()
        inputs = self.processor(
            audio=[str(audio_path)], sampling_rate=None, return_tensors="pt",
            padding=True, add_generation_prompt=True,
        )
        inputs = {k: (v.to(self.in_device) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}
        ilen = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        gen = out[0, ilen:]
        eos = (gen == self.processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos):
            gen = gen[:eos[0] + 1]
        text = self.processor.decode(gen, skip_special_tokens=True)

        try:
            segs = self.processor.post_process_transcription(text)
        except Exception as e:  # pragma: no cover - defensive
            print(f"  VibeVoice: failed to parse structured output: {e}")
            segs = []

        utterances = []
        for s in segs:
            utterances.append({
                "start_time": round(float(s.get("start_time", 0.0) or 0.0), 3),
                "end_time": round(float(s.get("end_time", 0.0) or 0.0), 3),
                "speaker_id": s.get("speaker_id", 0),
                "content": (s.get("text", "") or "").strip(),
            })

        print(f"  VibeVoice-ASR: {len(utterances)} utterances ({time.time()-t0:.1f}s)")
        return utterances

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        torch.cuda.empty_cache()
