"""LoRA-based sound effect detection component.

Uses MOSS-Audio-8B-Instruct with a LoRA adapter fine-tuned on
10,998 soundscapes for sound event detection with timestamps.
"""

import json
import re
import time
from typing import List, Dict, Optional

import torch


class SFXDetector:
    """Sound effect detector using MOSS-Audio-8B-Instruct + LoRA adapter.

    Base model: OpenMOSS-Team/MOSS-Audio-8B-Instruct
    LoRA adapter: LAION-AI/moss-audio-sfx-lora-v4
        - Rank 128, alpha 256
        - Trained on 10,998 LAION soundscapes (Gemini 2.5 Pro annotations)
        - Checkpoint: step 2750

    GPU VRAM: ~18 GB
    """

    BASE_MODEL = "OpenMOSS-Team/MOSS-Audio-8B-Instruct"
    LORA_REPO = "LAION-AI/moss-audio-sfx-lora-v4"

    PROMPT_TEMPLATE = (
        "Please describe all audio events in this audio together with "
        "start time, end time, and caption for {segment_duration} segments "
        "that are {overlapping_str}."
    )

    def __init__(
        self,
        device: str = "cuda:0",
        lora_path: Optional[str] = None,
        moss_audio_path: Optional[str] = None,
    ):
        """Load MOSS-Audio-8B-Instruct with LoRA adapter.

        Args:
            device: CUDA device string.
            lora_path: Local path to LoRA adapter directory. If None, downloads
                from HuggingFace (LAION-AI/moss-audio-sfx-lora-v4).
            moss_audio_path: Path to MOSS-Audio source code. If None, uses
                the installed package or trust_remote_code.
        """
        import sys
        if moss_audio_path:
            sys.path.insert(0, moss_audio_path)

        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor
        from peft import PeftModel

        print(f"Loading MOSS-Audio-8B-Instruct on {device}...")
        self.processor = MossAudioProcessor.from_pretrained(
            self.BASE_MODEL, trust_remote_code=True
        )
        model = MossAudioModel.from_pretrained(
            self.BASE_MODEL,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        )

        # Load and merge LoRA adapter
        adapter_path = lora_path or self._download_lora()
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        model.eval()

        self.model = model
        self.device = device
        self.mel_sr = self.processor.config.mel_sr
        self.audio_token_id = self.processor.audio_token_id
        print("SFX detector loaded.")

    def _download_lora(self) -> str:
        """Download LoRA adapter from HuggingFace."""
        from huggingface_hub import snapshot_download

        path = snapshot_download(self.LORA_REPO)
        return path

    def run(
        self,
        audio_path: str,
        segment_duration: str = "medium",
        overlapping: bool = True,
    ) -> List[Dict]:
        """Detect sound events in audio.

        Args:
            audio_path: Path to WAV audio file.
            segment_duration: 'short' or 'medium' segment length.
            overlapping: Whether to allow overlapping predictions.

        Returns:
            List of sound event dicts with keys:
                - start_time (float)
                - end_time (float)
                - caption (str)
        """
        from src.audio_io import load_audio

        t0 = time.time()

        overlapping_str = "overlapping" if overlapping else "not overlapping"
        prompt = self.PROMPT_TEMPLATE.format(
            segment_duration=segment_duration,
            overlapping_str=overlapping_str,
        )

        raw_audio = load_audio(str(audio_path), sample_rate=self.mel_sr)

        inputs = self.processor(
            text=prompt, audios=[raw_audio], return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(self.model.dtype)
        inputs["audio_input_mask"] = inputs["input_ids"] == self.audio_token_id

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=1.0,
                use_cache=True,
            )

        input_len = inputs["input_ids"].shape[1]
        response_text = self.processor.decode(
            generated_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        # Parse JSON output
        parsed = self._parse_events(response_text)

        elapsed = time.time() - t0
        print(f"  SFX LoRA: {len(parsed)} events ({elapsed:.1f}s)")
        return parsed

    def _parse_events(self, text: str) -> List[Dict]:
        """Parse sound events from model output."""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.processor
        torch.cuda.empty_cache()
