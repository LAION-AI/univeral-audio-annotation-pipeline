"""Vocal-burst candidate detector + sound-effect captioner.

Two specialist models, run together as one pre-pass that feeds extra candidate
sound events into the MOSS annotator:

  1. ``laion/vocalburst-locator`` — a Whisper-small encoder + frame-level head that
     emits a vocal-burst probability for every 20 ms frame (50 fps) of a 30 s window.
     Audio longer than 30 s is split into <=30 s windows, run as a batch, and the
     per-window probabilities are concatenated into one continuous timeline so a burst
     that straddles a window boundary stays a single event. The timeline is median-
     smoothed (to erase 1-2 frame outliers), thresholded, and grouped into events with
     small-gap merging and a minimum-duration filter.

  2. ``laion/sound-effect-captioning-whisper`` — a Whisper captioner. Each detected
     candidate is cut out and the batch is captioned in one call.

The output is a list of ``{start, end, confidence, caption}`` candidates. These are
handed to MOSS as *specialist-detected sound effects* (not labelled "vocal bursts");
MOSS decides whether each is genuinely audible and worth keeping.
"""

import time
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


SAMPLE_RATE = 16000
WINDOW_SECONDS = 30.0
FPS = 50                       # 1500 frames / 30 s
LOCATOR_REPO = "laion/vocalburst-locator"
CAPTIONER_REPO = "laion/sound-effect-captioning-whisper"


# ──────────────────────────────────────────────────────────────────────────
# Locator model (architecture mirrors the repo's inference.py)
# ──────────────────────────────────────────────────────────────────────────
class WhisperSegmenter(nn.Module):
    """Whisper-small encoder + per-frame binary segmentation head."""

    def __init__(self, whisper_id: str = "openai/whisper-small"):
        super().__init__()
        from transformers import WhisperModel
        self.whisper = WhisperModel.from_pretrained(whisper_id)
        d_model = self.whisper.config.d_model       # 768
        hidden = max(256, d_model // 2)             # 384
        self.proj = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(0.1))
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3), nn.GELU(), nn.Dropout(0.1))
        self.out = nn.Linear(hidden, 1)

    def forward(self, input_features: torch.FloatTensor) -> torch.FloatTensor:
        enc = self.whisper.encoder(input_features=input_features).last_hidden_state
        h = self.proj(enc)
        h = self.temporal(h.permute(0, 2, 1)).permute(0, 2, 1)
        return self.out(h).squeeze(-1)              # [B, 1500] logits


def _median_smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Median filter over the frame timeline — removes 1-2 frame spikes/dropouts."""
    if k <= 1 or len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))], dtype=np.float32)


def _extract_events(probs: np.ndarray, threshold: float, merge_gap: float,
                    min_dur: float) -> List[tuple]:
    """Group a frame-probability timeline into (start_s, end_s, confidence) events."""
    binary = (probs > threshold).astype(np.float32)
    raw, in_ev, start = [], False, 0
    for f in range(len(binary)):
        if binary[f] > 0.5 and not in_ev:
            start, in_ev = f, True
        elif binary[f] <= 0.5 and in_ev:
            raw.append((start, f)); in_ev = False
    if in_ev:
        raw.append((start, len(binary)))
    if not raw:
        return []

    sec = 1.0 / FPS
    events = [((s * sec, e * sec), (s, e)) for s, e in raw]
    merged = [events[0]]
    for (s, e), (fs, fe) in events[1:]:
        (ps, pe), (pfs, pfe) = merged[-1]
        if s - pe <= merge_gap:                      # bridge small gaps (outliers)
            merged[-1] = ((ps, e), (pfs, fe))
        else:
            merged.append(((s, e), (fs, fe)))

    out = []
    for (s, e), (fs, fe) in merged:
        if e - s >= min_dur:
            out.append((round(s, 3), round(e, 3), round(float(probs[fs:fe].mean()), 3)))
    return out


class VocalBurstLocator:
    """Detect vocal-burst candidates across an arbitrarily long clip.

    GPU VRAM: ~1 GB.
    """

    def __init__(self, device: str = "cuda:0"):
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor
        self.device = torch.device(device)
        print(f"Loading vocal-burst locator on {device}...")
        ckpt = hf_hub_download(repo_id=LOCATOR_REPO, filename="model.pt")
        self.model = WhisperSegmenter("openai/whisper-small")
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        self.model = self.model.to(self.device).eval()
        self.fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        print("Vocal-burst locator loaded.")

    @torch.no_grad()
    def detect(self, wav: np.ndarray, threshold: float = 0.7,
               merge_gap: float = 0.3, min_dur: float = 0.5,
               smooth_frames: int = 5) -> List[Dict]:
        """Return candidate events as [{start, end, confidence, duration}].

        Args:
            wav: mono 16 kHz float32 audio of the FULL clip (any length).
            threshold: confidence cutoff (user default 0.7).
        """
        win = int(WINDOW_SECONDS * SAMPLE_RATE)
        total = len(wav)
        n_win = max(1, int(np.ceil(total / win)))

        # Build the batch of <=30 s windows (last one zero-padded to 30 s).
        windows, valid_frames = [], []
        for i in range(n_win):
            seg = wav[i * win:(i + 1) * win]
            valid_frames.append(int(round(min(WINDOW_SECONDS, (total - i * win) / SAMPLE_RATE) * FPS)))
            if len(seg) < win:
                seg = np.pad(seg, (0, win - len(seg)))
            windows.append(seg)

        feats = self.fe(windows, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        feats = feats.to(self.device)
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(feats)
        else:
            logits = self.model(feats)
        probs = torch.sigmoid(logits).float().cpu().numpy()        # [n_win, 1500]

        # Concatenate the valid portion of each window into one continuous timeline,
        # so bursts spanning a window boundary remain a single contiguous event.
        timeline = np.concatenate([probs[i, :valid_frames[i]] for i in range(n_win)])
        timeline = _median_smooth(timeline, k=smooth_frames)

        events = _extract_events(timeline, threshold, merge_gap, min_dur)
        return [{"start": s, "end": e, "confidence": c, "duration": round(e - s, 3)}
                for s, e, c in events]

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()


class SoundEffectCaptioner:
    """Caption short audio segments with laion/sound-effect-captioning-whisper.

    GPU VRAM: ~1 GB.
    """

    def __init__(self, device: str = "cuda:0"):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.device = device
        print(f"Loading sound-effect captioner on {device}...")
        # The repo's tokenizer_config.json has a malformed `extra_special_tokens`
        # field; the captioner is a fine-tuned whisper-small with the standard vocab,
        # so build the processor from the base model instead.
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            CAPTIONER_REPO).to(device).eval()
        print("Sound-effect captioner loaded.")

    @torch.no_grad()
    def caption(self, segments: List[np.ndarray], batch_size: int = 8) -> List[str]:
        """Caption a list of mono 16 kHz segments (batched)."""
        captions = []
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            feats = self.processor(batch, sampling_rate=SAMPLE_RATE,
                                   return_tensors="pt").input_features.to(self.device)
            ids = self.model.generate(feats, max_new_tokens=96)
            captions.extend(t.strip() for t in
                            self.processor.batch_decode(ids, skip_special_tokens=True))
        return captions

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()


def detect_and_caption(audio_path: str, device: str = "cuda:0",
                       threshold: float = 0.7) -> List[Dict]:
    """Convenience: locate candidates in one clip and caption them.

    Returns [{start, end, confidence, caption}] sorted by start time.
    """
    import librosa
    t0 = time.time()
    wav, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    locator = VocalBurstLocator(device=device)
    cands = locator.detect(wav, threshold=threshold)
    locator.cleanup()

    if not cands:
        print(f"  Vocal-burst pre-pass: 0 candidates ({time.time()-t0:.1f}s)")
        return []

    segments = [wav[int(c["start"] * SAMPLE_RATE):int(c["end"] * SAMPLE_RATE)]
                for c in cands]
    captioner = SoundEffectCaptioner(device=device)
    caps = captioner.caption(segments)
    captioner.cleanup()

    out = [{"start": c["start"], "end": c["end"],
            "confidence": c["confidence"], "caption": cap}
           for c, cap in zip(cands, caps)]
    print(f"  Vocal-burst pre-pass: {len(out)} candidates captioned ({time.time()-t0:.1f}s)")
    return out
