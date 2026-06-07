"""pyannote speaker diarization + overlap detection (pyannote/segmentation-3.0 via the 3.1 pipeline).

Used by the default pipeline to (a) provide the speaker timeline that conditions DiCoW, and (b) detect
overlapping-speech regions. Gated models — needs an HF token with the pyannote terms accepted.
"""
import torch


def _patch_torch_load():
    """torch>=2.6 defaults torch.load(weights_only=True), which rejects pyannote's (trusted) ckpts."""
    _orig = torch.load
    def _trusting(*a, **k):
        k["weights_only"] = False
        return _orig(*a, **k)
    torch.load = _trusting


class PyannoteDiarizer:
    MODEL = "pyannote/speaker-diarization-3.1"

    def __init__(self, device: str = "cuda:0", token: str = None):
        _patch_torch_load()
        from pyannote.audio import Pipeline
        self.pipe = Pipeline.from_pretrained(self.MODEL, use_auth_token=token)
        self.pipe.to(torch.device(device))
        print(f"pyannote {self.MODEL} loaded on {device}")

    def run(self, audio_path: str):
        """Return (turns, overlaps): turns=[{start,end,speaker:int}], overlaps=[{start,end}]."""
        diar = self.pipe(audio_path)
        turns, spk_ids = [], {}
        for turn, _, label in diar.itertracks(yield_label=True):
            sid = spk_ids.setdefault(label, len(spk_ids))
            turns.append({"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": sid})
        turns.sort(key=lambda x: x["start"])
        return turns, self._overlaps(turns)

    @staticmethod
    def _overlaps(turns, min_dur: float = 0.15):
        """Spans covered by >=2 distinct speakers simultaneously."""
        pts = []
        for t in turns:
            pts.append((t["start"], 1)); pts.append((t["end"], -1))
        pts.sort()
        out, depth, st = [], 0, None
        for x, d in pts:
            prev = depth; depth += d
            if prev < 2 <= depth:
                st = x
            elif prev >= 2 > depth and st is not None:
                if x - st >= min_dur:
                    out.append({"start": round(st, 2), "end": round(x, 2)})
                st = None
        return out
