"""DiCoW — diarization-conditioned Whisper for overlap-aware, per-speaker ASR (BUT-FIT/DiCoW_v3_3).

Transcribes EACH speaker separately, recovering speech even through overlaps where a single full-clip
ASR pass garbles or drops the quieter voice. Conditioned on an external diarization (pyannote) via STNO
masks; processes audio in <=30 s chunks. Used by the default pipeline to feed the LLM fuser clean
per-speaker transcripts for overlapping speech.
"""
import numpy as np
import torch


class DiCoWASR:
    MODEL = "BUT-FIT/DiCoW_v3_3"
    SR = 16000
    CHUNK = 30.0
    FPS = 50            # diarization-mask frame rate (mel/2)

    def __init__(self, device: str = "cuda:0"):
        from transformers import AutoModelForSpeechSeq2Seq, AutoFeatureExtractor, AutoTokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.MODEL, trust_remote_code=True).to(device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(self.MODEL)
        self.tok = AutoTokenizer.from_pretrained(self.MODEL)
        self.model.tokenizer = self.tok    # generation.py references self.tokenizer
        self.device = device
        print(f"DiCoW {self.MODEL} loaded on {device}")

    @staticmethod
    def _stno(diar_mask, s):
        nt = torch.ones((diar_mask.shape[0],), dtype=torch.bool); nt[s] = False
        sil = (1 - diar_mask).prod(0)
        anyone_else = (1 - diar_mask[nt]).prod(0) if diar_mask.shape[0] > 1 else torch.ones_like(diar_mask[0])
        tgt = diar_mask[s] * anyone_else
        nontgt = (1 - diar_mask[s]) * (1 - anyone_else)
        ov = diar_mask[s] - tgt
        return torch.stack([sil, tgt, nontgt, ov], 0)

    def _chunk(self, audio, turns_by_spk):
        feat = self.fe(audio, sampling_rate=self.SR, return_tensors="pt").input_features.to(self.device, torch.float32)
        flen = feat.shape[-1] // 2
        spks = sorted(turns_by_spk)
        if not spks:
            return {}
        dmask = torch.zeros(len(spks), flen)
        for i, sp in enumerate(spks):
            for s, e in turns_by_spk[sp]:
                dmask[i, max(0, round(s * self.FPS)):min(flen, round(e * self.FPS))] = 1
        stno = torch.stack([self._stno(dmask, i) for i in range(len(spks))], 0).to(self.device, torch.float32)
        inp = feat.repeat(len(spks), 1, 1)
        attn = torch.ones(inp.shape[0], inp.shape[2], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            out = self.model.generate(input_features=inp, attention_mask=attn, stno_mask=stno,
                                      task="transcribe", return_timestamps=False)
        texts = self.tok.batch_decode(out, skip_special_tokens=True)
        return {sp: texts[i].strip() for i, sp in enumerate(spks)}

    def run(self, audio_path: str, pyannote_turns):
        """pyannote_turns = [{start,end,speaker}]. Returns [{speaker,text,turns}]."""
        import librosa
        if not pyannote_turns:
            return []
        audio = librosa.load(audio_path, sr=self.SR, mono=True)[0]
        dur = len(audio) / self.SR
        spk_turns = {}
        for d in pyannote_turns:
            spk_turns.setdefault(d["speaker"], []).append((d["start"], d["end"]))
        per_spk = {sp: [] for sp in spk_turns}
        for c in range(max(1, int(np.ceil(dur / self.CHUNK)))):
            cs, ce = c * self.CHUNK, min(dur, (c + 1) * self.CHUNK)
            seg = audio[int(cs * self.SR):int(ce * self.SR)]
            if len(seg) < int(0.2 * self.SR):
                continue
            local = {}
            for sp, turns in spk_turns.items():
                loc = [(max(0, s - cs), min(self.CHUNK, e - cs)) for s, e in turns if e > cs and s < ce]
                if loc:
                    local[sp] = loc
            for sp, txt in self._chunk(seg, local).items():
                if txt:
                    per_spk[sp].append(txt)
        return [{"speaker": int(sp), "text": " ".join(per_spk[sp]).strip(),
                 "turns": [[round(s, 2), round(e, 2)] for s, e in spk_turns[sp]]}
                for sp in sorted(spk_turns) if " ".join(per_spk[sp]).strip()]
