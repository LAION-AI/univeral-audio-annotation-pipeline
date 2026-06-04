#!/usr/bin/env python3
"""Stage 0 — prepare a work directory from input audio.

Decodes every input clip to a canonical 24 kHz mono WAV (so each downstream model
gets a format it can read without per-codec surprises) and writes ``index.json``.

Usage::

    python prepare_audio.py --audio /path/to/clips_or_file --workdir ./uaap_work

``--audio`` may be a single file or a directory of audio files (wav/mp3/flac/m4a/ogg).
The original file is kept; predictions are later written next to it as ``<name>_pred.json``.
"""
import argparse
import json
import subprocess
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}


def main():
    ap = argparse.ArgumentParser(description="Prepare audio work directory")
    ap.add_argument("--audio", required=True, help="Audio file or directory")
    ap.add_argument("--workdir", default="uaap_work", help="Output work directory")
    ap.add_argument("--sr", type=int, default=24000, help="Canonical WAV sample rate")
    args = ap.parse_args()

    src = Path(args.audio)
    files = ([src] if src.is_file()
             else sorted(p for p in src.iterdir() if p.suffix.lower() in AUDIO_EXTS))
    if not files:
        raise SystemExit(f"No audio files found at {src}")

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    index = []
    for f in files:
        stem = f.stem
        d = work / stem
        d.mkdir(exist_ok=True)
        wav = d / "audio_24k.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(f), "-ac", "1", "-ar", str(args.sr), str(wav)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        index.append({"stem": stem, "audio": str(f.resolve()),
                      "wav": str(wav.resolve()), "workdir": str(d.resolve())})

    (work / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Prepared {len(index)} clip(s) -> {work}/index.json")


if __name__ == "__main__":
    main()
