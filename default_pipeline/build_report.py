#!/usr/bin/env python3
"""Build a self-contained HTML report from a finished work directory.

Embeds each clip's audio as base64 and renders the MOSS predictions (speech /
vocal-burst / sound-event segments) with the expressive emotion & speaking-style
captions. Raw three-ASR transcripts and raw SFX detections are shown in collapsible
sections. An explanation of the pipeline is rendered at the top.

Usage::

    python build_report.py --workdir ./uaap_work --out report.html
"""
import argparse, base64, html, json
from pathlib import Path


def esc(x):
    return html.escape(str(x)) if x is not None else ""


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else []


def b64_audio(path):
    mt = "audio/mpeg" if str(path).lower().endswith(".mp3") else "audio/wav"
    return f"data:{mt};base64," + base64.b64encode(Path(path).read_bytes()).decode()


CSS = """<style>
:root{--bg:#0f1216;--card:#1a1f27;--ink:#e6e9ef;--mut:#9aa6b2;--acc:#6db3f2;--line:#2a313c;}
*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;padding:32px}
.wrap{max-width:1040px;margin:0 auto}h1{font-size:26px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:0 0 22px}
.note{background:#12171d;border-left:3px solid var(--acc);padding:10px 14px;border-radius:6px;color:#cdd6e0;font-size:14px}
.clip-h{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px}.clip-h h3{margin:0;font-size:18px;color:var(--acc)}
.meta{color:var(--mut);font-size:13px}audio{width:100%;margin:12px 0 16px}
table{width:100%;border-collapse:collapse;font-size:13.5px}th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.t{font-variant-numeric:tabular-nums;color:var(--mut);white-space:nowrap}
.tag{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11px;font-weight:600}
.speech{background:#16324a;color:#9fd0ff}.vocal_burst{background:#3a2a4d;color:#d6b3ff}.sound_event{background:#2e3a23;color:#bfe39a}.fill{background:#3a3320;color:#e0cf9a}
.emo{color:#ffd9a0}.sty{color:#a0e0c8}.spk{color:var(--mut);font-size:12px}
details{margin-top:12px}summary{cursor:pointer;color:var(--mut);font-size:13px}.asr{font-size:13px;color:#cdd6e0;margin:6px 0}.asr b{color:var(--acc)}
.empty{color:var(--mut);font-style:italic}
</style>"""

EXPLAIN = """<div class="card">
<h2>How this pipeline works (default configuration)</h2>
<ol>
<li><b>Three independent ASR systems</b> transcribe each clip and are reconciled by majority vote:
<b>VibeVoice-ASR</b> (built-in diarization), <b>Parakeet TDT v3</b> + <b>Sortformer</b>, and
<b>Qwen3-ASR-1.7B</b> + ForcedAligner.</li>
<li><b>Whisper experts</b> tag each utterance with raw emotion / timbre / speaking-style.</li>
<li><b>SFX LoRA</b> (MOSS-Audio-8B-Instruct + laion/moss-audio-sfx-lora-v4) proposes timestamped sound events.</li>
<li><b>Vocal-burst pre-pass</b> (laion/vocalburst-locator @ 0.7 + laion/sound-effect-captioning-whisper)
proposes extra candidate sound effects for MOSS to verify.</li>
<li><b>MOSS-Audio-8B-Thinking</b> listens to the audio with all the above context and writes the final
structured annotation. Decoding is <b>greedy</b>.</li>
</ol>
<p class="note"><b>Expressive captions:</b> <i>emotion</i> and <i>speaking_style</i> are short, precise
captions (EmoNet vocabulary, intensity modifiers, emotion blends), not single labels. Speaker count comes
only from the diarization (VibeVoice first, Sortformer fallback) — sound-event captions never create
speakers. Singing is flagged explicitly under <i>speaking_style</i>. The three ASR transcripts vote on the
words (two outvote one; single-ASR content is still kept). The annotation covers the <b>full timeline</b>:
any span MOSS leaves uncovered is filled from the SFX predictions (or marked silence) and shown as
<span class="tag fill">timeline fill</span>.</p>
</div>"""


def render(seg):
    t = seg.get("type")
    tm = f'{seg.get("start_time","?")}&ndash;{seg.get("end_time","?")}s'
    if t == "speech":
        extra = " &middot; ".join(filter(None, [esc(seg.get(k, "")) for k in
                 ("age", "gender", "language", "accent", "speaking_rate", "voice_timbre")]))
        return (f'<tr><td class="t">{tm}</td><td><span class="tag speech">speech</span>'
                f'<div class="spk">speaker {esc(seg.get("speaker_id","?"))}</div></td>'
                f'<td><div>{esc(seg.get("transcription",""))}</div>'
                f'<div class="emo">&#9829; {esc(seg.get("emotion",""))}</div>'
                f'<div class="sty">&#9835; {esc(seg.get("speaking_style",""))}</div>'
                f'<div class="meta">{extra}</div></td></tr>')
    if t == "vocal_burst":
        return (f'<tr><td class="t">{tm}</td><td><span class="tag vocal_burst">vocal burst</span>'
                f'<div class="spk">speaker {esc(seg.get("speaker_id","?"))}</div></td>'
                f'<td><div>{esc(seg.get("vocal_burst",""))}</div>'
                f'<div class="emo">&#9829; {esc(seg.get("emotion",""))}</div></td></tr>')
    fill = ' <span class="tag fill">timeline fill</span>' if seg.get("source")=="timeline_fill" else ""
    return (f'<tr><td class="t">{tm}</td><td><span class="tag sound_event">sound</span>{fill}</td>'
            f'<td><div>{esc(seg.get("description",""))}</div>'
            f'<div class="meta">loudness: {esc(seg.get("loudness",""))}</div></td></tr>')


def asr_line(name, utts):
    txt = " ".join(u.get("content", "") for u in utts).strip() or "&mdash;"
    return f'<div class="asr"><b>{name}:</b> {esc(txt)}</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="uaap_work")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--title", default="Universal Audio Annotation Pipeline — Report")
    args = ap.parse_args()

    work = Path(args.workdir)
    index = json.loads((work / "index.json").read_text())
    cards = []
    for it in index:
        wd = it["workdir"]
        preds = load(f"{wd}/predictions.json")
        rows = "".join(render(s) for s in sorted(preds, key=lambda x: float(x.get("start_time", 0) or 0)))
        table = (f'<table><tr><th>time</th><th>type</th><th>annotation</th></tr>{rows}</table>'
                 if rows else '<p class="empty">No annotations returned for this clip.</p>')
        asr = ("<details><summary>Raw transcripts from the three ASR systems</summary>"
               + asr_line("VibeVoice", load(f"{wd}/vibevoice.json"))
               + asr_line("Parakeet", load(f"{wd}/parakeet.json"))
               + asr_line("Qwen3-ASR", load(f"{wd}/qwen3.json")) + "</details>")
        sfx_events = load(f"{wd}/sfx.json")
        sfx = ""
        if sfx_events:
            items = "".join(
                f'<div class="asr"><span class="t">{e.get("start_time","?")}&ndash;{e.get("end_time","?")}s</span> '
                f'{esc(e.get("caption", e.get("description","")))}</div>'
                for e in sorted(sfx_events, key=lambda x: float(x.get("start_time", 0) or 0)))
            sfx = (f"<details><summary>Raw SFX LoRA detections ({len(sfx_events)})</summary>{items}</details>")
        vb_events = load(f"{wd}/vocalburst.json")
        vb = ""
        if vb_events:
            vitems = "".join(
                f'<div class="asr"><span class="t">{e.get("start","?")}&ndash;{e.get("end","?")}s</span> '
                f'<b>conf {e.get("confidence","?")}</b> &mdash; {esc(e.get("caption",""))}</div>'
                for e in sorted(vb_events, key=lambda x: float(x.get("start", 0) or 0)))
            vb = (f"<details><summary>Vocal-burst candidates (specialist locator, threshold 0.7) "
                  f"+ captions ({len(vb_events)})</summary>{vitems}</details>")
        cards.append(f'<div class="card"><div class="clip-h"><h3>{esc(it["stem"])}</h3></div>'
                     f'<audio controls preload="none" src="{b64_audio(it["audio"])}"></audio>'
                     f'{table}{asr}{sfx}{vb}</div>')

    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>{esc(args.title)}</title>{CSS}</head><body><div class="wrap">'
           f'<h1>{esc(args.title)}</h1><p class="sub">Triple-ASR ensemble &middot; greedy decoding '
           f'&middot; expressive emotion &amp; speaking-style captions</p>{EXPLAIN}'
           f'{"".join(cards)}<p class="meta">Generated from {len(index)} clip(s).</p></div></body></html>')
    Path(args.out).write_text(doc, encoding="utf-8")
    print(f"Wrote {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
