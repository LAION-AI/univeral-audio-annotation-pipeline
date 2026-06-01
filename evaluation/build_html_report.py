#!/usr/bin/env python3
"""Build HTML report comparing all ASR conditions + prompt variants."""

import json
import re
from pathlib import Path

EVAL_DIR = Path("/tmp/moss_eval")

# ── Data sources ──
VV_SYNTH_DIR = EVAL_DIR / "full_eval_50"
VV_YT_DIR = EVAL_DIR / "full_eval_yt"
ENS_DIR = EVAL_DIR / "full_eval_ensemble"
TRI_DIR = EVAL_DIR / "full_eval_triple"
TRI2_DIR = EVAL_DIR / "full_eval_triple_v2"
SCENES_DIR = EVAL_DIR / "scenes"
YT_SCENES_DIR = EVAL_DIR / "scenes_yt"
LORA_PRED_DIR = EVAL_DIR / "lora_predictions"
YT_LORA_DIR = EVAL_DIR / "lora_predictions_yt"
ENS_ASR_DIR = EVAL_DIR / "ensemble_asr"

DIM_LABELS = {
    "transcription_accuracy": "Transcription",
    "segment_completeness": "Completeness",
    "age_gender_accuracy": "Age/Gender",
    "timestamp_accuracy": "Timestamps",
    "speaker_diarization": "Diarization",
    "sound_event_accuracy": "SFX",
    "emotion_accuracy": "Emotion",
    "overall_quality": "Quality",
    "vocal_burst_accuracy": "Bursts",
}
DIM_ORDER = list(DIM_LABELS.keys())

SCENE_TYPES = {
    range(0, 5): "Solo speaker",
    range(5, 10): "Solo + burst",
    range(10, 18): "Dialog",
    range(18, 25): "Dialog + burst",
    range(25, 30): "Three speakers",
    range(30, 35): "Three + burst",
    range(35, 40): "Solo + 2 bursts",
    range(40, 45): "Overlapping",
    range(45, 50): "Dense",
}

def get_scene_type(sid):
    for rng, label in SCENE_TYPES.items():
        if sid in rng:
            return label
    return "Unknown"

def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def score_color(val):
    if not isinstance(val, (int, float)): return "#888"
    if val >= 4: return "#2ecc71"
    elif val >= 3: return "#f39c12"
    elif val >= 2: return "#e67e22"
    else: return "#e74c3c"

def delta_color(val):
    if val > 0.15: return "#2ecc71"
    elif val > 0.05: return "#85c88a"
    elif val > -0.05: return "#aaa"
    elif val > -0.15: return "#e6a06e"
    else: return "#e74c3c"

def delta_arrow(val):
    if val > 0.05: return "▲"
    elif val < -0.05: return "▼"
    return "─"

# ══════════════════════════════════════════════════════════════
# Load all data
# ══════════════════════════════════════════════════════════════

# ── VibeVoice synthetic ──
vv_synth_report = json.load(open(VV_SYNTH_DIR / "eval_report.json"))
vv_synth = {}
for cfg in vv_synth_report["configs"]:
    vv_synth[cfg["config"]] = cfg

# ── VibeVoice YT ──
vv_yt_report = json.load(open(VV_YT_DIR / "eval_report.json"))
vv_yt = {}
for cfg in vv_yt_report["configs"]:
    vv_yt[cfg["config"]] = cfg

# ── Ensemble ──
ens_report = json.load(open(ENS_DIR / "eval_report.json"))
ens = {}
for cfg in ens_report["configs"]:
    ens[cfg["config"]] = cfg

# ── Triple v1 ──
tri_report = json.load(open(TRI_DIR / "eval_report.json"))
tri = {}
for cfg in tri_report["configs"]:
    tri[cfg["config"]] = cfg

# ── Triple v2 (layered prompt) ──
tri2_report = json.load(open(TRI2_DIR / "eval_report.json"))
tri2 = {}
for cfg in tri2_report["configs"]:
    tri2[cfg["config"]] = cfg

# ── Compute EQUAL-WEIGHT combined scores for all systems ──
# Equal weight: (synth_avg + yt_avg) / 2 — not scene-count weighted

def equal_weight_combined(synth_dims, yt_dims):
    """Average synth and YT dimension scores equally."""
    merged = {}
    for d in DIM_ORDER:
        sv = synth_dims.get(d, 0)
        yv = yt_dims.get(d, 0)
        merged[d] = round((sv + yv) / 2, 2)
    overall = round(sum(merged.values()) / len(merged), 2)
    return {"overall": overall, "dims": merged}

# VV combined (equal weight)
vv_combined = {}
for cfgname in ["greedy", "temp05"]:
    s_dims = vv_synth.get(cfgname, {}).get("dimensions", {})
    y_dims = vv_yt.get(cfgname, {}).get("dimensions", {})
    if s_dims and y_dims:
        vv_combined[cfgname] = equal_weight_combined(s_dims, y_dims)

# Ensemble combined (equal weight — override scene-count weighted from eval_report)
for cfgname in ["greedy", "temp05"]:
    cfg = ens.get(cfgname, {})
    s_dims = cfg.get("synthetic", {}).get("dims", {})
    y_dims = cfg.get("yt", {}).get("dims", {})
    if s_dims and y_dims:
        eq = equal_weight_combined(s_dims, y_dims)
        cfg["combined_eq"] = eq

# Triple v1 combined (equal weight)
for cfgname in ["greedy", "temp05"]:
    cfg = tri.get(cfgname, {})
    s_dims = cfg.get("synthetic", {}).get("dims", {})
    y_dims = cfg.get("yt", {}).get("dims", {})
    if s_dims and y_dims:
        eq = equal_weight_combined(s_dims, y_dims)
        cfg["combined_eq"] = eq

# Triple v2 combined (equal weight)
for cfgname in ["greedy", "temp05"]:
    cfg = tri2.get(cfgname, {})
    s_dims = cfg.get("synthetic", {}).get("dims", {})
    y_dims = cfg.get("yt", {}).get("dims", {})
    if s_dims and y_dims:
        eq = equal_weight_combined(s_dims, y_dims)
        cfg["combined_eq"] = eq

# ── Load per-scene data for best predictions (triple) ──
tri_preds = {}
tri_raws = {}
tri_scores = {}

for cfgname in ["greedy", "temp05"]:
    tri_preds[cfgname] = {}
    tri_raws[cfgname] = {}
    tri_scores[cfgname] = {}
    cfg_dir = TRI_DIR / cfgname
    if not cfg_dir.exists():
        continue
    for f in sorted(cfg_dir.glob("*_pred.json")):
        sk = f.stem.replace("_pred", "")
        with open(f) as fh:
            tri_preds[cfgname][sk] = json.load(fh)
    for f in sorted(cfg_dir.glob("*_raw.txt")):
        sk = f.stem.replace("_raw", "")
        with open(f) as fh:
            tri_raws[cfgname][sk] = fh.read()
    for f in sorted(cfg_dir.glob("*_score.json")):
        sk = f.stem.replace("_score", "")
        with open(f) as fh:
            tri_scores[cfgname][sk] = json.load(fh)

# ── Load ensemble per-scene data too (for comparison in cards) ──
ens_preds = {}
ens_raws = {}
ens_scores = {}

for cfgname in ["greedy", "temp05"]:
    ens_preds[cfgname] = {}
    ens_raws[cfgname] = {}
    ens_scores[cfgname] = {}
    cfg_dir = ENS_DIR / cfgname
    if not cfg_dir.exists():
        continue
    for f in sorted(cfg_dir.glob("*_pred.json")):
        sk = f.stem.replace("_pred", "")
        with open(f) as fh:
            ens_preds[cfgname][sk] = json.load(fh)
    for f in sorted(cfg_dir.glob("*_raw.txt")):
        sk = f.stem.replace("_raw", "")
        with open(f) as fh:
            ens_raws[cfgname][sk] = fh.read()
    for f in sorted(cfg_dir.glob("*_score.json")):
        sk = f.stem.replace("_score", "")
        with open(f) as fh:
            ens_scores[cfgname][sk] = json.load(fh)

# ── Load VV per-scene scores for comparison ──
vv_synth_scores = {}
vv_yt_scores = {}
vv_synth_preds = {}
vv_yt_preds = {}
for cfgname in ["greedy", "temp05"]:
    vv_synth_scores[cfgname] = {}
    vv_yt_scores[cfgname] = {}
    vv_synth_preds[cfgname] = {}
    vv_yt_preds[cfgname] = {}
    sd = VV_SYNTH_DIR / cfgname
    if sd.exists():
        for f in sorted(sd.glob("scene_*_score.json")):
            sid = int(f.stem.split("_")[1])
            with open(f) as fh:
                vv_synth_scores[cfgname][sid] = json.load(fh)
        for f in sorted(sd.glob("scene_*_pred.json")):
            sid = int(f.stem.split("_")[1])
            with open(f) as fh:
                vv_synth_preds[cfgname][f"synth_{sid:02d}"] = json.load(fh)
    yd = VV_YT_DIR / cfgname
    if yd.exists():
        for f in sorted(yd.glob("scene_*_score.json")):
            sid = int(f.stem.split("_")[1])
            with open(f) as fh:
                vv_yt_scores[cfgname][sid] = json.load(fh)
        for f in sorted(yd.glob("scene_*_pred.json")):
            sid = int(f.stem.split("_")[1])
            with open(f) as fh:
                vv_yt_preds[cfgname][f"yt_{sid:02d}"] = json.load(fh)

# ── GT ──
gt_data = {}
for sid in range(50):
    p = SCENES_DIR / f"scene_{sid:02d}_gt.json"
    if p.exists():
        with open(p) as f:
            gt_data[sid] = json.load(f)

# ── Ensemble ASR data ──
ens_asr_parakeet = {}
ens_asr_qwen3 = {}
ens_whisper = {}
for f in sorted(ENS_ASR_DIR.glob("*_asr_parakeet.json")):
    sk = f.stem.replace("_asr_parakeet", "")
    with open(f) as fh:
        ens_asr_parakeet[sk] = json.load(fh)
for f in sorted(ENS_ASR_DIR.glob("*_asr_qwen3.json")):
    sk = f.stem.replace("_asr_qwen3", "")
    with open(f) as fh:
        ens_asr_qwen3[sk] = json.load(fh)
for f in sorted(ENS_ASR_DIR.glob("*_whisper.json")):
    sk = f.stem.replace("_whisper", "")
    with open(f) as fh:
        ens_whisper[sk] = json.load(fh)

# ── VV ASR data ──
vv_asr = {}
for sid in range(50):
    p = SCENES_DIR / f"scene_{sid:02d}_asr.json"
    if p.exists():
        with open(p) as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                data = data.get("primary_utterances", data.get("utterances", []))
            vv_asr[f"synth_{sid:02d}"] = data
for sid in range(10):
    p = YT_SCENES_DIR / f"scene_{sid:02d}_asr.json"
    if p.exists():
        with open(p) as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                data = data.get("primary_utterances", data.get("utterances", []))
            vv_asr[f"yt_{sid:02d}"] = data

# ══════════════════════════════════════════════════════════════
# Build HTML
# ══════════════════════════════════════════════════════════════

html = []
h = html.append

h("""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>ASR Ablation: VibeVoice vs Ensemble vs Triple (All 3 Conditions)</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; padding: 20px; max-width: 1600px; margin: 0 auto; }
h1 { color: #58a6ff; font-size: 1.8em; margin-bottom: 8px; }
h2 { color: #58a6ff; font-size: 1.4em; margin: 30px 0 12px; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
h3 { color: #8b949e; font-size: 1.1em; margin: 20px 0 8px; }
.subtitle { color: #8b949e; font-size: 0.95em; margin-bottom: 20px; }
.banner { background: linear-gradient(135deg, #1a1f35, #1a2940); border: 1px solid #30363d;
  border-radius: 10px; padding: 24px; margin: 20px 0; }
.banner h2 { border: none; margin: 0 0 12px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { padding: 8px 12px; text-align: center; border: 1px solid #30363d; font-size: 0.88em; }
th { background: #161b22; color: #58a6ff; font-weight: 600; }
td { background: #0d1117; }
.tbl-row-header { text-align: left; color: #c9d1d9; font-weight: 500; background: #161b22 !important; }
.best { font-weight: 700; }
.finding { background: #161b22; border-left: 3px solid #58a6ff; padding: 12px 16px; margin: 10px 0; border-radius: 4px; }
.finding strong { color: #58a6ff; }
.finding.winner { border-left-color: #2ecc71; background: rgba(46,204,113,0.05); }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin: 16px 0; overflow: hidden; }
.card-header { background: #1c2333; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; cursor: pointer; }
.card-header:hover { background: #22293a; }
.card-title { font-weight: 600; color: #58a6ff; }
.card-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.82em; font-weight: 600; }
.card-body { padding: 16px; display: none; }
.card.open .card-body { display: block; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.78em; margin: 2px; }
.three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.side-by-side { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.panel { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }
.panel-title { color: #58a6ff; font-size: 0.9em; font-weight: 600; margin-bottom: 8px; display: flex; justify-content: space-between; }
.pred-table { width: 100%; font-size: 0.8em; }
.pred-table th { font-size: 0.78em; padding: 4px 6px; }
.pred-table td { font-size: 0.78em; padding: 4px 6px; text-align: left; }
.raw-text { background: #0d1117; border: 1px solid #30363d; padding: 10px; border-radius: 4px;
  font-family: monospace; font-size: 0.75em; white-space: pre-wrap; word-break: break-all;
  max-height: 300px; overflow-y: auto; color: #8b949e; }
.think-text { color: #6e40aa; }
audio { width: 100%; margin: 8px 0; height: 36px; }
.dim-grid { display: grid; grid-template-columns: repeat(9, 1fr); gap: 4px; margin: 8px 0; }
.dim-cell { text-align: center; padding: 4px; border-radius: 4px; font-size: 0.78em; }
.dim-label { font-size: 0.65em; color: #8b949e; display: block; }
.scene-type-tag { background: #1c2333; color: #8b949e; padding: 2px 8px; border-radius: 8px; font-size: 0.75em; }
.delta-table td { font-size: 0.85em; }
.winner-row { background: rgba(46,204,113,0.08) !important; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
.toc { background: #161b22; border: 1px solid #30363d; padding: 16px; border-radius: 8px; margin: 16px 0; }
.toc a { display: block; padding: 3px 0; font-size: 0.9em; }
.highlight-cell { box-shadow: inset 0 0 0 2px #2ecc71; }
.asr-label { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; }
.asr-vv { background: #1a3a5c; color: #58a6ff; }
.asr-ens { background: #3a2a1a; color: #f0883e; }
.asr-tri { background: #1a3a2a; color: #2ecc71; }
.asr-tri2 { background: #2a2a3a; color: #d2a8ff; }
</style>
<script>
function toggleCard(el) {
  el.closest('.card').classList.toggle('open');
}
function openAll() {
  document.querySelectorAll('.card').forEach(c => c.classList.add('open'));
}
function closeAll() {
  document.querySelectorAll('.card').forEach(c => c.classList.remove('open'));
}
</script>
</head><body>
""")

# ── Title ──
h('<h1>ASR Ablation: VibeVoice vs Ensemble vs Triple ASR</h1>')
h('<div class="subtitle">MOSS-Audio-8B-Thinking &middot; 60 scenes (50 synthetic + 10 YouTube) &middot; Gemini 3.1 Pro judge &middot; 9 dimensions &middot; 6 configs</div>')

# ── TOC ──
h('<div class="toc">')
h('<strong>Contents</strong>')
h('<a href="#results">1. Grand Summary (all 6 configs)</a>')
h('<a href="#comparison">2. Dimension-Level Comparison</a>')
h('<a href="#findings">3. Key Findings &amp; Conclusions</a>')
h('<a href="#scene-type">4. Results by Scene Type</a>')
h('<a href="#tri-synth">5. Best Predictions (Triple ASR): Synthetic Scenes</a>')
h('<a href="#tri-yt">6. Best Predictions (Triple ASR): YouTube Scenes</a>')
h('</div>')

# ══════════════════════════════════════════════════════════════
# Section 1: Grand Summary
# ══════════════════════════════════════════════════════════════
h('<h2 id="results">1. Grand Summary</h2>')

h('<div class="banner">')
h('<h3>Overall Scores (9-dimension average) &mdash; Equal-Weighted: (Synth + YT) / 2</h3>')
h('<table>')
h('<tr><th>ASR System</th><th>Decoding</th><th>Synthetic (50)</th><th>YouTube (10)</th><th>Combined (equal wt)</th></tr>')

rows = []
# VV greedy
vv_g_s = vv_synth.get("greedy", {}).get("overall_avg", 0)
vv_g_y = vv_yt.get("greedy", {}).get("overall_avg", 0)
vv_g_c = vv_combined.get("greedy", {}).get("overall", 0)
rows.append(("VibeVoice-ASR", "Greedy", vv_g_s, vv_g_y, vv_g_c, "vv_greedy", "asr-vv"))

# VV temp05
vv_t_s = vv_synth.get("temp05", {}).get("overall_avg", 0)
vv_t_y = vv_yt.get("temp05", {}).get("overall_avg", 0)
vv_t_c = vv_combined.get("temp05", {}).get("overall", 0)
rows.append(("VibeVoice-ASR", "Temp 0.5", vv_t_s, vv_t_y, vv_t_c, "vv_temp05", "asr-vv"))

# Ens greedy
e_g = ens.get("greedy", {})
e_g_s = e_g.get("synthetic", {}).get("overall", 0)
e_g_y = e_g.get("yt", {}).get("overall", 0)
e_g_c = e_g.get("combined_eq", {}).get("overall", 0)
rows.append(("Ensemble (Parakeet+Qwen3)", "Greedy", e_g_s, e_g_y, e_g_c, "ens_greedy", "asr-ens"))

# Ens temp05
e_t = ens.get("temp05", {})
e_t_s = e_t.get("synthetic", {}).get("overall", 0)
e_t_y = e_t.get("yt", {}).get("overall", 0)
e_t_c = e_t.get("combined_eq", {}).get("overall", 0)
rows.append(("Ensemble (Parakeet+Qwen3)", "Temp 0.5", e_t_s, e_t_y, e_t_c, "ens_temp05", "asr-ens"))

# Triple greedy
t_g = tri.get("greedy", {})
t_g_s = t_g.get("synthetic", {}).get("overall", 0)
t_g_y = t_g.get("yt", {}).get("overall", 0)
t_g_c = t_g.get("combined_eq", {}).get("overall", 0)
rows.append(("Triple (VV+Parakeet+Qwen3)", "Greedy", t_g_s, t_g_y, t_g_c, "tri_greedy", "asr-tri"))

# Triple temp05
t_t = tri.get("temp05", {})
t_t_s = t_t.get("synthetic", {}).get("overall", 0)
t_t_y = t_t.get("yt", {}).get("overall", 0)
t_t_c = t_t.get("combined_eq", {}).get("overall", 0)
rows.append(("Triple (VV+Parakeet+Qwen3)", "Temp 0.5", t_t_s, t_t_y, t_t_c, "tri_temp05", "asr-tri"))

# Triple v2 greedy (layered prompt)
t2_g = tri2.get("greedy", {})
t2_g_s = t2_g.get("synthetic", {}).get("overall", 0)
t2_g_y = t2_g.get("yt", {}).get("overall", 0)
t2_g_c = t2_g.get("combined_eq", {}).get("overall", 0)
rows.append(("Triple v2-layered", "Greedy", t2_g_s, t2_g_y, t2_g_c, "tri2_greedy", "asr-tri2"))

# Triple v2 temp05
t2_t = tri2.get("temp05", {})
t2_t_s = t2_t.get("synthetic", {}).get("overall", 0)
t2_t_y = t2_t.get("yt", {}).get("overall", 0)
t2_t_c = t2_t.get("combined_eq", {}).get("overall", 0)
rows.append(("Triple v2-layered", "Temp 0.5", t2_t_s, t2_t_y, t2_t_c, "tri2_temp05", "asr-tri2"))

# Find best per column
best_s = max(r[2] for r in rows)
best_y = max(r[3] for r in rows)
best_c = max(r[4] for r in rows)

for asr, dec, s, y, c, key, css_class in rows:
    s_bold = ' class="best"' if s == best_s else ''
    y_bold = ' class="best"' if y == best_y else ''
    c_bold = ' class="best"' if c == best_c else ''
    is_winner = (c == best_c)
    row_class = ' class="winner-row"' if is_winner else ''
    h(f'<tr{row_class}><td class="tbl-row-header"><span class="asr-label {css_class}">{asr}</span></td><td>{dec}</td>'
      f'<td{s_bold} style="color:{score_color(s)}">{s:.2f}</td>'
      f'<td{y_bold} style="color:{score_color(y)}">{y:.2f}</td>'
      f'<td{c_bold} style="color:{score_color(c)}">{c:.2f}</td></tr>')

h('</table>')

# Absolute winner highlight - find winner dynamically
winner_row = max(rows, key=lambda r: r[4])
winner_asr, winner_dec, winner_s, winner_y, winner_c = winner_row[:5]
h(f'<div style="margin-top:12px;padding:12px 16px;background:rgba(46,204,113,0.08);border:1px solid #2ecc71;border-radius:8px">')
h(f'<strong style="color:#2ecc71">Winner: {winner_asr} + {winner_dec}</strong> &mdash; '
  f'Combined (equal wt) <strong style="color:#2ecc71">{winner_c:.2f}</strong>, '
  f'Synthetic {winner_s:.2f}, YouTube {winner_y:.2f}. '
  f'Note: Combined = (Synth + YT) / 2 to weight both domains equally despite different scene counts.')
h('</div>')
h('</div>')

# ══════════════════════════════════════════════════════════════
# Section 2: Detailed Dimension Comparison
# ══════════════════════════════════════════════════════════════
h('<h2 id="comparison">2. Dimension-Level Comparison</h2>')

for scope, scope_label in [("synth", "Synthetic Scenes (50)"), ("yt", "YouTube Scenes (10)"), ("combined", "Combined (equal weight)")]:
    h(f'<h3>{scope_label}</h3>')
    h('<table class="delta-table">')
    h('<tr><th rowspan="2">Dimension</th>'
      '<th colspan="2" style="color:#58a6ff">VibeVoice</th>'
      '<th colspan="2" style="color:#f0883e">Ensemble</th>'
      '<th colspan="2" style="color:#2ecc71">Triple v1</th>'
      '<th colspan="2" style="color:#d2a8ff">Triple v2-layered</th></tr>')
    h('<tr><th>Greedy</th><th>T=0.5</th><th>Greedy</th><th>T=0.5</th><th>Greedy</th><th>T=0.5</th><th>Greedy</th><th>T=0.5</th></tr>')

    if scope == "synth":
        vv_g_dims = vv_synth.get("greedy", {}).get("dimensions", {})
        vv_t_dims = vv_synth.get("temp05", {}).get("dimensions", {})
        e_g_dims = ens.get("greedy", {}).get("synthetic", {}).get("dims", {})
        e_t_dims = ens.get("temp05", {}).get("synthetic", {}).get("dims", {})
        tr_g_dims = tri.get("greedy", {}).get("synthetic", {}).get("dims", {})
        tr_t_dims = tri.get("temp05", {}).get("synthetic", {}).get("dims", {})
        t2_g_dims = tri2.get("greedy", {}).get("synthetic", {}).get("dims", {})
        t2_t_dims = tri2.get("temp05", {}).get("synthetic", {}).get("dims", {})
    elif scope == "yt":
        vv_g_dims = vv_yt.get("greedy", {}).get("dimensions", {})
        vv_t_dims = vv_yt.get("temp05", {}).get("dimensions", {})
        e_g_dims = ens.get("greedy", {}).get("yt", {}).get("dims", {})
        e_t_dims = ens.get("temp05", {}).get("yt", {}).get("dims", {})
        tr_g_dims = tri.get("greedy", {}).get("yt", {}).get("dims", {})
        tr_t_dims = tri.get("temp05", {}).get("yt", {}).get("dims", {})
        t2_g_dims = tri2.get("greedy", {}).get("yt", {}).get("dims", {})
        t2_t_dims = tri2.get("temp05", {}).get("yt", {}).get("dims", {})
    else:  # combined (equal weight)
        vv_g_dims = vv_combined.get("greedy", {}).get("dims", {})
        vv_t_dims = vv_combined.get("temp05", {}).get("dims", {})
        e_g_dims = ens.get("greedy", {}).get("combined_eq", {}).get("dims", {})
        e_t_dims = ens.get("temp05", {}).get("combined_eq", {}).get("dims", {})
        tr_g_dims = tri.get("greedy", {}).get("combined_eq", {}).get("dims", {})
        tr_t_dims = tri.get("temp05", {}).get("combined_eq", {}).get("dims", {})
        t2_g_dims = tri2.get("greedy", {}).get("combined_eq", {}).get("dims", {})
        t2_t_dims = tri2.get("temp05", {}).get("combined_eq", {}).get("dims", {})

    for d in DIM_ORDER:
        vals = [
            vv_g_dims.get(d, 0), vv_t_dims.get(d, 0),
            e_g_dims.get(d, 0), e_t_dims.get(d, 0),
            tr_g_dims.get(d, 0), tr_t_dims.get(d, 0),
            t2_g_dims.get(d, 0), t2_t_dims.get(d, 0),
        ]
        best_val = max(vals)

        cells = []
        for v in vals:
            bold = ' class="best"' if v == best_val and v > 0 else ''
            cells.append(f'<td{bold} style="color:{score_color(v)}">{v:.2f}</td>')

        h(f'<tr><td class="tbl-row-header">{DIM_LABELS[d]}</td>{"".join(cells)}</tr>')

    # Overall row
    if scope == "synth":
        overalls = [
            vv_synth.get("greedy", {}).get("overall_avg", 0),
            vv_synth.get("temp05", {}).get("overall_avg", 0),
            ens.get("greedy", {}).get("synthetic", {}).get("overall", 0),
            ens.get("temp05", {}).get("synthetic", {}).get("overall", 0),
            tri.get("greedy", {}).get("synthetic", {}).get("overall", 0),
            tri.get("temp05", {}).get("synthetic", {}).get("overall", 0),
            tri2.get("greedy", {}).get("synthetic", {}).get("overall", 0),
            tri2.get("temp05", {}).get("synthetic", {}).get("overall", 0),
        ]
    elif scope == "yt":
        overalls = [
            vv_yt.get("greedy", {}).get("overall_avg", 0),
            vv_yt.get("temp05", {}).get("overall_avg", 0),
            ens.get("greedy", {}).get("yt", {}).get("overall", 0),
            ens.get("temp05", {}).get("yt", {}).get("overall", 0),
            tri.get("greedy", {}).get("yt", {}).get("overall", 0),
            tri.get("temp05", {}).get("yt", {}).get("overall", 0),
            tri2.get("greedy", {}).get("yt", {}).get("overall", 0),
            tri2.get("temp05", {}).get("yt", {}).get("overall", 0),
        ]
    else:  # combined equal weight
        overalls = [
            vv_combined.get("greedy", {}).get("overall", 0),
            vv_combined.get("temp05", {}).get("overall", 0),
            ens.get("greedy", {}).get("combined_eq", {}).get("overall", 0),
            ens.get("temp05", {}).get("combined_eq", {}).get("overall", 0),
            tri.get("greedy", {}).get("combined_eq", {}).get("overall", 0),
            tri.get("temp05", {}).get("combined_eq", {}).get("overall", 0),
            tri2.get("greedy", {}).get("combined_eq", {}).get("overall", 0),
            tri2.get("temp05", {}).get("combined_eq", {}).get("overall", 0),
        ]
    best_o = max(overalls)
    cells = []
    for v in overalls:
        bold = ' class="best"' if v == best_o else ''
        cells.append(f'<td{bold} style="color:{score_color(v)}"><strong>{v:.2f}</strong></td>')
    h(f'<tr style="border-top:2px solid #58a6ff"><td class="tbl-row-header"><strong>OVERALL</strong></td>{"".join(cells)}</tr>')
    h('</table>')

# ── Delta table: Triple vs VibeVoice ──
h('<h3>Improvement: Triple vs VibeVoice (best-config per scope)</h3>')
h('<table>')
h('<tr><th>Dimension</th><th>Synth: VV T=0.5</th><th>Synth: Tri T=0.5</th><th>&Delta;</th>'
  '<th>YT: VV Greedy</th><th>YT: Tri Greedy</th><th>&Delta;</th></tr>')

vv_best_s_dims = vv_synth.get("temp05", {}).get("dimensions", {})
tr_best_s_dims = tri.get("temp05", {}).get("synthetic", {}).get("dims", {})
vv_best_y_dims = vv_yt.get("greedy", {}).get("dimensions", {})
tr_best_y_dims = tri.get("greedy", {}).get("yt", {}).get("dims", {})

for d in DIM_ORDER:
    vs = vv_best_s_dims.get(d, 0)
    ts = tr_best_s_dims.get(d, 0)
    ds = ts - vs
    vy = vv_best_y_dims.get(d, 0)
    ty = tr_best_y_dims.get(d, 0)
    dy = ty - vy
    h(f'<tr><td class="tbl-row-header">{DIM_LABELS[d]}</td>'
      f'<td style="color:{score_color(vs)}">{vs:.2f}</td>'
      f'<td style="color:{score_color(ts)}">{ts:.2f}</td>'
      f'<td style="color:{delta_color(ds)}">{delta_arrow(ds)} {ds:+.2f}</td>'
      f'<td style="color:{score_color(vy)}">{vy:.2f}</td>'
      f'<td style="color:{score_color(ty)}">{ty:.2f}</td>'
      f'<td style="color:{delta_color(dy)}">{delta_arrow(dy)} {dy:+.2f}</td></tr>')

# Overall
vs_o = vv_synth.get("temp05", {}).get("overall_avg", 0)
ts_o = tri.get("temp05", {}).get("synthetic", {}).get("overall", 0)
ds_o = ts_o - vs_o
vy_o = vv_yt.get("greedy", {}).get("overall_avg", 0)
ty_o = tri.get("greedy", {}).get("yt", {}).get("overall", 0)
dy_o = ty_o - vy_o
h(f'<tr style="border-top:2px solid #58a6ff"><td class="tbl-row-header"><strong>OVERALL</strong></td>'
  f'<td style="color:{score_color(vs_o)}"><strong>{vs_o:.2f}</strong></td>'
  f'<td style="color:{score_color(ts_o)}"><strong>{ts_o:.2f}</strong></td>'
  f'<td style="color:{delta_color(ds_o)}"><strong>{delta_arrow(ds_o)} {ds_o:+.2f}</strong></td>'
  f'<td style="color:{score_color(vy_o)}"><strong>{vy_o:.2f}</strong></td>'
  f'<td style="color:{score_color(ty_o)}"><strong>{ty_o:.2f}</strong></td>'
  f'<td style="color:{delta_color(dy_o)}"><strong>{delta_arrow(dy_o)} {dy_o:+.2f}</strong></td></tr>')
h('</table>')

# ══════════════════════════════════════════════════════════════
# Section 3: Key Findings
# ══════════════════════════════════════════════════════════════
h('<h2 id="findings">3. Key Findings &amp; Conclusions</h2>')

# Recompute combined for findings with equal-weight values
t_g_c_eq = t_g.get("combined_eq", {}).get("overall", 0)
t_t_c_eq = t_t.get("combined_eq", {}).get("overall", 0)
e_g_c_eq = e_g.get("combined_eq", {}).get("overall", 0)
e_t_c_eq = e_t.get("combined_eq", {}).get("overall", 0)
vv_g_c_eq = vv_combined.get("greedy", {}).get("overall", 0)
vv_t_c_eq = vv_combined.get("temp05", {}).get("overall", 0)

findings = [
    ("winner", "Triple ASR is the clear winner",
     f"Combining ALL three ASR systems (VibeVoice + Parakeet + Qwen3) yields <strong>{t_g_c_eq:.2f}</strong> combined "
     f"(greedy, equal weight), beating VibeVoice-only ({vv_t_c_eq:.2f} temp05) by <strong>+{t_g_c_eq - vv_t_c_eq:.2f}</strong> and "
     f"Ensemble-only ({e_g_c_eq:.2f} greedy) by <strong>+{t_g_c_eq - e_g_c_eq:.2f}</strong>. "
     f"The triple-ASR reconciliation prompt lets MOSS cross-reference three independent transcriptions via majority vote."),

    ("winner", "Massive YouTube improvement with triple ASR",
     f"Triple greedy achieves <strong>{t_g_y:.2f}</strong> on YT scenes — "
     f"+{t_g_y - vv_g_y:.2f} over VibeVoice greedy ({vv_g_y:.2f}) and "
     f"+{t_g_y - e_g_y:.2f} over Ensemble greedy ({e_g_y:.2f}). "
     f"Key gains: completeness {tr_best_y_dims.get('segment_completeness', 0):.1f} (VV: {vv_best_y_dims.get('segment_completeness', 0):.1f}), "
     f"bursts {tr_best_y_dims.get('vocal_burst_accuracy', 0):.1f} (VV: {vv_best_y_dims.get('vocal_burst_accuracy', 0):.1f}), "
     f"quality {tr_best_y_dims.get('overall_quality', 0):.1f} (VV: {vv_best_y_dims.get('overall_quality', 0):.1f})."),

    ("", "Triple ASR also improves synthetic scenes",
     f"Triple temp05 achieves <strong>{t_t_s:.2f}</strong> on synthetic, matching VV temp05 ({vv_t_s:.2f}) with "
     f"+{t_t_s - vv_t_s:.2f}. Unlike Ensemble-only ({e_t_s:.2f}) which hurt synthetic by "
     f"{e_t_s - vv_t_s:+.2f}, triple ASR avoids the degradation by including VibeVoice's end-to-end ASR as a third reference."),

    ("", "Three ASR systems compensate for each other's weaknesses",
     "VibeVoice excels at clean TTS with built-in segmentation. "
     "Parakeet provides precise word-level timestamps. "
     "Qwen3-ASR catches words the others miss. "
     "Together, MOSS can cross-validate and pick the best from each — majority vote resolves disagreements, extra words get included."),

    ("", "Ensemble-only (2 ASR) is the worst config",
     f"Removing VibeVoice from the mix hurts: Ensemble combined {e_g_c_eq:.2f} greedy / {e_t_c_eq:.2f} temp05 "
     f"is <strong>worse than VibeVoice-only</strong>. The 2-ASR reconciliation creates confusion on synthetic TTS "
     f"(completeness drops from {vv_synth.get('temp05', {}).get('dimensions', {}).get('segment_completeness', 0):.2f} "
     f"to {ens.get('temp05', {}).get('synthetic', {}).get('dims', {}).get('segment_completeness', 0):.2f})."),

    ("", "Greedy decoding wins with triple ASR",
     f"With triple ASR context, greedy outperforms temp=0.5 on equal-weight combined ({t_g_c_eq:.2f} vs {t_t_c_eq:.2f}). "
     f"The richer context from 3 ASR sources reduces the need for sampling diversity. "
     f"Exception: synthetic scenes slightly prefer temp=0.5 ({t_t_s:.2f} vs {t_g_s:.2f})."),

    ("", "v2 layered prompt HURTS — aggressive completeness causes hallucination",
     f"The v2 prompt instructing exhaustive foreground/background annotation drops combined from "
     f"<strong>{t_g_c_eq:.2f}</strong> (v1) to <strong>{t2_g.get('combined_eq', {}).get('overall', 0):.2f}</strong> (v2 greedy). "
     f"SFX accuracy collapses (synth: {tri.get('greedy', {}).get('synthetic', {}).get('dims', {}).get('sound_event_accuracy', 0):.2f} → "
     f"{tri2.get('greedy', {}).get('synthetic', {}).get('dims', {}).get('sound_event_accuracy', 0):.2f}, "
     f"−{tri.get('greedy', {}).get('synthetic', {}).get('dims', {}).get('sound_event_accuracy', 0) - tri2.get('greedy', {}).get('synthetic', {}).get('dims', {}).get('sound_event_accuracy', 0):.2f}). "
     f"The model hallucinates room tone, ambient hum, and orchestral underscore in clean TTS scenes. "
     f"Lesson: 'false positives are less harmful than missing events' is wrong — Gemini heavily penalizes phantom sounds."),

    ("", "Equal weighting reveals stronger triple advantage",
     f"With equal domain weighting (Synth + YT) / 2, the triple advantage is more visible: "
     f"Triple greedy {t_g_c_eq:.2f} vs VV temp05 {vv_t_c_eq:.2f} (+{t_g_c_eq - vv_t_c_eq:.2f}). "
     f"Scene-count weighting (old) diluted the YT improvement because synthetic had 5x more scenes."),
]

for css_class, title, body in findings:
    cls = f' {css_class}' if css_class else ''
    h(f'<div class="finding{cls}"><strong>{title}.</strong> {body}</div>')

# ══════════════════════════════════════════════════════════════
# Section 4: Results by Scene Type
# ══════════════════════════════════════════════════════════════
h('<h2 id="scene-type">4. Results by Scene Type (Synthetic)</h2>')
h('<table>')
h('<tr><th>Scene Type</th><th>VV G</th><th>VV T</th><th>Ens G</th><th>Ens T</th><th>Tri G</th><th>Tri T</th><th>Best</th></tr>')

for rng, stype in SCENE_TYPES.items():
    vals = {}
    # VV
    for label, score_src in [("VV_G", vv_synth_scores.get("greedy", {})), ("VV_T", vv_synth_scores.get("temp05", {}))]:
        scene_scores = [score_src.get(sid, {}).get("overall_quality", None) for sid in rng]
        scene_scores = [s for s in scene_scores if s is not None]
        vals[label] = round(sum(scene_scores) / len(scene_scores), 2) if scene_scores else 0

    # Ensemble
    for label, cfgname in [("E_G", "greedy"), ("E_T", "temp05")]:
        scene_scores = []
        for sid in rng:
            sk = f"synth_{sid:02d}"
            sc = ens_scores.get(cfgname, {}).get(sk, {})
            oq = sc.get("overall_quality")
            if oq is not None:
                scene_scores.append(oq)
        vals[label] = round(sum(scene_scores) / len(scene_scores), 2) if scene_scores else 0

    # Triple
    for label, cfgname in [("T_G", "greedy"), ("T_T", "temp05")]:
        scene_scores = []
        for sid in rng:
            sk = f"synth_{sid:02d}"
            sc = tri_scores.get(cfgname, {}).get(sk, {})
            oq = sc.get("overall_quality")
            if oq is not None:
                scene_scores.append(oq)
        vals[label] = round(sum(scene_scores) / len(scene_scores), 2) if scene_scores else 0

    all_vals = [("VV G", vals["VV_G"]), ("VV T", vals["VV_T"]),
                ("Ens G", vals["E_G"]), ("Ens T", vals["E_T"]),
                ("Tri G", vals["T_G"]), ("Tri T", vals["T_T"])]
    best_label = max(all_vals, key=lambda x: x[1])
    best_val = best_label[1]

    h(f'<tr><td class="tbl-row-header">{stype}</td>')
    for lbl, v in all_vals:
        bold = ' class="best"' if v == best_val else ''
        h(f'<td{bold} style="color:{score_color(v)}">{v:.2f}</td>')
    h(f'<td style="color:#58a6ff">{best_label[0]}</td></tr>')
h('</table>')

# ══════════════════════════════════════════════════════════════
# Section 5+6: Per-scene cards (Triple ASR predictions)
# ══════════════════════════════════════════════════════════════

def build_prediction_table(events):
    if not events:
        return '<em style="color:#8b949e">No events parsed</em>'
    rows = []
    rows.append('<table class="pred-table"><tr><th>Type</th><th>Time</th><th>Content</th><th>Details</th></tr>')
    for e in events:
        etype = e.get("type", "?")
        st = e.get("start_time", "?")
        et = e.get("end_time", "?")
        if etype == "speech":
            content = esc(e.get("transcription", ""))[:120]
            details = f'spk={e.get("speaker_id","?")} emo={e.get("emotion","?")} {e.get("gender","?")} {e.get("age","?")}'
            color = "#58a6ff"
        elif etype == "vocal_burst":
            content = esc(e.get("vocal_burst", ""))
            details = f'spk={e.get("speaker_id","?")} emo={e.get("emotion","?")}'
            color = "#d2a8ff"
        elif etype == "sound_event":
            content = esc(e.get("description", ""))[:120]
            details = f'loud={e.get("loudness","?")}'
            color = "#f0883e"
        else:
            content = esc(str(e)[:100])
            details = ""
            color = "#8b949e"
        rows.append(f'<tr><td style="color:{color}">{etype}</td><td>{st}&ndash;{et}</td><td>{content}</td><td style="color:#8b949e;font-size:0.75em">{details}</td></tr>')
    rows.append('</table>')
    return '\n'.join(rows)


def build_raw_block(raw_text):
    if not raw_text:
        return ''
    think_match = re.search(r'<think>(.*?)</think>', raw_text, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""
    parts = []
    if think:
        parts.append(f'<div class="raw-text"><span class="think-text">&lt;think&gt;\n{esc(think[:2000])}\n&lt;/think&gt;</span></div>')
    return '\n'.join(parts)


def get_vv_score(source, sid, cfgname):
    if source == "synth":
        return vv_synth_scores.get(cfgname, {}).get(sid, {})
    else:
        return vv_yt_scores.get(cfgname, {}).get(sid, {})


def render_scene_card(sk, source, sid):
    """Render a per-scene card for triple ASR predictions (best config)."""
    # Audio path
    if source == "synth":
        audio_path = f"scenes/scene_{sid:02d}.wav"
        gt = gt_data.get(sid)
    else:
        audio_path = f"scenes_yt/scene_{sid:02d}.wav"
        gt = None

    # Scores from all 3 systems for best config comparison
    tri_sc_g = tri_scores.get("greedy", {}).get(sk, {})
    tri_sc_t = tri_scores.get("temp05", {}).get(sk, {})
    ens_sc_g = ens_scores.get("greedy", {}).get(sk, {})
    ens_sc_t = ens_scores.get("temp05", {}).get(sk, {})
    vv_sc_g = get_vv_score(source, sid, "greedy")
    vv_sc_t = get_vv_score(source, sid, "temp05")

    # Find best overall quality across all 6 configs
    configs = [
        ("VV G", vv_sc_g), ("VV T", vv_sc_t),
        ("Ens G", ens_sc_g), ("Ens T", ens_sc_t),
        ("Tri G", tri_sc_g), ("Tri T", tri_sc_t),
    ]
    oq_vals = [(lbl, sc.get("overall_quality", 0) or 0) for lbl, sc in configs]
    best_label, best_oq = max(oq_vals, key=lambda x: x[1])

    tri_oq_g = tri_sc_g.get("overall_quality", "?")
    tri_oq_t = tri_sc_t.get("overall_quality", "?")

    # Pick best triple config for display
    tri_best_cfg = "greedy" if (tri_oq_g or 0) >= (tri_oq_t or 0) else "temp05"
    tri_best_score = tri_oq_g if tri_best_cfg == "greedy" else tri_oq_t

    scene_label = get_scene_type(sid) if source == "synth" else ("Frankenstein" if sid < 5 else "Nuclear Shelter")

    card = []
    badge_color = score_color(tri_best_score) if isinstance(tri_best_score, (int, float)) else "#888"
    card.append(f'<div class="card">')
    card.append(f'<div class="card-header" onclick="toggleCard(this)">')
    card.append(f'<span class="card-title">{sk}</span>')
    card.append(f'<span><span class="scene-type-tag">{scene_label}</span> ')

    # Show all 3 system scores
    for lbl, css, sc_g, sc_t in [
        ("Tri", "asr-tri", tri_oq_g, tri_oq_t),
        ("Ens", "asr-ens", ens_sc_g.get("overall_quality", "?"), ens_sc_t.get("overall_quality", "?")),
        ("VV", "asr-vv", vv_sc_g.get("overall_quality", "?"), vv_sc_t.get("overall_quality", "?")),
    ]:
        # Show best of greedy/temp for this system
        g_val = sc_g if isinstance(sc_g, (int, float)) else 0
        t_val = sc_t if isinstance(sc_t, (int, float)) else 0
        best_v = max(g_val, t_val)
        best_d = "G" if g_val >= t_val else "T"
        c = score_color(best_v) if best_v > 0 else "#888"
        card.append(f'<span class="card-badge" style="background:{c}20;color:{c}">{lbl} {best_d}: {best_v}</span> ')

    card.append(f'<span class="card-badge" style="background:#30363d;color:#c9d1d9">Best: {best_label}</span>')
    card.append('</span></div>')

    # Body
    card.append('<div class="card-body">')
    card.append(f'<audio controls preload="none"><source src="{audio_path}" type="audio/wav"></audio>')

    # Dimension scores grid for best triple config
    sc = tri_sc_g if tri_best_cfg == "greedy" else tri_sc_t
    if sc:
        card.append('<div class="dim-grid">')
        for d in DIM_ORDER:
            v = sc.get(d, "?")
            c = score_color(v) if isinstance(v, (int, float)) else "#888"
            card.append(f'<div class="dim-cell" style="background:{c}20;color:{c}"><span class="dim-label">{DIM_LABELS[d]}</span>{v}</div>')
        card.append('</div>')

    # Gemini comment
    comment = sc.get("comments", "")
    if comment:
        card.append(f'<div style="color:#8b949e;font-size:0.82em;margin:6px 0;padding:6px 10px;background:#0d1117;border-radius:4px"><strong>Gemini:</strong> {esc(comment)}</div>')

    # Side by side: Triple greedy vs Triple temp05
    card.append('<h3 style="margin-top:12px">Triple ASR Predictions</h3>')
    card.append('<div class="side-by-side">')
    for cfg_label, cfgname in [("Triple Greedy", "greedy"), ("Triple Temp 0.5", "temp05")]:
        sc_here = tri_scores.get(cfgname, {}).get(sk, {})
        oq = sc_here.get("overall_quality", "?")
        color = score_color(oq) if isinstance(oq, (int, float)) else "#888"
        pred = tri_preds.get(cfgname, {}).get(sk, [])
        card.append(f'<div class="panel"><div class="panel-title"><span>{cfg_label}</span><span style="color:{color}">Q: {oq}</span></div>')
        card.append(build_prediction_table(pred))
        card.append('</div>')
    card.append('</div>')

    # Triple ASR sources: VibeVoice, Parakeet, Qwen3
    vv_d = vv_asr.get(sk, [])
    p_asr = ens_asr_parakeet.get(sk, [])
    q_asr = ens_asr_qwen3.get(sk, [])
    if vv_d or p_asr or q_asr:
        card.append('<details style="margin-top:12px"><summary style="color:#8b949e;cursor:pointer;font-size:0.85em">Show triple ASR sources</summary>')
        card.append('<div class="three-col">')
        for asr_label, asr_data, css in [
            ("VibeVoice-ASR", vv_d, "asr-vv"),
            ("Parakeet TDT v3", p_asr, "asr-ens"),
            ("Qwen3-ASR-1.7B", q_asr, "asr-ens"),
        ]:
            card.append(f'<div class="panel"><div class="panel-title"><span class="asr-label {css}">{asr_label}</span></div>')
            if asr_data:
                card.append('<table class="pred-table"><tr><th>Spk</th><th>Time</th><th>Text</th></tr>')
                for u in asr_data[:15]:
                    spk = u.get("speaker_id", u.get("speaker", "?"))
                    st = u.get("start_time", "?")
                    et = u.get("end_time", "?")
                    txt = esc(u.get("content", u.get("text", "")))[:80]
                    card.append(f'<tr><td>spk_{spk}</td><td>{st}&ndash;{et}</td><td>{txt}</td></tr>')
                card.append('</table>')
            else:
                card.append('<em style="color:#8b949e">No data</em>')
            card.append('</div>')
        card.append('</div>')
        card.append('</details>')

    # Thinking trace
    raw = tri_raws.get(tri_best_cfg, {}).get(sk, "")
    if raw:
        card.append('<details style="margin-top:10px"><summary style="color:#8b949e;cursor:pointer;font-size:0.85em">Show thinking trace</summary>')
        card.append(build_raw_block(raw))
        card.append('</details>')

    # GT (synthetic only)
    if gt:
        card.append('<details style="margin-top:10px"><summary style="color:#8b949e;cursor:pointer;font-size:0.85em">Show ground truth</summary>')
        card.append(build_prediction_table(gt))
        card.append('</details>')

    card.append('</div></div>')
    return '\n'.join(card)


# ── Section 5: Synthetic scene cards ──
h('<h2 id="tri-synth">5. Best Predictions (Triple ASR): Synthetic Scenes</h2>')
h('<div style="margin-bottom:12px"><button onclick="openAll()" style="background:#1c2333;color:#58a6ff;border:1px solid #30363d;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:8px">Open All</button>')
h('<button onclick="closeAll()" style="background:#1c2333;color:#8b949e;border:1px solid #30363d;padding:6px 14px;border-radius:6px;cursor:pointer">Close All</button></div>')

for sid in range(50):
    sk = f"synth_{sid:02d}"
    h(render_scene_card(sk, "synth", sid))

# ── Section 6: YT scene cards ──
h('<h2 id="tri-yt">6. Best Predictions (Triple ASR): YouTube Scenes</h2>')

for sid in range(10):
    sk = f"yt_{sid:02d}"
    h(render_scene_card(sk, "yt", sid))

# Footer
h('<div style="margin-top:40px;padding:16px;color:#8b949e;font-size:0.8em;border-top:1px solid #30363d">')
h('Generated by build_html_ensemble.py &middot; 3 ASR conditions x 2 decoding strategies = 6 configs &middot; MOSS-Audio-8B-Thinking &middot; Gemini 3.1 Pro judge')
h('</div>')
h('</body></html>')

# ── Write ──
out_path = EVAL_DIR / "ensemble_eval.html"
out_path.write_text('\n'.join(html))
print(f"Written {out_path} ({out_path.stat().st_size // 1024} KB)")
