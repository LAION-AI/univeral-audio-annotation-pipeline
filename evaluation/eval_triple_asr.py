#!/usr/bin/env python3
"""
Triple ASR evaluation: VibeVoice + Parakeet TDT v3 + Qwen3-ASR + Sortformer
All ASR data already computed — this script only runs MOSS inference + Gemini judging.

Context includes THREE independent ASR transcriptions for maximum reconciliation.
"""

import json
import os
import re
import sys
import time
import base64
import traceback
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_XET"] = "1"

EVAL_DIR = Path("/tmp/moss_eval")
SYNTH_SCENES_DIR = EVAL_DIR / "scenes"
YT_SCENES_DIR = EVAL_DIR / "scenes_yt"
SYNTH_SEGMENTS_DIR = EVAL_DIR / "segments"
SYNTH_LORA_DIR = EVAL_DIR / "lora_predictions"
YT_LORA_DIR = EVAL_DIR / "lora_predictions_yt"
ENS_ASR_DIR = EVAL_DIR / "ensemble_asr"

OUTPUT_DIR = EVAL_DIR / "full_eval_triple"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_THINKING = "OpenMOSS-Team/MOSS-Audio-8B-Thinking"
MAX_NEW_TOKENS = 16384


# ════════════════════════════════════════════════════════════
# Scene list
# ════════════════════════════════════════════════════════════

def get_all_scenes():
    scenes = []
    for i in range(50):
        p = SYNTH_SCENES_DIR / f"scene_{i:02d}.wav"
        if p.exists():
            scenes.append(("synth", i, p))
    for i in range(10):
        p = YT_SCENES_DIR / f"scene_{i:02d}.wav"
        if p.exists():
            scenes.append(("yt", i, p))
    return scenes


def skey(source, sid):
    return f"{source}_{sid:02d}"


def get_gt_path(source, sid):
    if source == "synth":
        return SYNTH_SCENES_DIR / f"scene_{sid:02d}_gt.json"
    return None


def get_lora_path(source, sid):
    if source == "synth":
        return SYNTH_LORA_DIR / f"scene_{sid:02d}_medium_overlap.json"
    else:
        return YT_LORA_DIR / f"scene_{sid:02d}_medium_overlap.json"


# ════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════

def strip_thinking(text):
    result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    if result.startswith("<think>"):
        return ""
    return result


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    objects = []
    for m in re.finditer(r'\{[^{}]*\}', text):
        try:
            objects.append(json.loads(m.group()))
        except Exception:
            pass
    return objects if objects else None


def dedup_events(events):
    if not events:
        return events
    seen = []
    for evt in events:
        is_dup = False
        for s in seen:
            if evt.get("type") != s.get("type"):
                continue
            s1 = float(evt.get("start_time", 0))
            e1 = float(evt.get("end_time", 0))
            s2 = float(s.get("start_time", 0))
            e2 = float(s.get("end_time", 0))
            overlap = max(0, min(e1, e2) - max(s1, s2))
            duration = max(e1 - s1, e2 - s2, 0.01)
            if overlap / duration < 0.7:
                continue
            if evt.get("type") == "speech":
                if evt.get("transcription", "").strip() == s.get("transcription", "").strip():
                    is_dup = True
                    break
            elif evt.get("type") == "sound_event":
                d1 = evt.get("description", "").lower()[:60]
                d2 = s.get("description", "").lower()[:60]
                if d1 == d2 or (len(d1) > 20 and d1[:20] == d2[:20]):
                    is_dup = True
                    break
            else:
                is_dup = True
                break
        if not is_dup:
            seen.append(evt)
    return seen


# ════════════════════════════════════════════════════════════
# Context builder + prompt
# ════════════════════════════════════════════════════════════

SCHEMA_BLOCK = r"""## Output Schema

Return a JSON array of segment dictionaries. Each segment uses one of three schemas:

### Speech segment
```json
{
  "type": "speech",
  "start_time": 2.31,
  "end_time": 5.87,
  "transcription": "I can't believe you actually did that",
  "speaker_id": "speaker_1",
  "emotion": "anger_3",
  "age": "adult_30s",
  "gender": "female",
  "voice_timbre": "alto, warm, slightly raspy",
  "speaking_style": "confrontational, accusatory, raised voice",
  "language": "en",
  "accent": "American Midwest",
  "speaking_rate": "fast"
}
```

### Vocal burst segment
```json
{
  "type": "vocal_burst",
  "start_time": 5.87,
  "end_time": 6.14,
  "transcription": null,
  "speaker_id": "speaker_1",
  "vocal_burst": "scoff",
  "emotion": "contempt_2"
}
```

### Sound event segment
```json
{
  "type": "sound_event",
  "start_time": 3.10,
  "end_time": 5.20,
  "description": "Medium-sized dog barking aggressively in the background",
  "loudness": "moderate"
}
```

### Field details
- start_time/end_time: Seconds from audio start, 2 decimal places.
- speaker_id: Consistent label per unique voice (speaker_1, speaker_2, ...).
- emotion: EmoNet taxonomy label with intensity suffix 1-4 (anger_3, joy_2, sadness_4, etc.).
- age: baby, toddler, child, teenager, young_adult_20s, adult_30s, adult_40s, middle_aged_50s, senior_60s, elderly_70s_plus.
- gender: male, female, nonbinary, unclear.
- speaking_rate: very_slow, slow, normal, fast, very_fast.
- voice_timbre: comma-separated descriptors.
- speaking_style: Free description of delivery manner.
- language: ISO 639-1 code.
- vocal_burst: Category label (chuckle, belly_laugh, gentle_sob, gasp, sigh, scoff, scream, etc.).
- loudness (sound events only): quiet, moderate, loud, very_loud.
- Segments may overlap in time."""

SPEAKER_RULE = """## SPEAKER IDENTITY RULE (MANDATORY)

The ASR pipelines are the DEFINITIVE ground truth for:
- How many speakers exist in this audio
- Which speaker says what and when
- Speaker IDs (speaker_0, speaker_1, etc.)

Do NOT create additional speakers beyond what the ASR systems identify."""

PROMPT_TEMPLATE = r"""You are an expert audio annotation model. Annotate every audible event.

{schema}

{speaker_rule}

## SOUND EVENT ANNOTATION

The upstream LoRA model provides MEDIUM-LENGTH overlapping sound event predictions below. These are broader windows that may contain multiple distinct sounds within each prediction.

Your job:
1. For each upstream prediction, listen to that time range
2. Break broad descriptions into separate, specific sound_event entries where appropriate
3. Refine timestamps to match what you actually hear
4. Add sounds the upstream missed (transitions, brief impacts, room tone shifts)
5. Continuous backgrounds (drones, music) = single entries spanning full duration

## TRIPLE-ASR RECONCILIATION

You have THREE independent ASR transcriptions below. Two use Sortformer diarization (Parakeet, Qwen3-ASR), one uses its own built-in pipeline (VibeVoice-ASR).

Reconciliation strategy:
- **Majority vote**: If 2 or 3 ASR systems agree on a word → HIGH CONFIDENCE, use it
- **Single disagreement**: If one system differs from two others → trust the majority
- **All three differ**: Listen carefully to the audio and pick the most plausible version
- **Extra words**: If any ASR has words the others missed → likely real speech, INCLUDE it
- **Timestamps**: Average or prefer the source most consistent with what you hear
- **Speaker IDs**: Trust Sortformer diarization (used by Parakeet and Qwen3); cross-check with VibeVoice

## SPEECH ANNOTATION

Use the reconciled transcripts. Fill speaker attributes from voice analysis + your listening.

## COMPLETENESS

Every second of audio should be covered by at least one annotation (speech, sound event, or both). Gaps between speech always contain at least ambient sound or room tone.

## Background Information

{context}

Be thorough. Output ONLY the JSON array."""


def _format_utterances(utterances, label):
    """Format a list of utterances as markdown lines."""
    lines = [f"### {label}"]
    for seg in utterances:
        spk = f"Speaker {seg.get('speaker_id', '?')}"
        st = seg.get("start_time", "?")
        et = seg.get("end_time", "?")
        content = seg.get("content", "")
        lines.append(f'- {spk}: [{st}s - {et}s] "{content}"')
    lines.append("")
    return lines


def build_triple_context(source, sid):
    """Build context block from VibeVoice + Parakeet + Qwen3 + Whisper + LoRA SFX."""
    sk = skey(source, sid)
    lines = []

    # ── ASR Source 1: VibeVoice-ASR ──
    if source == "synth":
        vv_path = SYNTH_SCENES_DIR / f"scene_{sid:02d}_asr.json"
    else:
        vv_path = YT_SCENES_DIR / f"scene_{sid:02d}_asr.json"

    if vv_path.exists():
        with open(vv_path) as f:
            vv_asr = json.load(f)
        if isinstance(vv_asr, dict):
            vv_asr = vv_asr.get("primary_utterances", vv_asr.get("utterances", []))
        if vv_asr:
            lines.extend(_format_utterances(vv_asr, "ASR Source 1: VibeVoice-ASR (end-to-end, built-in diarization)"))

    # ── ASR Source 2: Parakeet + Sortformer ──
    parakeet_path = ENS_ASR_DIR / f"{sk}_asr_parakeet.json"
    if parakeet_path.exists():
        with open(parakeet_path) as f:
            p_utts = json.load(f)
        if p_utts:
            lines.extend(_format_utterances(p_utts, "ASR Source 2: Parakeet TDT v3 + Sortformer diarization"))

    # ── ASR Source 3: Qwen3-ASR + Sortformer ──
    qwen3_path = ENS_ASR_DIR / f"{sk}_asr_qwen3.json"
    if qwen3_path.exists():
        with open(qwen3_path) as f:
            q_utts = json.load(f)
        if q_utts:
            lines.extend(_format_utterances(q_utts, "ASR Source 3: Qwen3-ASR-1.7B + Sortformer diarization"))

    # ── Whisper experts (from ensemble pipeline, Parakeet-based segments) ──
    whisper_path = ENS_ASR_DIR / f"{sk}_whisper.json"
    if whisper_path.exists():
        with open(whisper_path) as f:
            whisper_data = json.load(f)
        if whisper_data:
            lines.append("### Per-Segment Voice Analysis (emotion / timbre / style)")
            for seg in whisper_data:
                st = seg.get("start_time", "?")
                et = seg.get("end_time", "?")
                lines.append(f"**Speaker {seg.get('speaker_id', '?')} [{st}s - {et}s]**:")
                for k in ["emotion", "timbre", "style"]:
                    v = seg.get(k, "N/A")
                    if v:
                        v = str(v)[:400]
                        lines.append(f"- {k.title()}: {v}")
            lines.append("")

    # ── Also check VibeVoice whisper (different segmentation) for synth scenes ──
    if source == "synth":
        whisper_files = sorted(SYNTH_SEGMENTS_DIR.glob(f"scene_{sid:02d}_seg_*_whisper.json"))
        if whisper_files and not whisper_path.exists():
            # Fallback: use VibeVoice-pipeline whisper if ensemble whisper missing
            lines.append("### Per-Segment Voice Analysis (emotion / timbre / style)")
            for wp in whisper_files:
                with open(wp) as f:
                    wd = json.load(f)
                spk = wd.get("speaker_id", "?")
                st = wd.get("start_time", "?")
                et = wd.get("end_time", "?")
                lines.append(f"**Speaker {spk} [{st}s - {et}s]**:")
                for k in ["emotion", "timbre", "style"]:
                    v = wd.get(k, wd.get("tags", ""))
                    if v:
                        v = str(v)[:400]
                        lines.append(f"- {k.title()}: {v}")
            lines.append("")

    # ── LoRA SFX predictions (reused) ──
    lora_path = get_lora_path(source, sid)
    if lora_path and lora_path.exists():
        with open(lora_path) as f:
            ld = json.load(f)
        parsed = ld.get("parsed", [])
        if parsed:
            lines.append("### Sound Event Predictions (fine-tuned LoRA model, medium overlap)")
            lines.append("**IMPORTANT: Verify each prediction against what you hear. Refine timestamps. Add sounds the model missed.**")
            lines.append("")
            for i, e in enumerate(parsed):
                lines.append(f"**Event {i+1}** [{e.get('start_time','?')}s - {e.get('end_time','?')}s]: {e.get('caption','')}")
                lines.append("")
    else:
        lines.append("### Sound Event Predictions: Not available for this scene")
        lines.append("")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# MOSS inference
# ════════════════════════════════════════════════════════════

def run_thinking_worker(gpu_id, config_name, scene_items, do_sample, temperature):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    try:
        import torch
        sys.path.insert(0, "/home/deployer/laion/MOSS-Audio")
        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor
        from src.audio_io import load_audio

        processor = MossAudioProcessor.from_pretrained(MODEL_THINKING, trust_remote_code=True)
        model = MossAudioModel.from_pretrained(
            MODEL_THINKING, trust_remote_code=True,
            dtype=torch.bfloat16, device_map=device,
        )
        model.eval()
        mel_sr = processor.config.mel_sr
        audio_token_id = processor.audio_token_id

        print(f"[GPU {gpu_id}] Thinking model ready for {config_name}", flush=True)

        out_dir = OUTPUT_DIR / config_name
        out_dir.mkdir(exist_ok=True)

        for source, sid, wav_path in scene_items:
            sk = skey(source, sid)
            pred_path = out_dir / f"{sk}_pred.json"
            if pred_path.exists():
                print(f"  [GPU {gpu_id}] {config_name} {sk}: cached", flush=True)
                continue

            t0 = time.time()
            try:
                context = build_triple_context(source, sid)
                if not context:
                    print(f"  [GPU {gpu_id}] {config_name} {sk}: no context, skipping", flush=True)
                    continue

                instruction = (PROMPT_TEMPLATE
                    .replace("{context}", context)
                    .replace("{schema}", SCHEMA_BLOCK)
                    .replace("{speaker_rule}", SPEAKER_RULE))

                raw_audio = load_audio(str(wav_path), sample_rate=mel_sr)
                inputs = processor(text=instruction, audios=[raw_audio], return_tensors="pt").to(device)
                if inputs.get("audio_data") is not None:
                    inputs["audio_data"] = inputs["audio_data"].to(torch.bfloat16)
                inputs["audio_input_mask"] = inputs["input_ids"] == audio_token_id

                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=do_sample,
                        temperature=temperature,
                        use_cache=True,
                    )

                raw_text = processor.decode(
                    gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                clean = strip_thinking(raw_text)
                parsed = extract_json(clean)
                if parsed:
                    parsed = dedup_events(parsed)

                elapsed = time.time() - t0
                n_events = len(parsed) if parsed else 0
                print(f"  [GPU {gpu_id}] {config_name} {sk}: {n_events} events, {elapsed:.1f}s", flush=True)

                if parsed:
                    with open(pred_path, "w") as f:
                        json.dump(parsed, f, indent=2)
                with open(out_dir / f"{sk}_raw.txt", "w") as f:
                    f.write(raw_text)

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [GPU {gpu_id}] {config_name} {sk}: ERROR {e} ({elapsed:.1f}s)", flush=True)
                traceback.print_exc()
                with open(out_dir / f"{sk}_error.txt", "w") as f:
                    f.write(f"{e}\n\n{traceback.format_exc()}")

    except Exception as e:
        print(f"[GPU {gpu_id}] Thinking INIT FAILED for {config_name}: {e}", flush=True)
        traceback.print_exc()


# ════════════════════════════════════════════════════════════
# Gemini judging
# ════════════════════════════════════════════════════════════

API_KEY = os.environ.get("GEMINI_API_KEY", "")
API_URL = os.environ.get(
    "GEMINI_API_URL",
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={API_KEY}",
)

DIM_LABELS = {
    "emotion_accuracy": "Emotion",
    "age_gender_accuracy": "Age/Gender",
    "transcription_accuracy": "Transcription",
    "timestamp_accuracy": "Timestamps",
    "vocal_burst_accuracy": "Vocal Burst",
    "speaker_diarization": "Diarization",
    "sound_event_accuracy": "Sound Events",
    "segment_completeness": "Completeness",
    "overall_quality": "Overall",
}
DIM_ORDER = list(DIM_LABELS.keys())


def judge_with_gemini(config_name, source, scene_id, prediction, audio_b64):
    import requests

    pred_str = json.dumps(prediction, indent=2) if prediction else "[]"
    gt_path = get_gt_path(source, scene_id)

    if gt_path and gt_path.exists():
        with open(gt_path) as f:
            gt = json.load(f)
        gt_str = json.dumps(gt, indent=2)
        prompt = f"""You are an expert audio annotation evaluator. Listen to the audio and evaluate the model's predicted annotation.

Rate each dimension 0-5 (5=perfect, 0=random). Speech may be in English or German.

Dimensions:
- emotion_accuracy: Are emotion labels correct?
- age_gender_accuracy: Are age/gender correct?
- transcription_accuracy: Is transcription correct?
- timestamp_accuracy: Are start/end times correct?
- vocal_burst_accuracy: Are vocal bursts correctly identified?
- speaker_diarization: Are speakers correctly distinguished?
- sound_event_accuracy: Are sound effects, music, ambient sounds correctly identified and described? Timestamps reasonable?
- segment_completeness: Are ALL audible events captured? Sound events between/during speech?
- overall_quality: Holistic assessment

Ground Truth:
```json
{gt_str}
```

Prediction:
```json
{pred_str}
```

Output ONLY a JSON object: {{"emotion_accuracy":N,"age_gender_accuracy":N,"transcription_accuracy":N,"timestamp_accuracy":N,"vocal_burst_accuracy":N,"speaker_diarization":N,"sound_event_accuracy":N,"segment_completeness":N,"overall_quality":N,"comments":"brief explanation"}}"""
    else:
        prompt = f"""You are an expert audio annotation evaluator. Listen to the audio and evaluate how well the model's predicted annotation captures what's in the audio.

Rate each dimension 0-5 (5=perfect, 0=random). You must judge by listening to the audio — there is no ground truth.

Dimensions:
- emotion_accuracy: Do the emotion labels match what you hear?
- age_gender_accuracy: Are age/gender estimates plausible?
- transcription_accuracy: Does the transcription match the spoken words?
- timestamp_accuracy: Are start/end times aligned with what you hear?
- vocal_burst_accuracy: Are vocal bursts (laughs, gasps, sighs, etc.) correctly identified?
- speaker_diarization: Are speakers correctly distinguished?
- sound_event_accuracy: Are sound effects, music, ambient sounds correctly identified? Timestamps reasonable?
- segment_completeness: Are ALL audible events captured? Missing speech, sounds, or bursts?
- overall_quality: Holistic quality assessment

Prediction:
```json
{pred_str}
```

Output ONLY a JSON object: {{"emotion_accuracy":N,"age_gender_accuracy":N,"transcription_accuracy":N,"timestamp_accuracy":N,"vocal_burst_accuracy":N,"speaker_diarization":N,"sound_event_accuracy":N,"segment_completeness":N,"overall_quality":N,"comments":"brief explanation"}}"""

    payload = {
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 16384,
            "thinkingConfig": {"thinkingBudget": 2048},
        },
    }

    for attempt in range(5):
        try:
            resp = requests.post(API_URL, json=payload,
                                 headers={"Content-Type": "application/json"}, timeout=180)
            if resp.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            if resp.status_code != 200:
                time.sleep(5 * (attempt + 1))
                continue
            data = resp.json()
            if not data.get("candidates"):
                time.sleep(5)
                continue
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            text = "\n".join(p["text"] for p in parts if "text" in p).strip()
            if not text:
                time.sleep(5)
                continue
            fm = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if fm:
                text = fm.group(1)
            jm = re.search(r'\{[\s\S]*\}', text)
            if jm:
                try:
                    scores = json.loads(jm.group())
                    scores["config"] = config_name
                    scores["source"] = source
                    scores["scene_id"] = scene_id
                    return scores
                except Exception:
                    pass
            try:
                scores = json.loads(text)
                scores["config"] = config_name
                scores["source"] = source
                scores["scene_id"] = scene_id
                return scores
            except Exception:
                time.sleep(3)
                continue
        except Exception as e:
            if attempt < 4:
                time.sleep(5 * (attempt + 1))
            else:
                return {"error": str(e), "overall_quality": -1}
    return {"error": "max retries", "overall_quality": -1}


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    all_scenes = get_all_scenes()
    n_synth = sum(1 for s in all_scenes if s[0] == "synth")
    n_yt = sum(1 for s in all_scenes if s[0] == "yt")
    print(f"Found {len(all_scenes)} scenes ({n_synth} synth + {n_yt} YT)")

    # Verify all ASR data exists
    for source, sid, _ in all_scenes:
        sk = skey(source, sid)
        missing = []
        if source == "synth":
            if not (SYNTH_SCENES_DIR / f"scene_{sid:02d}_asr.json").exists():
                missing.append("vv_asr")
        else:
            if not (YT_SCENES_DIR / f"scene_{sid:02d}_asr.json").exists():
                missing.append("vv_asr")
        if not (ENS_ASR_DIR / f"{sk}_asr_parakeet.json").exists():
            missing.append("parakeet")
        if not (ENS_ASR_DIR / f"{sk}_asr_qwen3.json").exists():
            missing.append("qwen3")
        if missing:
            print(f"  WARNING: {sk} missing: {', '.join(missing)}")

    # ════════════════════════════════════════
    # Phase 1: MOSS thinking model (greedy + temp=0.5)
    # ════════════════════════════════════════
    configs = [
        ("greedy", False, 1.0),
        ("temp05", True, 0.5),
    ]

    print(f"\n{'='*60}")
    print(f"Phase 1: MOSS-8B-Thinking with triple ASR context (greedy + temp=0.5)")
    print(f"{'='*60}")

    # 4 GPUs per config: greedy on GPUs 0-3, temp05 on GPUs 4-7
    procs = []
    for ci, (config_name, do_sample, temperature) in enumerate(configs):
        gpu_offset = ci * 4
        n_gpus = 4
        scenes_per_gpu = [[] for _ in range(n_gpus)]
        for j, scene_item in enumerate(all_scenes):
            scenes_per_gpu[j % n_gpus].append(scene_item)

        for g in range(n_gpus):
            if scenes_per_gpu[g]:
                p = Process(target=run_thinking_worker,
                           args=(gpu_offset + g, config_name, scenes_per_gpu[g],
                                 do_sample, temperature))
                procs.append(p)
                p.start()

    for p in procs:
        p.join()

    for config_name, _, _ in configs:
        out_dir = OUTPUT_DIR / config_name
        n_pred = sum(1 for s, i, _ in all_scenes
                     if (out_dir / f"{skey(s,i)}_pred.json").exists())
        print(f"  {config_name}: {n_pred}/{len(all_scenes)} predictions")
    print()

    # ════════════════════════════════════════
    # Phase 2: Gemini judging
    # ════════════════════════════════════════
    print(f"{'='*60}")
    print(f"Phase 2: Gemini judging")
    print(f"{'='*60}")

    print("Loading audio files...")
    audio_cache = {}
    for source, sid, wav_path in all_scenes:
        sk = skey(source, sid)
        audio_cache[sk] = base64.b64encode(wav_path.read_bytes()).decode()
    print(f"Loaded {len(audio_cache)} audio files")

    all_scores = {}
    tasks = []

    for config_name, _, _ in configs:
        out_dir = OUTPUT_DIR / config_name
        for source, sid, wav_path in all_scenes:
            sk = skey(source, sid)
            score_path = out_dir / f"{sk}_score.json"
            if score_path.exists():
                with open(score_path) as f:
                    all_scores[(config_name, source, sid)] = json.load(f)
                continue
            pred_path = out_dir / f"{sk}_pred.json"
            if pred_path.exists() and sk in audio_cache:
                with open(pred_path) as f:
                    prediction = json.load(f)
                tasks.append((config_name, source, sid, prediction))

    print(f"Judging {len(tasks)} predictions (skipping {len(all_scores)} cached)...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for config_name, source, sid, prediction in tasks:
            sk = skey(source, sid)
            fut = executor.submit(judge_with_gemini, config_name, source, sid,
                                  prediction, audio_cache[sk])
            futures[fut] = (config_name, source, sid)

        for fut in futures:
            config_name, source, sid = futures[fut]
            sk = skey(source, sid)
            try:
                scores = fut.result()
                if scores and scores.get("overall_quality", -1) >= 0:
                    all_scores[(config_name, source, sid)] = scores
                    oq = scores.get("overall_quality", "?")
                    print(f"    {config_name}/{sk}: overall={oq}")
                    out_dir = OUTPUT_DIR / config_name
                    with open(out_dir / f"{sk}_score.json", "w") as f:
                        json.dump(scores, f, indent=2)
                else:
                    print(f"    {config_name}/{sk}: FAILED")
            except Exception as e:
                print(f"    {config_name}/{sk}: ERROR {e}")

    # ════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")

    report = {
        "configs": [],
        "n_scenes": len(all_scenes),
        "n_synth": n_synth,
        "n_yt": n_yt,
    }

    for config_name, _, _ in configs:
        valid_all = [all_scores.get((config_name, s, i)) for s, i, _ in all_scenes]
        valid_all = [sc for sc in valid_all if sc and sc.get("overall_quality", -1) >= 0]
        valid_synth = [all_scores.get((config_name, s, i)) for s, i, _ in all_scenes if s == "synth"]
        valid_synth = [sc for sc in valid_synth if sc and sc.get("overall_quality", -1) >= 0]
        valid_yt = [all_scores.get((config_name, s, i)) for s, i, _ in all_scenes if s == "yt"]
        valid_yt = [sc for sc in valid_yt if sc and sc.get("overall_quality", -1) >= 0]

        def compute_dims(scores_list):
            if not scores_list:
                return {}, 0
            dims = {}
            for d in DIM_ORDER:
                vals = [sc[d] for sc in scores_list if d in sc]
                dims[d] = sum(vals) / len(vals) if vals else 0
            overall = sum(dims.values()) / len(dims) if dims else 0
            return dims, overall

        dims_all, overall_all = compute_dims(valid_all)
        dims_synth, overall_synth = compute_dims(valid_synth)
        dims_yt, overall_yt = compute_dims(valid_yt)

        entry = {
            "config": config_name,
            "combined": {"overall": round(overall_all, 2), "dims": {k: round(v, 2) for k, v in dims_all.items()}, "n_scored": len(valid_all)},
            "synthetic": {"overall": round(overall_synth, 2), "dims": {k: round(v, 2) for k, v in dims_synth.items()}, "n_scored": len(valid_synth)},
            "yt": {"overall": round(overall_yt, 2), "dims": {k: round(v, 2) for k, v in dims_yt.items()}, "n_scored": len(valid_yt)},
            "per_scene": {skey(s, i): all_scores.get((config_name, s, i)) for s, i, _ in all_scenes},
        }
        report["configs"].append(entry)

        print(f"\n{config_name}:")
        print(f"  Combined: overall={overall_all:.2f} ({len(valid_all)} scored)")
        for d, v in sorted(dims_all.items(), key=lambda x: -x[1]):
            print(f"    {d}: {v:.2f}")
        print(f"  Synthetic: overall={overall_synth:.2f} ({len(valid_synth)} scored)")
        print(f"  YT:        overall={overall_yt:.2f} ({len(valid_yt)} scored)")

    with open(OUTPUT_DIR / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Report saved: {OUTPUT_DIR / 'eval_report.json'}")


if __name__ == "__main__":
    main()
