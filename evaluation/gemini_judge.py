#!/usr/bin/env python3
"""Gemini 3.1 Pro evaluation scorer for audio annotations.

Scores predicted annotations against ground truth (if available) or
audio-only evaluation across 9 dimensions, each rated 0-5.

Requires a Gemini API key set via GEMINI_API_KEY environment variable
or passed directly to the judge function.
"""

import json
import os
import re
import time
from typing import Dict, Optional

import requests


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


def judge_with_gemini(
    prediction: list,
    audio_b64: str,
    ground_truth: Optional[list] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    config_name: str = "",
    source: str = "",
    scene_id: int = 0,
    max_retries: int = 5,
) -> Dict:
    """Score a prediction using Gemini 3.1 Pro.

    Args:
        prediction: List of predicted annotation dicts.
        audio_b64: Base64-encoded audio (WAV).
        ground_truth: Optional list of ground truth annotation dicts.
        api_key: Gemini API key (falls back to GEMINI_API_KEY env var).
        api_url: Full Gemini API endpoint URL. If None, constructs from api_key.
        config_name: Configuration name for metadata.
        source: Source type ('synth' or 'yt') for metadata.
        scene_id: Scene ID for metadata.
        max_retries: Maximum retry attempts for API calls.

    Returns:
        Dict with scores for each dimension, plus metadata.
    """
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    if api_url is None:
        api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-pro:generateContent?key={api_key}"
        )

    pred_str = json.dumps(prediction, indent=2) if prediction else "[]"

    if ground_truth:
        gt_str = json.dumps(ground_truth, indent=2)
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

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                api_url, json=payload,
                headers={"Content-Type": "application/json"},
                timeout=180,
            )
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
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return {"error": str(e), "overall_quality": -1}

    return {"error": "max retries", "overall_quality": -1}


def compute_dimension_averages(scores_list: list) -> Dict:
    """Compute average score per dimension from a list of score dicts.

    Args:
        scores_list: List of score dicts (from judge_with_gemini).

    Returns:
        Dict with per-dimension averages and overall combined score.
    """
    if not scores_list:
        return {"dims": {}, "overall": 0, "n_scored": 0}

    dims = {}
    for d in DIM_ORDER:
        vals = [sc[d] for sc in scores_list if d in sc and sc.get(d, -1) >= 0]
        dims[d] = round(sum(vals) / len(vals), 2) if vals else 0

    overall = round(sum(dims.values()) / len(dims), 2) if dims else 0
    return {"dims": dims, "overall": overall, "n_scored": len(scores_list)}
