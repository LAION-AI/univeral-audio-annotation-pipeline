# Evaluation Results

## Benchmark Setup

- **60 test scenes**: 50 synthetic TTS + 10 YouTube (real-world)
- **Synthetic scenes** (19-56 seconds each):
  - Scenes 0-4: Solo speaker (anger, fear, joy, sadness, tenderness)
  - Scenes 5-9: Solo speaker + 1 vocal burst
  - Scenes 10-17: Dialog (2 speakers, various emotions)
  - Scenes 18-24: Dialog + vocal burst
  - Scenes 25-29: Three speakers
  - Scenes 30-34: Three speakers + vocal burst
  - Scenes 35-39: Solo + 2 vocal bursts
  - Scenes 40-44: Overlapping speech + vocal burst
  - Scenes 45-49: Dense scenes (3 speakers, 4 speech segments, 2 bursts)
- **YouTube scenes**: 5 Frankenstein trailer + 5 Nuclear Shelter clips (~30s each)
- **Evaluator**: Gemini 3.1 Pro, 9 dimensions rated 0-5

## Scoring Dimensions

| Dimension | Description |
|-----------|-------------|
| `emotion_accuracy` | Are emotion labels correct? |
| `age_gender_accuracy` | Are age/gender correct? |
| `transcription_accuracy` | Is transcription correct? |
| `timestamp_accuracy` | Are start/end times correct? |
| `vocal_burst_accuracy` | Are vocal bursts correctly identified? |
| `speaker_diarization` | Are speakers correctly distinguished? |
| `sound_event_accuracy` | Are sound effects, music, ambient sounds correctly identified? |
| `segment_completeness` | Are ALL audible events captured? |
| `overall_quality` | Holistic assessment |

## Full 8-Configuration Ranking

Equal-weight combined: (Synthetic + YouTube) / 2

| # | ASR Config | Decoding | Synthetic | YouTube | Combined |
|---|------------|----------|-----------|---------|----------|
| 1 | **Triple** | **greedy** | 3.70 | **4.56** | **4.13** |
| 2 | Triple | temp=0.5 | **3.74** | 4.23 | 3.99 |
| 3 | Ensemble | greedy | 3.43 | 4.39 | 3.91 |
| 4 | VibeVoice | greedy | 3.53 | 4.08 | 3.80 |
| 5 | VibeVoice | temp=0.5 | 3.70 | 3.65 | 3.67 |
| 6 | Ensemble | temp=0.5 | 3.50 | 3.81 | 3.66 |

**Triple ASR** = VibeVoice + Parakeet TDT v3 + Qwen3-ASR (all three fed to MOSS)
**Ensemble ASR** = Parakeet TDT v3 + Qwen3-ASR (two systems, no VibeVoice)
**VibeVoice** = VibeVoice-ASR only

## Per-Dimension Results: Triple ASR Greedy (Best Config)

### Synthetic Scenes (50)

| Dimension | Score |
|-----------|-------|
| Transcription | 4.65 |
| Age/Gender | 4.20 |
| Diarization | 4.02 |
| Completeness | 3.90 |
| Timestamps | 3.75 |
| SFX | 3.56 |
| Quality | 3.46 |
| Emotion | 3.31 |
| Vocal Bursts | 2.42 |
| **Overall** | **3.70** |

### YouTube Scenes (10)

| Dimension | Score |
|-----------|-------|
| Diarization | 5.00 |
| Age/Gender | 4.90 |
| Transcription | 4.50 |
| Completeness | 4.50 |
| Timestamps | 4.50 |
| Vocal Bursts | 4.50 |
| Quality | 4.50 |
| SFX | 4.40 |
| Emotion | 4.20 |
| **Overall** | **4.56** |

## Key Findings

### 1. Triple ASR is the clear winner

Three ASR systems compensate for each other's weaknesses:
- **VibeVoice** excels at clean TTS segmentation and vocal burst detection
- **Parakeet TDT v3** provides precise word-level timestamps
- **Qwen3-ASR** catches missed words

The majority-vote reconciliation strategy lets MOSS pick the best from each.

### 2. Greedy decoding wins with rich context

With triple ASR context, the model receives enough information that sampling diversity is unnecessary. Combined: greedy 4.13 > temp=0.5 3.99.

### 3. More ASR sources = better real-world performance

| Config | YouTube Score | Improvement |
|--------|-------------|-------------|
| VibeVoice only | 4.08 | baseline |
| Ensemble (2 ASR) | 4.39 | +0.31 |
| Triple (3 ASR) | 4.56 | +0.48 |

### 4. Ensemble-only (no VibeVoice) hurts synthetic performance

Removing VibeVoice drops synthetic scores below single-ASR baseline. VibeVoice's built-in segmentation is important for clean TTS audio and vocal burst detection.

### 5. Vocal burst accuracy is the weakest dimension

Scores of 2.42-2.62 on synthetic scenes across all configs. This remains the most challenging aspect of the pipeline.

### 6. Domain-specific recommendations

- **Synthetic/controlled audio**: Triple temp=0.5 is slightly better (3.74 vs 3.70)
- **Real-world audio**: Triple greedy is clearly better (4.56 vs 4.23)
- **Equal-weighted**: Triple greedy is the best default

## Prompt Optimization History

Tested 11 system prompt variants on a 9-scene pilot set:

| Rank | Variant | SFX Config | Score |
|------|---------|------------|-------|
| 1 | p4_medium_sfx | medium_overlap | **4.28** |
| 2 | p8_two_pass | medium_overlap | 4.05 |
| 3 | p10_structured | short_overlap | 3.90 |

**Finding**: Medium-overlap SFX predictions consistently outperform short-overlap. The winning prompt instructs the model to "break broad descriptions into separate, specific sound_event entries."

## Beam Search Comparison

| Config | Score |
|--------|-------|
| Thinking + p4 (greedy) | **4.28** |
| Thinking + p10 (beam=3) | 4.26 |
| Thinking + p4 (beam=3) | 4.01 |

**Finding**: Beam search does not help the thinking model with the p4 prompt. It makes the model more conservative (2.7 vs 4.4 events/scene) and severely hurts LoRA models (heavy hallucination).

## LoRA Training History

| Version | Training Data | Eval Loss | Notes |
|---------|--------------|-----------|-------|
| v4 | 10,998 Gemini 2.5 Pro annotations | 2.76 | Best standalone |
| v5 epoch1 | 50/50 Gemini + LAION soundscapes | 5.53 | Mixed data hurts |
| v5 epoch2 | Gemini-only, resumed from v5e1 | 5.55 | Did not recover |
