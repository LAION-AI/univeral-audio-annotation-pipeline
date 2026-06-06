#!/usr/bin/env python3
"""Universal Audio Annotation Pipeline - Main entry point.

Produces structured JSON annotations from audio files, covering:
- Speech transcription with speaker diarization
- Emotion, timbre, and speaking style analysis
- Vocal burst detection
- Sound event detection (via LoRA fine-tuned model)
- Final structured annotation (via MOSS-Audio-8B-Thinking)

Usage:
    python -m pipeline.run_pipeline --audio input.wav --output output.json

Configurations:
    nemotron_vibevoice - DEFAULT, best on SoundScape-Bench: Nemotron 3.5 words +
                         VibeVoice/Sortformer diarization + detailed sound-event & music captions
    triple_greedy   - Legacy ensemble: VibeVoice + Parakeet + Qwen3 + greedy decoding
    ensemble_greedy - Dual ASR (Parakeet + Qwen3) + greedy decoding
    vibevoice       - Single ASR (VibeVoice only) + greedy decoding
"""

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal Audio Annotation Pipeline"
    )
    parser.add_argument(
        "--audio", required=True, help="Path to input audio file (WAV)"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output JSON file"
    )
    parser.add_argument(
        "--config",
        choices=["nemotron_vibevoice", "triple_greedy", "ensemble_greedy", "vibevoice"],
        default="nemotron_vibevoice",
        help="Pipeline configuration (default: nemotron_vibevoice — best on SoundScape-Bench)",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU IDs to use (default: '0')",
    )
    parser.add_argument(
        "--moss-audio-path",
        default=None,
        help="Path to MOSS-Audio source code directory",
    )
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Local path to LoRA adapter (otherwise downloads from HuggingFace)",
    )
    parser.add_argument(
        "--sfx-segment-duration",
        choices=["short", "medium"],
        default="medium",
        help="SFX detection segment duration (default: medium)",
    )
    parser.add_argument(
        "--no-sfx",
        action="store_true",
        help="Skip SFX LoRA detection step",
    )
    return parser.parse_args()


def run_pipeline(args):
    """Run the full annotation pipeline."""
    import os
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    audio_path = args.audio
    config = args.config
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    primary_gpu = f"cuda:{gpu_ids[0]}"

    print(f"{'='*60}")
    print(f"Universal Audio Annotation Pipeline")
    print(f"{'='*60}")
    print(f"Audio:  {audio_path}")
    print(f"Config: {config}")
    print(f"GPUs:   {args.gpus}")
    print(f"{'='*60}\n")

    t_start = time.time()

    # ── Step 1: ASR Transcription ──────────────────────────────

    vibevoice_utts = []
    parakeet_utts = []
    qwen3_utts = []
    nemotron_utts = []
    diar_segs = []

    if config in ("nemotron_vibevoice", "triple_greedy", "vibevoice"):
        print("[1/5] Running VibeVoice-ASR...")
        from .asr_vibevoice import VibeVoiceASR
        vv = VibeVoiceASR(device=primary_gpu)
        vibevoice_utts = vv.run(audio_path)
        vv.cleanup()

    if config == "nemotron_vibevoice":
        print("[1/5] Running Nemotron 3.5 + Sortformer...")
        from .asr_nemotron import NemotronSortformerASR
        diar_gpu = f"cuda:{gpu_ids[1]}" if len(gpu_ids) > 1 else primary_gpu
        nemo = NemotronSortformerASR(asr_device=primary_gpu, diar_device=diar_gpu)
        diar_segs = nemo._diarize(audio_path)
        words = nemo._transcribe(audio_path)
        nemotron_utts = nemo._merge(words, diar_segs)
        nemo.cleanup()

    if config in ("triple_greedy", "ensemble_greedy"):
        print("[1/5] Running Parakeet TDT v3 + Sortformer...")
        from .asr_parakeet import ParakeetSortformerASR
        parakeet_gpu = primary_gpu
        diar_gpu = f"cuda:{gpu_ids[1]}" if len(gpu_ids) > 1 else primary_gpu
        parakeet = ParakeetSortformerASR(asr_device=parakeet_gpu, diar_device=diar_gpu)
        parakeet_utts = parakeet.run(audio_path)
        # Save diar segments for Qwen3
        diar_segs = parakeet._diarize(audio_path)
        parakeet.cleanup()

        print("[1/5] Running Qwen3-ASR...")
        from .asr_qwen3 import Qwen3ASR
        qwen3 = Qwen3ASR(device=primary_gpu)
        qwen3_utts = qwen3.run(audio_path, diar_segs=diar_segs)
        qwen3.cleanup()

    # Use best available utterances for Whisper segmentation
    primary_utts = nemotron_utts or parakeet_utts or vibevoice_utts or qwen3_utts
    if not primary_utts:
        print("ERROR: No ASR results obtained. Exiting.")
        sys.exit(1)

    # ── Step 2: Whisper Voice Analysis ─────────────────────────

    print("\n[2/5] Running Whisper expert analysis...")
    from .whisper_experts import WhisperExperts
    whisper = WhisperExperts(device=primary_gpu)
    whisper_analysis = whisper.analyze(audio_path, primary_utts)
    whisper.cleanup()

    # ── Step 3: SFX LoRA Detection ─────────────────────────────

    sfx_predictions = []
    if not args.no_sfx:
        print("\n[3/5] Running SFX LoRA detection...")
        from .sfx_lora import SFXDetector
        sfx = SFXDetector(
            device=primary_gpu,
            lora_path=args.lora_path,
            moss_audio_path=args.moss_audio_path,
        )
        sfx_predictions = sfx.run(
            audio_path,
            segment_duration=args.sfx_segment_duration,
            overlapping=True,
        )
        sfx.cleanup()
    else:
        print("\n[3/5] Skipping SFX LoRA detection (--no-sfx)")

    # ── Step 4: MOSS Final Annotation ──────────────────────────

    print("\n[4/5] Running MOSS-Audio-8B-Thinking annotation...")
    from .moss_annotator import MOSSAnnotator
    moss = MOSSAnnotator(
        device=primary_gpu,
        moss_audio_path=args.moss_audio_path,
    )

    if config == "nemotron_vibevoice":
        context = moss.build_nemotron_context(
            vibevoice_utts, nemotron_utts, diar_segs,
            whisper_analysis, sfx_predictions,
        )
        annotations = moss.annotate(
            audio_path, context, prompt_mode="nemotron", do_sample=False
        )
    elif config == "triple_greedy":
        context = moss.build_triple_context(
            vibevoice_utts, parakeet_utts, qwen3_utts,
            whisper_analysis, sfx_predictions,
        )
        annotations = moss.annotate(
            audio_path, context, prompt_mode="triple", do_sample=False
        )
    elif config == "ensemble_greedy":
        context = moss.build_ensemble_context(
            parakeet_utts, qwen3_utts,
            whisper_analysis, sfx_predictions,
        )
        annotations = moss.annotate(
            audio_path, context, prompt_mode="ensemble", do_sample=False
        )
    else:  # vibevoice
        context = moss.build_triple_context(
            vibevoice_utts, [], [],
            whisper_analysis, sfx_predictions,
        )
        annotations = moss.annotate(
            audio_path, context, prompt_mode="triple", do_sample=False
        )

    moss.cleanup()

    # ── Step 5: Save Output ────────────────────────────────────

    print(f"\n[5/5] Saving output to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete: {len(annotations)} annotations")
    print(f"Runtime: {elapsed:.0f}s")
    print(f"Output: {args.output}")
    print(f"{'='*60}")


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
