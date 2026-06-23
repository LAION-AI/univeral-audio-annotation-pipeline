"""DramaBox prompt generation from audio analysis data.

Fuses upstream expert outputs (ASR, Whisper emotion/timbre/style, speaker
embeddings) into DramaBox-format performance prompts using Gemma 4 E4B-it.

Unlike the JSON annotation fusion (gemma_fusion.py), this module produces
**DramaBox prompts** — single-speaker performance scripts with stage directions
in parentheses and dialogue in double quotes — suitable for TTS synthesis.

One DramaBox prompt is generated per speaker-consistent group of utterances.
"""
import json
from typing import List, Dict, Optional

from .speaker_embeddings import SIMILARITY_THRESHOLD
from .utils import extract_json, strip_thinking


# ---------------------------------------------------------------------------
# Few-shot DramaBox examples
# ---------------------------------------------------------------------------
DRAMABOX_EXAMPLES = [
    # 1. Single standalone scene (simple emotion)
    {
        "label": "Simple emotion — weariness",
        "prompt": (
            'A 35-year-old man with a deep, gravelly voice and weary undertone, '
            'delivering this high-quality studio voice recording with no background noise.\n\n'
            '(exhaling slowly, voice thick with exhaustion) "I\'ve been walking for three days now." '
            '(a pause, swallowing hard) "My feet are raw. Every step feels like glass." '
            '(voice dropping to a murmur) "But I can see the lights... I think I can see the lights."'
        ),
    },
    # 2. Dramatic/intense scene (extreme physical)
    {
        "label": "Intense physical — pain and determination",
        "prompt": (
            'A woman in her late 20s with a clear, alto voice strained by physical exertion, '
            'delivering this high-quality studio voice recording with no background noise.\n\n'
            '(breathing heavily through gritted teeth) "Don\'t you dare let go of that rope." '
            '(a sharp gasp of pain) "My shoulder... it popped. I heard it pop." '
            '(forcing words through shallow breaths) "Pull me up. Pull me up NOW." '
            '(voice breaking) "I can\'t hold on much longer."'
        ),
    },
    # 3. Two-scene CUT TO: format (character consistency)
    {
        "label": "CUT TO: — emotional contrast",
        "prompt": (
            'A 50-year-old man with a warm baritone and subtle rasp, '
            'delivering this high-quality studio voice recording with no background noise.\n\n'
            '(leaning forward, voice soft with paternal warmth) "You know, when you were little, '
            'you used to hide behind my legs at parties." (a gentle chuckle) "Now look at you. '
            'Giving speeches to hundreds of people."\n\n'
            'CUT TO:\n\n'
            '(voice tight, barely controlled) "I told you not to go back there." '
            '(slamming a hand on the table) "Every single time, you walk into that place and '
            'come out worse than before." (voice cracking) "I can\'t keep watching this."'
        ),
    },
    # 4. Reference-audio style (with timbre description)
    {
        "label": "Timbre-guided — sardonic humor",
        "prompt": (
            'A 40-year-old woman with a smoky contralto, slightly nasal resonance, and a '
            'sardonic edge, delivering this high-quality studio voice recording with no background noise.\n\n'
            '(leaning back, voice dripping with dry amusement) "Oh, you thought that was going '
            'to work?" (a short, breathy laugh) "Darling, I\'ve seen better schemes from a '
            'twelve-year-old." (pausing, tone shifting to something almost kind) '
            '"But I admire the audacity. I really do."'
        ),
    },
]

# ---------------------------------------------------------------------------
# System prompt for Gemma
# ---------------------------------------------------------------------------
DRAMABOX_SYSTEM_PROMPT = """You generate DramaBox TTS prompts from audio analysis data.
A DramaBox prompt describes ONE speaker performing ONE scene.

FORMAT RULES:
1. Start with a speaker description paragraph (age, gender, voice timbre, recording quality).
   Always end with: "delivering this high-quality studio voice recording with no background noise."
2. Then alternate stage directions in parentheses and dialogue in double quotes.
3. Stage directions describe physical actions, emotions, breathing, pauses — NOT sound effects.
4. Dialogue goes inside "double quotes" exactly as transcribed.
5. Keep it to ONE speaker per prompt. If there are multiple speakers, generate separate prompts.
6. The prompt should read like a performance script for a voice actor.

EXAMPLES:

""" + "\n\n---\n\n".join(
    f"**{ex['label']}:**\n{ex['prompt']}" for ex in DRAMABOX_EXAMPLES
) + "\n\n---\n\n"

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------
DRAMABOX_USER_TEMPLATE = """Given the following analysis of an audio utterance:
- Transcription: {asr_text}
- Emotion (BUD-E Whisper): {emotion}
- Voice Timbre (timbre-whisper): {timbre}
- Speaking Style (voice-tagging-whisper): {style}
- Timestamps: {start_time}s - {end_time}s
- Speaker ID: {speaker_id}

Generate a DramaBox prompt for this utterance. Assume exactly ONE speaker.
Use the transcription as the dialogue (in double quotes).
Derive stage directions from the emotion, timbre, and style analysis.

MAJORITY VOTING for conflicting signals:
- If 2 of 3 experts agree on a characteristic (e.g. 2 say "calm"), use the majority.
- If all disagree: prefer BUD-E Whisper for emotion, timbre-whisper for voice quality,
  voice-tagging-whisper for style/delivery.

Output ONLY the DramaBox prompt text (no JSON, no markdown fences)."""

# ---------------------------------------------------------------------------
# Multi-utterance grouped template
# ---------------------------------------------------------------------------
DRAMABOX_GROUP_TEMPLATE = """Given the following analysis of {n_utterances} utterances from the SAME speaker:

{utterance_details}

Generate a SINGLE DramaBox prompt that covers all these utterances as one continuous performance.
Combine all dialogue segments in order, with appropriate stage directions between them
based on the emotion/timbre/style analysis of each segment.

MAJORITY VOTING for conflicting signals:
- If 2 of 3 experts agree on a characteristic, use the majority.
- If all disagree: prefer BUD-E Whisper for emotion, timbre-whisper for voice quality,
  voice-tagging-whisper for style/delivery.

Output ONLY the DramaBox prompt text (no JSON, no markdown fences)."""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------
def build_dramabox_context(
    utterance: Dict,
    whisper_analysis: Optional[Dict] = None,
) -> str:
    """Build the user prompt for a single utterance."""
    asr_text = utterance.get("content", utterance.get("text", ""))
    start_time = utterance.get("start_time", "?")
    end_time = utterance.get("end_time", "?")
    speaker_id = utterance.get("speaker_id", 0)

    # Get whisper analysis if available
    emotion = "unknown"
    timbre = "unknown"
    style = "unknown"
    if whisper_analysis:
        emotion = whisper_analysis.get("emotion", "unknown")
        timbre = whisper_analysis.get("timbre", "unknown")
        style = whisper_analysis.get("style", "unknown")

    return DRAMABOX_USER_TEMPLATE.format(
        asr_text=asr_text,
        emotion=emotion,
        timbre=timbre,
        style=style,
        start_time=start_time,
        end_time=end_time,
        speaker_id=speaker_id,
    )


def build_grouped_context(
    utterances: List[Dict],
    whisper_analyses: List[Dict],
) -> str:
    """Build the user prompt for a group of utterances from the same speaker."""
    details = []
    for i, (utt, wa) in enumerate(zip(utterances, whisper_analyses)):
        asr_text = utt.get("content", utt.get("text", ""))
        start_time = utt.get("start_time", "?")
        end_time = utt.get("end_time", "?")

        emotion = wa.get("emotion", "unknown") if wa else "unknown"
        timbre_val = wa.get("timbre", "unknown") if wa else "unknown"
        style_val = wa.get("style", "unknown") if wa else "unknown"

        details.append(
            f"Utterance {i+1} [{start_time}s-{end_time}s]:\n"
            f"  Transcription: {asr_text}\n"
            f"  Emotion: {emotion}\n"
            f"  Timbre: {timbre_val}\n"
            f"  Style: {style_val}"
        )

    return DRAMABOX_GROUP_TEMPLATE.format(
        n_utterances=len(utterances),
        utterance_details="\n\n".join(details),
    )


# ---------------------------------------------------------------------------
# DramaBox Fuser (Gemma 4 E4B-it via transformers)
# ---------------------------------------------------------------------------
class DramaBoxFuser:
    """Generate DramaBox prompts using Gemma 4 E4B-it.

    Unlike GemmaFuser (which uses GGUF via llama.cpp for the larger 12B),
    this uses the smaller E4B-it directly via transformers since it fits
    easily in GPU memory.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-E4B-it",
        device: str = "cuda:0",
        max_new_tokens: int = 1024,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading DramaBox fuser ({model_id}) on {device}...")
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        print(f"DramaBox fuser loaded on {device}")

    def generate_prompt(
        self,
        utterance: Dict,
        whisper_analysis: Optional[Dict] = None,
    ) -> str:
        """Generate a DramaBox prompt for a single utterance."""
        user_msg = build_dramabox_context(utterance, whisper_analysis)
        return self._generate(user_msg)

    def generate_grouped_prompt(
        self,
        utterances: List[Dict],
        whisper_analyses: List[Dict],
    ) -> str:
        """Generate a DramaBox prompt for a group of same-speaker utterances."""
        user_msg = build_grouped_context(utterances, whisper_analyses)
        return self._generate(user_msg)

    def _generate(self, user_message: str) -> str:
        """Run generation with the system prompt + user message."""
        import torch

        messages = [
            {"role": "user", "content": DRAMABOX_SYSTEM_PROMPT + user_message},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs.shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean up: remove any thinking tags
        text = strip_thinking(text)

        # Remove markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        return text

    def fuse_clip(
        self,
        asr_utterances: List[Dict],
        whisper_analyses: List[Dict],
        speaker_embeddings: Optional[Dict] = None,
    ) -> List[Dict]:
        """Generate DramaBox prompts for all utterances in a clip.

        Groups utterances by speaker (using speaker embedding clusters if
        available, else by ASR speaker_id). Generates one prompt per group.

        Args:
            asr_utterances: Primary ASR output (nemotron/vibevoice).
            whisper_analyses: Per-utterance voice analysis from stage 2.
            speaker_embeddings: Output from stage 2b (optional).

        Returns:
            List of dicts: [{speaker_id, prompt, start_time, end_time, utterances}]
        """
        if not asr_utterances:
            return []

        # Build whisper lookup by (start_time, end_time)
        wa_lookup = {}
        for wa in whisper_analyses:
            key = (round(wa.get("start_time", 0), 2), round(wa.get("end_time", 0), 2))
            wa_lookup[key] = wa

        # Group utterances by speaker
        if speaker_embeddings and speaker_embeddings.get("clusters"):
            # Use embedding-based clusters
            clusters = speaker_embeddings["clusters"]
            embs = speaker_embeddings.get("embeddings", [])

            # Map utterance index to cluster
            utt_to_cluster = {}
            for cluster_id, indices in enumerate(clusters):
                for idx in indices:
                    utt_to_cluster[idx] = cluster_id

            groups = {}
            for i, utt in enumerate(asr_utterances):
                cid = utt_to_cluster.get(i, i)  # fallback: own group
                groups.setdefault(cid, []).append((i, utt))
        else:
            # Group by ASR speaker_id
            groups = {}
            for i, utt in enumerate(asr_utterances):
                sid = utt.get("speaker_id", 0)
                groups.setdefault(sid, []).append((i, utt))

        # Generate one prompt per speaker group
        results = []
        for group_id, members in sorted(groups.items()):
            utts = [m[1] for m in members]

            # Find matching whisper analyses
            was = []
            for utt in utts:
                key = (round(utt.get("start_time", 0), 2), round(utt.get("end_time", 0), 2))
                was.append(wa_lookup.get(key, {}))

            if len(utts) == 1:
                prompt = self.generate_prompt(utts[0], was[0] if was else None)
            else:
                prompt = self.generate_grouped_prompt(utts, was)

            results.append({
                "speaker_id": group_id,
                "prompt": prompt,
                "start_time": min(u.get("start_time", 0) for u in utts),
                "end_time": max(u.get("end_time", 0) for u in utts),
                "n_utterances": len(utts),
            })

        return results

    def cleanup(self):
        """Free GPU memory."""
        import torch
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
