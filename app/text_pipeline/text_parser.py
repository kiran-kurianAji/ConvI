"""
ConvI — Text Transcript Parser
Module : app/text_pipeline/text_parser.py

Parses raw text transcripts into segment-like turns for the interim pipeline output.
Supports formats:
  - "Agent: ..." / "Customer: ..."
  - "Speaker 1: ..." / "Speaker 2: ..."
  - Plain lines (alternating speakers)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from app.speech_pipeline.schemas import SpeechSegment


@dataclass
class TextTurn:
    """A single parsed turn from text transcript."""
    speaker_id: str
    text: str
    turn_index: int  # for synthetic start/end time


def parse_transcript(text: str) -> List[TextTurn]:
    """
    Parse raw transcript text into turns.

    Supports:
      Agent: ... / Customer: ...  (works with or without newlines between turns)
      Speaker 1: ... / Speaker 2: ...
      Plain lines (alternate SPEAKER_00, SPEAKER_01)
    """
    text = (text or "").strip()
    if not text:
        return []

    turns: List[TextTurn] = []
    turn_index = 0

    # Split by Agent:/Customer:/Speaker N: (works with or without newlines between turns)
    split_pattern = re.compile(
        r"\s*(Agent|Customer|Speaker\s*\d+)\s*:\s*",
        re.I,
    )
    parts = split_pattern.split(text)

    # parts[0] = leading text; parts[1]=label, parts[2]=content, parts[3]=label, ...
    if len(parts) < 2:
        # No labels — split by newlines, alternate speakers
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            turns.append(TextTurn(
                speaker_id="SPEAKER_00" if i % 2 == 0 else "SPEAKER_01",
                text=ln,
                turn_index=i,
            ))
        return turns

    # Leading content before first label
    first = parts[0].strip()
    if first:
        turns.append(TextTurn(speaker_id="SPEAKER_00", text=first, turn_index=0))
        turn_index = 1
    else:
        turn_index = 0

    # Pairs: (label, content)
    i = 1
    while i + 1 < len(parts):
        label = parts[i].strip()
        content = parts[i + 1].strip()
        if label.lower().startswith("agent"):
            speaker_id = "SPEAKER_00"
        elif label.lower().startswith("customer"):
            speaker_id = "SPEAKER_01"
        elif label.lower().startswith("speaker"):
            num_match = re.search(r"\d+", label)
            idx = int(num_match.group()) if num_match else 1
            speaker_id = f"SPEAKER_{idx - 1:02d}" if idx >= 1 else "SPEAKER_00"
        else:
            speaker_id = "SPEAKER_00"
        if content:
            turns.append(TextTurn(
                speaker_id=speaker_id,
                text=content,
                turn_index=turn_index,
            ))
            turn_index += 1
        i += 2

    return turns


def text_turns_to_speech_segments(turns: List[TextTurn], lang: str = "en") -> List[SpeechSegment]:
    """Convert parsed text turns to SpeechSegment format (for consistent API output)."""
    segments: List[SpeechSegment] = []
    for i, t in enumerate(turns):
        # Use synthetic timings (index-based) since we have no audio
        start = float(i)
        end = float(i + 1)
        segments.append(SpeechSegment(
            speaker_id=t.speaker_id,
            start_time=start,
            end_time=end,
            original_text=t.text,
            language=lang,
            emotion=None,  # no audio for emotion
            emotion_confidence=0.0,
            transcription_confidence=1.0,
            audio_language=lang,
        ))
    return segments
