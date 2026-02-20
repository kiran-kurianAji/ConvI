"""
Text Pipeline — Orchestrator

Entry point for the text path of the ConvI pipeline.

Input format expected (one turn per line):
    Agent: Good morning, this is XYZ Bank. How may I help you?
    Customer: I accidentally transferred ₹25,000 to the wrong account number.
    ...

Processing stages:
    1. Turn parser         — splits raw transcript into (role, text) turns
    2. Language detector   — langdetect per turn + dominant language for the call
    3. NLP processor       — Stanza tokenization, lemmatization, NER per turn
    4. Output assembler    — builds TextPipelineOutput (→ downstream ConversationTurn list)

Technology:
    • Language detection : langdetect  (replaces fasttext-wheel — no C++ compiler needed)
    • NLP / NER          : Stanza      (replaces spaCy — pure Python/PyTorch, Python 3.14 safe)
"""

from __future__ import annotations

import re
from collections import Counter
from loguru import logger

from app.schemas import Role
from app.text_pipeline.schemas import NamedEntity, ProcessedTurn, TextPipelineOutput
from app.text_pipeline.language_detector import detect_language, dominant_language
from app.text_pipeline.nlp_processor import process_text


# ── 1. Turn parser ────────────────────────────────────────────────────────

# Matches lines like  "Agent:", "Customer:", "AGENT:", "agent :" etc.
_TURN_RE = re.compile(r"^\s*([A-Za-z][A-Za-z\s]*?)\s*:\s*(.*)", re.DOTALL)

# Maps common speaker labels → canonical Role enum
_ROLE_MAP: dict[str, Role] = {
    "agent":       Role.agent,
    "representative": Role.agent,
    "rep":         Role.agent,
    "banker":      Role.agent,
    "officer":     Role.agent,
    "executive":   Role.agent,
    "support":     Role.agent,
    "customer":    Role.customer,
    "client":      Role.customer,
    "caller":      Role.customer,
    "user":        Role.customer,
}


def _parse_turns(transcript: str) -> list[tuple[str, str]]:
    """
    Parse a raw transcript string into a list of (speaker_label, text) tuples.

    Handles multi-line turns: text that does not start with a "Label:" prefix
    is treated as a continuation of the previous turn.
    """
    turns: list[tuple[str, str]] = []
    current_label: str | None = None
    current_lines: list[str] = []

    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = _TURN_RE.match(line)
        if m:
            # Save previous turn
            if current_label is not None and current_lines:
                turns.append((current_label, " ".join(current_lines).strip()))
            current_label = m.group(1).strip()
            current_lines = [m.group(2).strip()]
        else:
            # Continuation line
            if current_label is not None:
                current_lines.append(line)
            # Ignore lines before the first labelled turn

    # Flush last turn
    if current_label is not None and current_lines:
        turns.append((current_label, " ".join(current_lines).strip()))

    return turns


def _resolve_role(label: str) -> Role:
    return _ROLE_MAP.get(label.lower().strip(), Role.unknown)


def _make_speaker_id(role: Role, role_counter: Counter) -> str:
    """Generate deterministic speaker IDs: AGENT_0, CUSTOMER_0, UNKNOWN_0 …"""
    key = role.value.upper()
    idx = role_counter[key]
    role_counter[key] += 1
    return f"{key}_{idx}"


# ── 2-4. Pipeline orchestrator ────────────────────────────────────────────

def run_text_pipeline(transcript: str) -> TextPipelineOutput:
    """
    Main entry point.  Accepts a raw multi-speaker text transcript and
    returns a fully populated :class:`TextPipelineOutput`.

    Parameters
    ----------
    transcript : str
        Raw conversation text in  ``Speaker: text``  format.

    Returns
    -------
    TextPipelineOutput
        Structured result ready for the Conversation Normalizer stage.
    """
    logger.info("[TextPipeline] Starting text pipeline …")

    # ── Stage 1: parse turns ──────────────────────────────────────────
    raw_turns = _parse_turns(transcript)
    if not raw_turns:
        raise ValueError(
            "Could not parse any speaker turns from the transcript. "
            "Expected format: 'Speaker: text' (one turn per line)."
        )
    logger.info(f"[TextPipeline] Parsed {len(raw_turns)} turns")

    # ── Stage 2: per-turn language detection ─────────────────────────
    turn_langs: list[tuple[str, float]] = [
        detect_language(text) for _, text in raw_turns
    ]
    call_dominant_lang = dominant_language([text for _, text in raw_turns])
    logger.info(f"[TextPipeline] Dominant language: {call_dominant_lang}")

    # ── Stage 3: NLP processing ───────────────────────────────────────
    processed_turns: list[ProcessedTurn] = []
    all_entities: list[NamedEntity] = []
    role_counter: Counter = Counter()
    seen_speakers: dict[str, str] = {}   # label → speaker_id (consistent IDs)

    for idx, ((label, text), (lang, conf)) in enumerate(
        zip(raw_turns, turn_langs)
    ):
        logger.debug(f"[TextPipeline] Turn {idx}: speaker={label!r}, lang={lang}")

        # Consistent speaker IDs per unique label
        if label not in seen_speakers:
            role = _resolve_role(label)
            speaker_id = _make_speaker_id(role, role_counter)
            seen_speakers[label] = speaker_id
        else:
            speaker_id = seen_speakers[label]
            role = _resolve_role(label)

        nlp_result = process_text(text, lang=lang)

        turn = ProcessedTurn(
            turn_index=idx,
            speaker_label=label,
            role=role,
            original_text=text,
            cleaned_text=nlp_result["cleaned_text"],
            lemmatized_text=nlp_result["lemmatized_text"],
            language=lang,
            language_confidence=conf,
            entities=nlp_result["entities"],
            tokens=nlp_result["tokens"],
        )
        processed_turns.append(turn)
        all_entities.extend(nlp_result["entities"])

    # Deduplicate entities (same text + label)
    seen_ent_keys: set[tuple[str, str]] = set()
    unique_entities: list[NamedEntity] = []
    for ent in all_entities:
        key = (ent.text.lower(), ent.label)
        if key not in seen_ent_keys:
            seen_ent_keys.add(key)
            unique_entities.append(ent)

    unique_speakers = len(seen_speakers)
    logger.info(
        f"[TextPipeline] Done — {len(processed_turns)} turns, "
        f"{unique_speakers} speakers, {len(unique_entities)} unique entities"
    )

    return TextPipelineOutput(
        raw_transcript=transcript,
        dominant_language=call_dominant_lang,
        turns=processed_turns,
        all_entities=unique_entities,
        speaker_count=unique_speakers,
    )


__all__ = ["run_text_pipeline", "TextPipelineOutput"]

