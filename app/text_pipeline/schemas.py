"""
Text Pipeline — Pydantic Schemas

Internal output contracts for the text pipeline stage.
These are separate from the top-level ConversationAnalyticsResponse schemas.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from app.schemas import Role


# ── Single named entity extracted by Stanza ──────────────────────────────

class NamedEntity(BaseModel):
    text: str                    # surface form  e.g. "Rahul Menon"
    label: str                   # NER type      e.g. "PERSON"
    start_char: int              # char offset in original turn text
    end_char: int


# ── Processed output for a single conversation turn ──────────────────────

class ProcessedTurn(BaseModel):
    turn_index: int
    speaker_label: str           # raw label from transcript  e.g. "Agent"
    role: Role
    original_text: str
    cleaned_text: str            # lowercased, whitespace-normalised
    lemmatized_text: str         # space-joined lemmas from Stanza
    language: str                # BCP-47 code   e.g. "en", "ml", "hi"
    language_confidence: float = Field(ge=0.0, le=1.0)
    entities: list[NamedEntity] = []
    tokens: list[str] = []       # list of surface-form tokens


# ── Full text pipeline output ─────────────────────────────────────────────

class TextPipelineOutput(BaseModel):
    raw_transcript: str
    dominant_language: str       # language detected across the whole call
    turns: list[ProcessedTurn]
    all_entities: list[NamedEntity] = []   # deduplicated across all turns
    speaker_count: int
