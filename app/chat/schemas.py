"""
ConvI — Chat Module Schemas
============================
Pydantic models for the AI chat feature.

The chat endpoint lets a user:
  - Send natural-language questions about a conversation or banking policy
  - Optionally attach a previously analysed session (analytics_session_id)
    to load its structured analytics as extra context
  - Get grounded answers backed by RAG document retrieval
  - Have the full exchange persisted in PostgreSQL (chat memory)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Incoming request ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Sent by the client to POST /api/v1/chat.

    chat_session_id        → if None a new chat session is created and the id
                             is returned in ChatResponse so the client can
                             continue the conversation.

    analytics_session_id   → optional.  When provided the stored analytics
                             for that /analyze call are loaded and injected
                             into the LLM context (summary, intent, risk,
                             compliance flags, fraud indicators …).
    """
    message: str = Field(..., min_length=1, description="User's question or message.")
    chat_session_id: Optional[str] = Field(
        default=None,
        description="Existing chat session to continue. Omit to start a new session.",
    )
    analytics_session_id: Optional[str] = Field(
        default=None,
        description=(
            "Session ID from a previous /analyze call. "
            "Loads its analytics as context for grounded answers."
        ),
    )
    domain: str = Field(
        default="financial_banking",
        description="Domain hint for RAG retrieval.",
    )


# ── Per-message record (used in history endpoint) ────────────────────────────

class ChatMessageOut(BaseModel):
    """A single message in the conversation history."""
    message_id: int
    role: str          # "user" | "assistant"
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Response ──────────────────────────────────────────────────────────────────

class PolicySource(BaseModel):
    """A single policy document reference surfaced by RAG retrieval."""
    source: str
    page: int
    doc_type: str
    score: float


class ChatResponse(BaseModel):
    """
    Returned by POST /api/v1/chat.
    """
    chat_session_id: str = Field(..., description="Use this in subsequent requests to continue the chat.")
    message_id: int = Field(..., description="DB primary key of the assistant reply saved in memory.")
    reply: str = Field(..., description="LLM answer, grounded in RAG context and analytics.")
    sources: list[PolicySource] = Field(default_factory=list, description="Policy documents used.")
    rag_chunks_used: int = Field(default=0)
    analytics_context_loaded: bool = Field(
        default=False,
        description="True when analytics from analytics_session_id were injected.",
    )


# ── Chat history response ─────────────────────────────────────────────────────

class ChatHistoryResponse(BaseModel):
    chat_session_id: str
    message_count: int
    messages: list[ChatMessageOut]
