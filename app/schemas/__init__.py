"""
ConvI — Pydantic Schemas

Defines all request / response data contracts used across the API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────

class InputType(str, Enum):
    audio = "audio"
    text = "text"


class Role(str, Enum):
    agent = "agent"
    customer = "customer"
    unknown = "unknown"


class EscalationLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


# ── Request ───────────────────────────────────────────────────────────────

class ConversationRequest(BaseModel):
    """
    Main ingestion request.
    Either audio_file (uploaded via multipart) or text_transcript must be present.
    audio_file is handled as an UploadFile at the router level.
    """
    text_transcript: Optional[str] = Field(
        default=None,
        description="Raw text transcript of the conversation (used when no audio file is provided).",
    )
    domain: str = Field(
        default="financial_banking",
        description="Business domain for RAG and compliance context.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for linking multiple requests.",
    )


# ── Unified Conversation Turn ─────────────────────────────────────────────

class ConversationTurn(BaseModel):
    speaker_id: str
    role: Role
    original_text: str
    normalized_text_en: str
    language: str
    emotion: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


# ── Sub-analysis Blocks ───────────────────────────────────────────────────

class BasicConversationalAnalysis(BaseModel):
    conversation_summary: str
    customer_intention: str
    key_topics: list[str]
    overall_emotional_tone: str
    call_outcome: str
    language_detected: str


class RAGBasedAnalysis(BaseModel):
    compliance_flags: list[str]
    fraud_indicators: list[str]
    policy_references: list[str]
    rag_context_chunks: list[str]


class TimelinePoint(BaseModel):
    speaker_id: str
    timestamp: Optional[float]
    emotion: Optional[str]
    sentiment_score: Optional[float]
    risk_score: Optional[float]


class TimelineAnalysis(BaseModel):
    emotion_timeline: list[TimelinePoint]
    sentiment_timeline: list[TimelinePoint]
    risk_timeline: list[TimelinePoint]


class AgentPerformanceAnalysis(BaseModel):
    performance_score: float = Field(..., ge=0.0, le=10.0)
    de_escalation_detected: bool
    tone_shift_detected: bool
    interaction_metrics: dict[str, Any]


class ConfidenceScores(BaseModel):
    transcription: Optional[float] = None
    language_detection: Optional[float] = None
    emotion_detection: Optional[float] = None
    llm_reasoning: Optional[float] = None
    rag_retrieval: Optional[float] = None


# ── Interim Pipeline Response (until full analytics implemented) ───────────

class PipelineSegment(BaseModel):
    """Single turn from speech or text pipeline (interim output)."""
    speaker_id: str
    start_time: float
    end_time: float
    original_text: str
    language: str = "en"
    emotion: Optional[str] = None
    emotion_confidence: float = 0.0
    transcription_confidence: float = 1.0


class PipelineResponse(BaseModel):
    """Interim response from /analyze/audio and /analyze/text."""
    session_id: str
    input_type: InputType
    domain: str = "financial_banking"
    segments: list[PipelineSegment]
    audio_language: Optional[str] = None  # audio only
    total_segments: int = 0
    unique_speakers: int = 0


# ── Final Response ────────────────────────────────────────────────────────

class ConversationAnalyticsResponse(BaseModel):
    session_id: str
    input_type: InputType
    domain: str
    conversation_timeline: list[ConversationTurn]
    basic_conversational_analysis: BasicConversationalAnalysis
    rag_based_analysis: RAGBasedAnalysis
    timeline_analysis: TimelineAnalysis
    agent_performance_analysis: AgentPerformanceAnalysis
    confidence_scores: ConfidenceScores
    risk_score: float = Field(..., ge=0.0, le=10.0)
    escalation_level: EscalationLevel
