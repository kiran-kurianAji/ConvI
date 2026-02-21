"""
ConvI — Storage Layer
======================
PostgreSQL session memory using SQLAlchemy (sync, simple).

Tables:
  sessions         — one row per API call
  conversation_turns — individual turns per session
  analytics_results  — final JSON analytics per session
  audit_logs         — immutable event log

Design: schema-first, alembic migrations not included here
(for hackathon, tables created via create_all on startup).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from loguru import logger
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Text,
    DateTime,
    JSON,
    Integer,
    Boolean,
    Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


# ── SQLAlchemy setup ──────────────────────────────────────────────────────────

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class SessionRecord(Base):
    __tablename__ = "sessions"

    session_id    = Column(String(64), primary_key=True, index=True)
    domain        = Column(String(64), default="financial_banking")
    input_type    = Column(String(10))                       # "audio" | "text"
    created_at    = Column(DateTime, default=datetime.utcnow)
    risk_score    = Column(Float, nullable=True)
    escalation_level = Column(String(20), nullable=True)
    call_outcome  = Column(String(50), nullable=True)


class ConversationTurnRecord(Base):
    __tablename__ = "conversation_turns"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    session_id    = Column(String(64), index=True)
    turn_index    = Column(Integer)
    speaker_id    = Column(String(32))
    role          = Column(String(16))
    original_text = Column(Text)
    language      = Column(String(8))
    emotion       = Column(String(32), nullable=True)
    start_time    = Column(Float, nullable=True)
    end_time      = Column(Float, nullable=True)


class AnalyticsResult(Base):
    __tablename__ = "analytics_results"

    session_id          = Column(String(64), primary_key=True, index=True)
    created_at          = Column(DateTime, default=datetime.utcnow)
    basic_analysis_json = Column(JSON)
    rag_analysis_json   = Column(JSON)
    timeline_json       = Column(JSON)
    agent_perf_json     = Column(JSON)
    confidence_json     = Column(JSON)
    risk_score          = Column(Float)
    escalation_level    = Column(String(20))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    event      = Column(String(128))
    detail     = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Table initialization ──────────────────────────────────────────────────────

def init_db() -> bool:
    """
    Create all tables if they don't exist.
    Safe to call multiple times (idempotent).
    Returns True on success, False on failure.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("[Storage] Database tables initialized.")
        return True
    except Exception as e:
        logger.warning(f"[Storage] DB init failed (non-fatal): {e}")
        return False


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def save_session(
    session_id: str,
    domain: str,
    input_type: str,
    risk_score: Optional[float] = None,
    escalation_level: Optional[str] = None,
    call_outcome: Optional[str] = None,
) -> bool:
    try:
        with SessionLocal() as db:
            record = SessionRecord(
                session_id=session_id,
                domain=domain,
                input_type=input_type,
                risk_score=risk_score,
                escalation_level=escalation_level,
                call_outcome=call_outcome,
            )
            db.merge(record)
            db.commit()
        return True
    except Exception as e:
        logger.warning(f"[Storage] save_session failed: {e}")
        return False


def save_turns(session_id: str, turns: list) -> bool:
    try:
        with SessionLocal() as db:
            for i, t in enumerate(turns):
                record = ConversationTurnRecord(
                    session_id=session_id,
                    turn_index=i,
                    speaker_id=t.speaker_id,
                    role=t.role.value,
                    original_text=t.original_text,
                    language=t.language,
                    emotion=t.emotion,
                    start_time=t.start_time,
                    end_time=t.end_time,
                )
                db.add(record)
            db.commit()
        return True
    except Exception as e:
        logger.warning(f"[Storage] save_turns failed: {e}")
        return False


def save_analytics(session_id: str, analysis: dict) -> bool:
    """Save full analytics dict to DB. Pydantic models serialized via .model_dump()."""
    def _dump(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "value"):
            return obj.value
        return obj

    try:
        with SessionLocal() as db:
            record = AnalyticsResult(
                session_id=session_id,
                basic_analysis_json=_dump(analysis.get("basic_conversational_analysis")),
                rag_analysis_json=_dump(analysis.get("rag_based_analysis")),
                timeline_json=_dump(analysis.get("timeline_analysis")),
                agent_perf_json=_dump(analysis.get("agent_performance_analysis")),
                confidence_json=_dump(analysis.get("confidence_scores")),
                risk_score=analysis.get("risk_score"),
                escalation_level=_dump(analysis.get("escalation_level")),
            )
            db.merge(record)
            db.commit()
        return True
    except Exception as e:
        logger.warning(f"[Storage] save_analytics failed: {e}")
        return False


def log_event(session_id: str, event: str, detail: Optional[str] = None) -> None:
    try:
        with SessionLocal() as db:
            db.add(AuditLog(session_id=session_id, event=event, detail=detail))
            db.commit()
    except Exception as e:
        logger.warning(f"[Storage] log_event failed: {e}")
