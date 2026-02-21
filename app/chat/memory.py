"""
ConvI — Chat Memory (PostgreSQL)
==================================
Owns the two chat-specific tables:

    chat_sessions    — one row per chat conversation thread
    chat_messages    — individual user / assistant turns per session

Design notes:
  - Uses the SAME database URL as the main storage module but its OWN
    SQLAlchemy Base so it never interferes with the existing tables.
  - init_chat_db() is additive (create_all only — no drop) so chat
    history survives server restarts.
  - All CRUD helpers swallow exceptions and return safe defaults so a
    DB outage never crashes the chat endpoint.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from loguru import logger
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

# ── Engine (shared connection pool, separate declaration) ─────────────────────

_settings = get_settings()

_engine = create_engine(
    _settings.database_url,
    pool_pre_ping=True,
    echo=False,
)
_SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


class _ChatBase(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class ChatSession(_ChatBase):
    """One record per chat conversation thread."""
    __tablename__ = "chat_sessions"

    chat_session_id      = Column(String(64), primary_key=True, index=True)
    analytics_session_id = Column(String(64), nullable=True, index=True)
    domain               = Column(String(64), default="financial_banking")
    created_at           = Column(DateTime, default=datetime.utcnow)


class ChatMessage(_ChatBase):
    """Individual user / assistant message within a chat session."""
    __tablename__ = "chat_messages"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    chat_session_id = Column(String(64), index=True)
    role            = Column(String(16))   # "user" | "assistant"
    content         = Column(Text)
    created_at      = Column(DateTime, default=datetime.utcnow)


# ── Table initialization ──────────────────────────────────────────────────────

def init_chat_db() -> bool:
    """
    Create chat tables if they don't already exist.
    Intentionally additive (no drop) so history persists across restarts.
    """
    try:
        _ChatBase.metadata.create_all(bind=_engine)
        logger.info("[ChatMemory] Tables ready (chat_sessions, chat_messages).")
        return True
    except Exception as e:
        logger.warning(f"[ChatMemory] DB init failed (non-fatal): {e}")
        return False


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def get_or_create_session(
    chat_session_id: Optional[str],
    analytics_session_id: Optional[str] = None,
    domain: str = "financial_banking",
) -> str:
    """
    Return an existing chat session id, or create a new one.
    If chat_session_id is None a new UUID is minted.
    Returns the (possibly new) chat_session_id string.
    """
    sid = chat_session_id or str(uuid.uuid4())
    try:
        with _SessionLocal() as db:
            existing = db.get(ChatSession, sid)
            if existing is None:
                db.add(ChatSession(
                    chat_session_id=sid,
                    analytics_session_id=analytics_session_id,
                    domain=domain,
                ))
                db.commit()
    except Exception as e:
        logger.warning(f"[ChatMemory] get_or_create_session failed: {e}")
    return sid


def save_message(chat_session_id: str, role: str, content: str) -> int:
    """
    Persist a single message.  Returns the new row's primary-key id (or -1 on error).
    role must be "user" or "assistant".
    """
    try:
        with _SessionLocal() as db:
            msg = ChatMessage(
                chat_session_id=chat_session_id,
                role=role,
                content=content,
            )
            db.add(msg)
            db.commit()
            db.refresh(msg)
            return msg.id
    except Exception as e:
        logger.warning(f"[ChatMemory] save_message failed: {e}")
        return -1


def get_history(chat_session_id: str, last_n: int = 20) -> list[dict]:
    """
    Return the most recent `last_n` messages for a session, oldest-first.
    Each item: {"role": str, "content": str, "message_id": int, "created_at": datetime}
    Returns [] on any DB error.
    """
    try:
        with _SessionLocal() as db:
            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.chat_session_id == chat_session_id)
                .order_by(ChatMessage.id.desc())
                .limit(last_n)
                .all()
            )
            rows = list(reversed(rows))   # oldest first
            return [
                {
                    "message_id": r.id,
                    "role":       r.role,
                    "content":    r.content,
                    "created_at": r.created_at,
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning(f"[ChatMemory] get_history failed: {e}")
        return []


def get_full_history(chat_session_id: str) -> list[dict]:
    """
    Return ALL messages for a session (for the /history endpoint).
    Each item: {"message_id", "role", "content", "created_at"}
    """
    try:
        with _SessionLocal() as db:
            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.chat_session_id == chat_session_id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            return [
                {
                    "message_id": r.id,
                    "role":       r.role,
                    "content":    r.content,
                    "created_at": r.created_at,
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning(f"[ChatMemory] get_full_history failed: {e}")
        return []


def get_session_analytics_id(chat_session_id: str) -> Optional[str]:
    """Return the analytics_session_id linked to a chat session, or None."""
    try:
        with _SessionLocal() as db:
            row = db.get(ChatSession, chat_session_id)
            return row.analytics_session_id if row else None
    except Exception as e:
        logger.warning(f"[ChatMemory] get_session_analytics_id failed: {e}")
        return None
