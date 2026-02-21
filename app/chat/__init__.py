"""
ConvI — AI Chat Engine
========================
Provides grounded, memory-backed AI chat using:

  • PostgreSQL  — persistent multi-turn chat history (via chat.memory)
  • FAISS / bge-m3 — RAG document retrieval (via rag_engine)
  • Ollama (Qwen2.5) — local LLM inference (same endpoint as analytics)
  • Stored analytics  — optional structured context from a prior /analyze call

Public API
----------
    from app.chat import run_chat

    response = await run_chat(request)   # returns ChatResponse
"""

from __future__ import annotations

import json
import re
import httpx
from typing import Optional
from loguru import logger

from app.chat.schemas import (
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    ChatMessageOut,
    PolicySource,
)
from app.chat.memory import (
    init_chat_db,
    get_or_create_session,
    save_message,
    get_history,
    get_full_history,
    get_session_analytics_id,
)

import os as _os
_OLLAMA_URL   = _os.getenv("OLLAMA_URL",   "http://localhost:11434")
_OLLAMA_MODEL = _os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
_TIMEOUT      = 180.0
_OLLAMA_CTX   = 8192
_MAX_HISTORY  = 20   # last N turns fed into the prompt

# ── Initialise chat tables on first import ────────────────────────────────────
init_chat_db()


# ═══════════════════════════════════════════════════════════════════════════════
# Analytics context loader
# ═══════════════════════════════════════════════════════════════════════════════

def _load_analytics_context(analytics_session_id: str) -> str:
    """
    Pull structured analytics from PostgreSQL for a prior /analyze session
    and format them as a compact text block for prompt injection.
    Returns an empty string when the session is not found or DB is unavailable.
    """
    try:
        from app.storage import SessionLocal, AnalyticsResult, SessionRecord  # type: ignore

        with SessionLocal() as db:
            session_row   = db.get(SessionRecord,  analytics_session_id)
            analytics_row = db.get(AnalyticsResult, analytics_session_id)

        if not analytics_row and not session_row:
            return ""

        lines: list[str] = [
            f"[Analytics context for session: {analytics_session_id}]",
        ]

        if session_row:
            lines += [
                f"Domain       : {session_row.domain}",
                f"Input type   : {session_row.input_type}",
                f"Risk score   : {session_row.risk_score}",
                f"Escalation   : {session_row.escalation_level}",
                f"Call outcome : {session_row.call_outcome}",
            ]

        if analytics_row:
            basic = analytics_row.basic_analysis_json or {}
            rag   = analytics_row.rag_analysis_json   or {}
            agent = analytics_row.agent_perf_json     or {}

            if basic.get("conversation_summary"):
                lines.append(f"Summary      : {basic['conversation_summary']}")
            if basic.get("customer_intention"):
                lines.append(f"Intent       : {basic['customer_intention']}")
            if basic.get("key_topics"):
                lines.append(f"Key topics   : {', '.join(basic['key_topics'])}")
            if basic.get("overall_emotional_tone"):
                lines.append(f"Emotion tone : {basic['overall_emotional_tone']}")
            if rag.get("compliance_flags"):
                lines.append(f"Compliance   : {', '.join(rag['compliance_flags'])}")
            if rag.get("fraud_indicators"):
                lines.append(f"Fraud flags  : {', '.join(rag['fraud_indicators'])}")
            if agent:
                lines.append(f"Agent score  : {agent.get('performance_score', 'N/A')}")
                lines.append(f"De-escalated : {agent.get('de_escalation_detected', False)}")

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"[Chat] Could not load analytics context: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_chat_prompt(
    user_message: str,
    history: list[dict],
    rag_chunks: list[str],
    analytics_context: str,
    domain: str,
) -> list[dict]:
    """
    Build a multi-turn Ollama messages list:
      [system, ...history, user]

    The system message injects:
      • Domain + role description
      • RAG policy context (up to 5 chunks)
      • Structured analytics context (if available)
    """
    # ── System prompt ─────────────────────────────────────────────────────
    rag_section = (
        "\n\n---\n".join(rag_chunks[:5])
        if rag_chunks
        else "No specific policy documents retrieved for this query."
    )

    system_parts = [
        f"You are an expert financial banking AI assistant for the ConvI system.",
        f"Domain: {domain}",
        "",
        "You help users understand conversation analytics, banking policies, "
        "compliance requirements, and fraud indicators.",
        "Always ground your answers in the provided policy context and analytics data.",
        "Be concise, accurate, and professional.",
    ]

    if analytics_context:
        system_parts += [
            "",
            "=== CONVERSATION ANALYTICS CONTEXT ===",
            analytics_context,
            "=== END ANALYTICS CONTEXT ===",
        ]

    system_parts += [
        "",
        "=== RELEVANT POLICY / DOCUMENT CONTEXT (from knowledge base) ===",
        rag_section,
        "=== END POLICY CONTEXT ===",
    ]

    system_content = "\n".join(system_parts)

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # ── Conversation history ──────────────────────────────────────────────
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # ── Current user message ──────────────────────────────────────────────
    messages.append({"role": "user", "content": user_message})

    return messages


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama caller
# ═══════════════════════════════════════════════════════════════════════════════

def _call_ollama_chat(messages: list[dict]) -> str:
    """
    Call Ollama /api/chat with a multi-turn messages list.
    Returns the assistant reply string.
    Raises RuntimeError on connection failure.
    """
    payload = {
        "model":   _OLLAMA_MODEL,
        "messages": messages,
        "stream":  False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
            "num_ctx":     _OLLAMA_CTX,
        },
    }
    try:
        resp = httpx.post(
            f"{_OLLAMA_URL}/api/chat",
            json=payload,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot reach Ollama at {_OLLAMA_URL}. Is `ollama serve` running?"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_chat(request: ChatRequest) -> ChatResponse:
    """
    Synchronous chat handler.

    Steps:
      1. Resolve / create chat session in PostgreSQL
      2. Load chat history from memory
      3. Load analytics context (if analytics_session_id set)
      4. RAG retrieval for the user query
      5. Build multi-turn prompt and call Ollama
      6. Persist user + assistant messages to memory
      7. Return ChatResponse

    Parameters
    ----------
    request : ChatRequest

    Returns
    -------
    ChatResponse
    """
    # ── 1. Session ────────────────────────────────────────────────────────
    chat_sid = get_or_create_session(
        chat_session_id=request.chat_session_id,
        analytics_session_id=request.analytics_session_id,
        domain=request.domain,
    )

    # If no analytics_session_id in request, check if the session was created
    # with one earlier (user may omit it in follow-up messages).
    analytics_sid = request.analytics_session_id or get_session_analytics_id(chat_sid)

    logger.info(
        f"[Chat] session={chat_sid} | analytics_ref={analytics_sid} | "
        f"msg_len={len(request.message)}"
    )

    # ── 2. History ───────────────────────────────────────────────────────
    history = get_history(chat_sid, last_n=_MAX_HISTORY)

    # ── 3. Analytics context ──────────────────────────────────────────────
    analytics_context = ""
    analytics_loaded  = False
    if analytics_sid:
        analytics_context = _load_analytics_context(analytics_sid)
        analytics_loaded  = bool(analytics_context)

    # ── 4. RAG retrieval ──────────────────────────────────────────────────
    rag_chunks:  list[str]  = []
    rag_sources: list[dict] = []
    try:
        from app.rag_engine import retriever as _rag  # lazy import

        if not _rag.is_ready:
            _rag.load()

        rag_result  = _rag.retrieve(request.message)
        rag_chunks  = rag_result.get("rag_context_chunks", [])
        rag_sources = rag_result.get("policy_references",  [])
        logger.info(f"[Chat] RAG: {len(rag_chunks)} chunks retrieved.")
    except Exception as e:
        logger.warning(f"[Chat] RAG retrieval failed (non-fatal): {e}")

    # ── 5. LLM call ───────────────────────────────────────────────────────
    messages = _build_chat_prompt(
        user_message=request.message,
        history=history,
        rag_chunks=rag_chunks,
        analytics_context=analytics_context,
        domain=request.domain,
    )

    try:
        reply = _call_ollama_chat(messages)
        logger.info(f"[Chat] Ollama reply: {len(reply)} chars.")
    except RuntimeError as e:
        raise  # let router handle this

    # ── 6. Persist to memory ──────────────────────────────────────────────
    save_message(chat_sid, "user",      request.message)
    reply_id = save_message(chat_sid, "assistant", reply)

    # ── 7. Build response ─────────────────────────────────────────────────
    policy_sources = [
        PolicySource(
            source=s["source"],
            page=s["page"],
            doc_type=s["doc_type"],
            score=s["score"],
        )
        for s in rag_sources
    ]

    return ChatResponse(
        chat_session_id=chat_sid,
        message_id=reply_id,
        reply=reply,
        sources=policy_sources,
        rag_chunks_used=len(rag_chunks),
        analytics_context_loaded=analytics_loaded,
    )


# ── Chat history helper (used by GET endpoint) ────────────────────────────────

def fetch_chat_history(chat_session_id: str) -> ChatHistoryResponse:
    """Return the full message history for a chat session."""
    messages_raw = get_full_history(chat_session_id)
    messages = [
        ChatMessageOut(
            message_id=m["message_id"],
            role=m["role"],
            content=m["content"],
            created_at=m["created_at"],
        )
        for m in messages_raw
    ]
    return ChatHistoryResponse(
        chat_session_id=chat_session_id,
        message_count=len(messages),
        messages=messages,
    )
