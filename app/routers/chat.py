"""
ConvI — Chat Router
=====================
Exposes AI chat endpoints:

    POST  /api/v1/chat
        Send a message, get a grounded reply with RAG + analytics context.
        Chat history is persisted in PostgreSQL.
        Supply analytics_session_id to reference a prior /analyze session.

    GET   /api/v1/chat/{chat_session_id}/history
        Retrieve the full message history for a chat session.
"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.chat.schemas import ChatRequest, ChatResponse, ChatHistoryResponse
from app.chat import run_chat, fetch_chat_history

router = APIRouter(prefix="/api/v1/chat", tags=["AI Chat"])


# ── POST /api/v1/chat ─────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=ChatResponse,
    summary="AI Chat — ask questions, get grounded answers",
    description=(
        "Send a natural-language message to the ConvI AI assistant.\n\n"
        "**Features:**\n"
        "- Answers are grounded in banking policy documents via RAG retrieval.\n"
        "- Optionally link a prior `/analyze` session via `analytics_session_id` "
        "to get answers that reference that conversation's summary, risk score, "
        "compliance flags, fraud indicators, etc.\n"
        "- Full multi-turn memory is persisted in PostgreSQL. Omit `chat_session_id` "
        "to start a new session; use the returned `chat_session_id` in follow-up "
        "requests to continue the same conversation.\n\n"
        "**Example questions:**\n"
        "- *What is the RBI guideline for wrong-credit recovery?*\n"
        "- *Is there any fraud risk in the analysed conversation?*\n"
        "- *What should an agent do when a customer reports an unauthorised transaction?*"
    ),
    status_code=status.HTTP_200_OK,
)
def chat(request: ChatRequest) -> ChatResponse:
    logger.info(
        f"[ChatRouter] POST /chat | session={request.chat_session_id!r} | "
        f"analytics_ref={request.analytics_session_id!r}"
    )
    try:
        return run_chat(request)
    except RuntimeError as e:
        # Ollama unreachable or model error
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"[ChatRouter] Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat engine error: {e}",
        )


# ── GET /api/v1/chat/{chat_session_id}/history ────────────────────────────────

@router.get(
    "/{chat_session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Retrieve chat history for a session",
    description=(
        "Returns all messages (user + assistant) for the given `chat_session_id`, "
        "ordered oldest-first."
    ),
    status_code=status.HTTP_200_OK,
)
def chat_history(chat_session_id: str) -> ChatHistoryResponse:
    logger.info(f"[ChatRouter] GET /chat/{chat_session_id}/history")
    result = fetch_chat_history(chat_session_id)
    if result.message_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chat session found with id '{chat_session_id}'.",
        )
    return result
