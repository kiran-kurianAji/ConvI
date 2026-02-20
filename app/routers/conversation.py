"""
ConvI — Conversation Router

API endpoints for audio and text conversation analysis.
"""

import uuid
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from typing import Optional
from loguru import logger

from app.config import Settings, get_settings
from app.schemas import (
    PipelineResponse,
    PipelineSegment,
    InputType,
)
from app.speech_pipeline.pipeline import (
    run_speech_pipeline_async,
    save_upload_to_temp,
    delete_temp_file,
)
from app.text_pipeline.text_parser import parse_transcript, text_turns_to_speech_segments

router = APIRouter(prefix="/api/v1", tags=["Conversation Intelligence"])


def _segments_to_response(
    segments: list,
    session_id: str,
    input_type: InputType,
    domain: str,
    audio_language: Optional[str] = None,
) -> PipelineResponse:
    """Convert pipeline segments to PipelineResponse."""
    pipeline_segments = [
        PipelineSegment(
            speaker_id=s.speaker_id,
            start_time=s.start_time,
            end_time=s.end_time,
            original_text=s.original_text,
            language=s.language,
            emotion=s.emotion,
            emotion_confidence=s.emotion_confidence or 0.0,
            transcription_confidence=s.transcription_confidence,
        )
        for s in segments
    ]
    return PipelineResponse(
        session_id=session_id,
        input_type=input_type,
        domain=domain,
        segments=pipeline_segments,
        audio_language=audio_language or (segments[0].audio_language if segments else None),
        total_segments=len(segments),
        unique_speakers=len({s.speaker_id for s in segments}),
    )


@router.post(
    "/analyze/audio",
    response_model=PipelineResponse,
    summary="Analyze audio conversation",
    description=(
        "Upload an **audio file** (WAV preferred). Runs the full speech pipeline: "
        "speaker diarization, ASR, and emotion detection. Returns segmented transcript."
    ),
    status_code=status.HTTP_200_OK,
)
async def analyze_audio(
    audio_file: UploadFile = File(
        ...,
        description="Audio recording (WAV preferred). Multi-speaker supported.",
    ),
    domain: str = Form(default="financial_banking"),
    session_id: Optional[str] = Form(default=None),
    skip_emotion: bool = Form(
        default=False,
        description="Skip emotion detection for faster processing.",
    ),
    settings: Settings = Depends(get_settings),
):
    session = session_id or str(uuid.uuid4())
    logger.info(f"[{session}] Audio upload: {audio_file.filename}")

    tmp_path: Optional[Path] = None
    try:
        tmp_path = await save_upload_to_temp(audio_file)
        segments = await run_speech_pipeline_async(
            audio_path=tmp_path,
            forced_language=None,
            skip_emotion=skip_emotion,
        )

        if not segments:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No speech detected in audio. Ensure the file contains clear speech.",
            )

        return _segments_to_response(
            segments=segments,
            session_id=session,
            input_type=InputType.audio,
            domain=domain,
            audio_language=segments[0].audio_language if segments else None,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {e}",
        )
    finally:
        if tmp_path:
            delete_temp_file(tmp_path)


@router.post(
    "/analyze/text",
    response_model=PipelineResponse,
    summary="Analyze text transcript",
    description=(
        "Submit a **text transcript**. Parses Agent/Customer turns and returns "
        "structured segments. Use format: 'Agent: ...' / 'Customer: ...' or plain lines."
    ),
    status_code=status.HTTP_200_OK,
)
async def analyze_text(
    text_transcript: str = Form(
        ...,
        description="Raw text transcript. Use 'Agent: ...' and 'Customer: ...' for speaker labels.",
    ),
    domain: str = Form(default="financial_banking"),
    session_id: Optional[str] = Form(default=None),
    settings: Settings = Depends(get_settings),
):
    session = session_id or str(uuid.uuid4())
    logger.info(f"[{session}] Text transcript ({len(text_transcript)} chars)")

    turns = parse_transcript(text_transcript)
    segments = text_turns_to_speech_segments(turns)

    if not segments:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No turns parsed from transcript. Use 'Agent: ...' and 'Customer: ...' format.",
        )

    return _segments_to_response(
        segments=segments,
        session_id=session,
        input_type=InputType.text,
        domain=domain,
    )
