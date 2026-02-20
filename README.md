# ConvI — Multimodal Conversation Intelligence API

> **Transight Hackathon** | Financial Banking Domain | Local-First Backend

## Overview

ConvI is an **enterprise-grade, local-first** multimodal conversation intelligence backend. It accepts banking customer support conversations as either **audio recordings** or **text transcripts** and returns full structured analytics JSON.

---

## Quick Start

```bash
# 1. Activate virtual environment
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Configure environment
cp .env.example .env
# Edit .env with your model paths and DB credentials

# 4. Run the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API Docs → http://localhost:8000/docs

---

## Project Structure

```
ConvI/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── config.py                  # Settings (pydantic-settings)
│   ├── schemas/                   # Pydantic request/response models
│   ├── routers/
│   │   └── conversation.py        # POST /api/v1/analyze
│   │
│   ├── speech_pipeline/           # DEV 2 — Diarization · ASR · Emotion
│   ├── text_pipeline/             # DEV 1 — FastText · spaCy
│   ├── conversation_normalizer/   # Unified timeline object builder
│   ├── rag_engine/                # DEV 3 — bge-m3 · FAISS retrieval
│   ├── llm_engine/                # DEV 3 — Qwen2.5 reasoning
│   ├── analytics/                 # DEV 4 — Risk · timelines · agent perf
│   └── storage/                   # DEV 4 — PostgreSQL / SQLAlchemy
│
├── data/
│   └── faiss_index/               # FAISS vector store (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## API Endpoints

| Method | Path              | Description                          |
| ------ | ----------------- | ------------------------------------ |
| GET    | `/`               | Service info & health                |
| GET    | `/health`         | Health check                         |
| POST   | `/api/v1/analyze` | Analyze conversation (audio \| text) |

### POST `/api/v1/analyze`

**Multipart form fields:**

| Field             | Type           | Required                          |
| ----------------- | -------------- | --------------------------------- |
| `audio_file`      | File (WAV/MP3) | One of these                      |
| `text_transcript` | string         | is required                       |
| `domain`          | string         | No (default: `financial_banking`) |
| `session_id`      | string         | No                                |

---

## Module Ownership

| Module                    | Developer | Tech Stack                                    |
| ------------------------- | --------- | --------------------------------------------- |
| `speech_pipeline`         | DEV 2     | pyannote.audio · faster-whisper · speechbrain |
| `text_pipeline`           | DEV 1     | fastText · spaCy                              |
| `conversation_normalizer` | Shared    | Python · Qwen2.5                              |
| `rag_engine`              | DEV 3     | bge-m3 · FAISS                                |
| `llm_engine`              | DEV 3     | Qwen2.5 Instruct                              |
| `analytics`               | DEV 4     | Custom Python                                 |
| `storage`                 | DEV 4     | SQLAlchemy · PostgreSQL                       |

---

## Environment Variables

See `.env.example` for all configurable settings.

---

## Tech Stack

- **API**: FastAPI + Uvicorn
- **Speech**: pyannote.audio · faster-whisper · speechbrain
- **NLP**: fastText · spaCy
- **RAG**: bge-m3 · FAISS
- **LLM**: Qwen2.5 Instruct (local)
- **Storage**: PostgreSQL + SQLAlchemy
