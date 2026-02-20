"""
RAG Engine — PDF Ingestion Script
==================================
One-time script to:
  1. Extract text from all PDFs in data_source/
  2. Clean and chunk the text
  3. Embed chunks with bge-m3
  4. Build a FAISS index
  5. Save index + metadata to data/faiss_index/

Usage:
    python -m app.rag_engine.ingest
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

import pdfplumber
import faiss
import numpy as np
from FlagEmbedding import FlagModel
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
DATA_SOURCE   = ROOT / "data_source"
INDEX_DIR     = ROOT / "data" / "faiss_index"
INDEX_PATH    = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"

# ── Chunking config ───────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # characters
CHUNK_OVERLAP = 64    # characters

# ── Embedding model ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text page-by-page from a PDF using pdfplumber."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": i + 1, "text": text})
    except Exception as e:
        logger.warning(f"Failed to extract {pdf_path.name}: {e}")
    return pages


def clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters."""
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # strip non-ASCII
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window character-level chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def classify_doc(filename: str) -> str:
    """Label document type based on filename keywords."""
    name = filename.lower()
    if "kyc" in name and "aml" in name:
        return "KYC_AML_Policy"
    if "kyc" in name:
        return "KYC_Policy"
    if "aml" in name:
        return "AML_Policy"
    if "communication" in name or "rbi" in name:
        return "Communication_Policy"
    return "Banking_Policy"


# ─────────────────────────────────────────────────────────────────────────
# Main ingestion
# ─────────────────────────────────────────────────────────────────────────

def build_index():
    logger.info("=" * 60)
    logger.info("ConvI RAG — Starting PDF Ingestion")
    logger.info("=" * 60)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect all chunks + metadata from all PDFs
    all_chunks: list[str] = []
    metadata:   list[dict] = []

    pdf_files = sorted(DATA_SOURCE.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {DATA_SOURCE}")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

    for pdf_path in pdf_files:
        doc_type = classify_doc(pdf_path.name)
        logger.info(f"📄 Processing [{doc_type}]: {pdf_path.name}")

        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            logger.warning(f"  → No extractable text, skipping.")
            continue

        doc_chunks = 0
        for page_data in pages:
            cleaned = clean_text(page_data["text"])
            chunks = chunk_text(cleaned)
            for chunk in chunks:
                all_chunks.append(chunk)
                metadata.append({
                    "chunk_id": len(metadata),
                    "source":   pdf_path.name,
                    "doc_type": doc_type,
                    "page":     page_data["page"],
                    "text":     chunk,
                })
                doc_chunks += 1

        logger.info(f"  → {len(pages)} pages, {doc_chunks} chunks")

    logger.info(f"\n📦 Total chunks to embed: {len(all_chunks)}")

    if not all_chunks:
        logger.error("No chunks created — check PDF content.")
        return

    # 2. Embed all chunks with bge-m3
    logger.info(f"\n🤖 Loading embedding model: {EMBEDDING_MODEL}")
    model = FlagModel(
        EMBEDDING_MODEL,
        use_fp16=False,            # CPU-safe
        normalize_embeddings=True, # cosine via inner product
        query_instruction_for_retrieval="Represent this banking document chunk:",
    )

    logger.info("⚙️  Embedding chunks in batches of 8 (CPU — please wait)...")
    BATCH_SIZE = 8
    all_embeddings = []
    batches = [all_chunks[i:i+BATCH_SIZE] for i in range(0, len(all_chunks), BATCH_SIZE)]
    for batch in tqdm(batches, desc="Embedding", unit="batch"):
        batch_emb = model.encode(batch)
        all_embeddings.append(np.array(batch_emb, dtype=np.float32))
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    logger.info(f"✅ Embeddings shape: {embeddings.shape}")

    # 3. Build FAISS index (Inner Product on normalized vectors = cosine)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"✅ FAISS index built — {index.ntotal} vectors, dim={dim}")

    # 4. Save index + metadata
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.success(f"\n🎉 Ingestion complete!")
    logger.success(f"   FAISS index  → {INDEX_PATH}")
    logger.success(f"   Metadata     → {METADATA_PATH} ({len(metadata)} chunks)")


if __name__ == "__main__":
    build_index()
