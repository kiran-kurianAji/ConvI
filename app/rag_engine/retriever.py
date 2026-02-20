"""
RAG Engine — Retriever Service
================================
Loads the FAISS index + metadata at startup and exposes a fast
synchronous retrieval interface used by the LLM engine.

Usage (from other modules):
    from app.rag_engine.retriever import RAGRetriever
    retriever = RAGRetriever()          # loads index once
    results = retriever.retrieve("customer wants refund after fraud")
"""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from FlagEmbedding import FlagModel
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
INDEX_DIR     = ROOT / "data" / "faiss_index"
INDEX_PATH    = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_TOP_K   = 5


class RAGRetriever:
    """
    Singleton-friendly retriever.
    Load once at API startup, call .retrieve() per request.
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        self._index:    Optional[faiss.Index] = None
        self._metadata: Optional[list[dict]]  = None
        self._model:    Optional[FlagModel]   = None
        self._ready = False

    # ── Lazy load ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load FAISS index, metadata, and embedding model into memory."""
        if self._ready:
            return

        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            logger.error(
                "FAISS index not found. "
                "Run: python -m app.rag_engine.ingest"
            )
            raise FileNotFoundError(
                f"Missing FAISS index at {INDEX_PATH}. "
                "Run the ingestion script first."
            )

        logger.info("📂 Loading FAISS index...")
        self._index = faiss.read_index(str(INDEX_PATH))

        logger.info("📂 Loading chunk metadata...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        logger.info(f"🤖 Loading embedding model: {EMBEDDING_MODEL}")
        self._model = FlagModel(
            EMBEDDING_MODEL,
            use_fp16=False,
            normalize_embeddings=True,
            query_instruction_for_retrieval=(
                "Represent this banking query for retrieving relevant policy:"
            ),
        )

        logger.success(
            f"✅ RAG Retriever ready | "
            f"{self._index.ntotal} vectors | "
            f"{len(self._metadata)} chunks"
        )
        self._ready = True

    # ── Core retrieval ────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> dict:
        """
        Embed query and return top-K most relevant chunks.

        Returns:
            {
                "rag_context_chunks": [str, ...],
                "policy_references":  [{"source", "page", "doc_type", "score"}, ...]
            }
        """
        if not self._ready:
            self.load()

        k = top_k or self.top_k

        # Embed query
        query_vec = self._model.encode([query])
        query_vec = np.array(query_vec, dtype=np.float32)

        # Search FAISS  (returns distances + indices)
        scores, indices = self._index.search(query_vec, k)

        rag_context_chunks = []
        policy_references  = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_meta = self._metadata[idx]
            rag_context_chunks.append(chunk_meta["text"])
            policy_references.append({
                "source":   chunk_meta["source"],
                "page":     chunk_meta["page"],
                "doc_type": chunk_meta["doc_type"],
                "score":    round(float(score), 4),
            })

        logger.debug(
            f"RAG query: '{query[:60]}...' | "
            f"top-{k} retrieved from "
            f"{len(set(r['source'] for r in policy_references))} doc(s)"
        )

        return {
            "rag_context_chunks": rag_context_chunks,
            "policy_references":  policy_references,
        }

    # ── Convenience ───────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    def __repr__(self) -> str:
        status = f"{self._index.ntotal} vectors" if self._ready else "not loaded"
        return f"<RAGRetriever [{status}]>"
