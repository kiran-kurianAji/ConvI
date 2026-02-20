"""
RAG Engine Package
==================
Financial Banking domain knowledge retrieval using bge-m3 + FAISS.

Quickstart:
    # 1. Build the index (run once after adding PDFs to data_source/)
    python -m app.rag_engine.ingest

    # 2. Use retriever in your code
    from app.rag_engine import retriever
    retriever.load()    # call once at startup
    result = retriever.retrieve("customer claims unauthorised transaction")
"""

from app.rag_engine.retriever import RAGRetriever

# Module-level singleton — shared across all pipeline calls
retriever = RAGRetriever()

__all__ = ["retriever", "RAGRetriever"]
