"""
src/rag/retriever.py — NCERT Context Retriever
================================================
This is the "R" in RAG — Retrieval.

WHAT IT DOES:
  Given a student's question, it finds the most relevant paragraphs
  from the NCERT textbooks stored in ChromaDB.

THE RETRIEVAL PROCESS:
  1. Embed the question (text → vector)
  2. Ask ChromaDB: "find me the top_k vectors closest to this question vector"
  3. ChromaDB returns the matching text chunks + their metadata
  4. (Optional) Filter by subject if the student selected Biology/Physics/Chemistry

WHY DOES THIS WORK?
  Because semantically similar text has geometrically close vectors.
  "Explain the role of mitochondria" will be close to the NCERT paragraph
  "Mitochondria are called the powerhouse of the cell because..."
  even though the words are different. This is the power of embeddings.

METADATA FILTERING:
  Each stored chunk has metadata like {"subject": "Biology", "chapter": "Cell"}.
  ChromaDB can filter on metadata BEFORE doing the vector search.
  So "subject=Biology" + question → only Biology chunks are searched.
  This is faster and more accurate than searching all subjects.
"""

import logging
from typing import List

from src.db.chroma_db import get_or_create_collection
from src.rag.embedder import Embedder

logger = logging.getLogger(__name__)


class NCERTRetriever:
    """
    Retrieves relevant NCERT chunks for a given question.
    """

    def __init__(self, chroma, embedder: Embedder, top_k: int = 4):
        """
        Args:
            chroma:   The ChromaDB client (stored on app.chroma)
            embedder: The Embedder instance (stored on app.embedder)
            top_k:    How many chunks to retrieve per question (configurable in .env)
        """
        self._collection = get_or_create_collection(chroma)
        self._embedder   = embedder
        self._top_k      = top_k

    def get_context(self, query: str, subject: str = "All", filters: dict =None) -> List[dict]:
        """
        Main method: given a question, return the top relevant NCERT chunks.

        Args:
            query:   The student's question (raw text)
            subject: "All" | "Biology" | "Physics" | "Chemistry"

        Returns:
            List of dicts: [{"text": "...", "metadata": {...}}, ...]
        """

        # ── Step 1: Check if we have any data ─────────────────────────────
        # If the NCERT data hasn't been ingested yet, return empty list
        # (the LLM will still answer from its training, just without RAG context)
        if self._collection.count() == 0:
            logger.warning(
                "ChromaDB collection is empty. "
                "Run `python scripts/ingest_ncert.py` to load NCERT content."
            )
            return []

        # ── Step 2: Embed the question ────────────────────────────────────
        query_vector = self._embedder.embed(query)

        # ── Step 3: Build optional metadata filter ────────────────────────
        # ChromaDB's `where` parameter filters documents before vector search.
        # Only filter if a specific subject is selected (not "All").
        # ── Step 3: Build optional metadata filter ────────────────────────
        where_filter = None
        conditions   = []

        if subject != "All":
            conditions.append({"subject": {"$eq": subject}})

        if filters:
            for key, value in filters.items():
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        # ── Step 4: Query ChromaDB ────────────────────────────────────────
        try:
            results = self._collection.query(
                query_embeddings=[query_vector],   # Must be a list of vectors
                n_results=min(self._top_k, self._collection.count()),
                where=where_filter,                # Metadata filter (or None)
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return []

        # ── Step 5: Format and return ─────────────────────────────────────
        # ChromaDB returns results nested in lists (because you can query
        # multiple vectors at once). We queried one vector, so [0] unwraps it.
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Distance is a cosine distance (0 = identical, 2 = opposite).
            # We only include chunks with reasonable similarity (distance < 1.0).
            if dist < 1.0:
                chunks.append({
                    "text":      doc,
                    "metadata":  meta,
                    "relevance": round(1 - dist, 3),  # Convert to similarity score
                })

        logger.debug("Retrieved %d chunks for query: '%s...'", len(chunks), query[:50])
        return chunks
