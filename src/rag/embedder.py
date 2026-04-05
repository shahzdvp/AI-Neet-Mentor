"""
src/rag/embedder.py — Text Embedder
=====================================
WHY do we need an embedder?
- Both ChromaDB ingestion AND query-time retrieval need to convert text → vectors.
- We centralise that logic here so the model loads ONCE and is reused everywhere.

HOW sentence-transformers works:
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("all-MiniLM-L6-v2")
  vector = model.encode("What is photosynthesis?")
  # vector is a numpy array of 384 floats

MODEL CHOICE — "all-MiniLM-L6-v2":
  - Size: ~22 MB (downloads once, cached locally)
  - Speed: ~14,000 sentences/sec on CPU
  - Quality: Very good for semantic search / Q&A
  - Free: runs 100% locally, no API calls

SINGLETON PATTERN:
  The model takes ~1-2 seconds to load.
  We load it ONCE at app startup (init_embedder in app.py)
  and reuse the same object for every request.
  Calling SentenceTransformer() on every request would be extremely slow.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps the sentence-transformers model.
    Provides a single .embed() method used by both ingestion and retrieval.
    """

    def __init__(self, model_name: str):
        logger.info("Loading embedding model: %s  (this takes ~5s on first run)", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model ready.")

    def embed(self, text: str) -> list[float]:
        """
        Converts a single string to a list of floats (a vector).

        Used at query time: embed the student's question before searching ChromaDB.

        Args:
            text: Any string (question, sentence, paragraph).
        Returns:
            A list of floats. Length = 384 for all-MiniLM-L6-v2.
        """
        vector: np.ndarray = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()   # ChromaDB expects a plain Python list

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Converts a list of strings to a list of vectors.

        Used at ingestion time: embed all NCERT chunks in one batch.
        Batch encoding is much faster than calling .encode() in a loop.

        Args:
            texts: List of strings (NCERT paragraphs).
        Returns:
            List of vectors (one per text).
        """
        vectors: np.ndarray = self._model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=64,         # Process 64 texts at a time
            show_progress_bar=True # Shows a progress bar during ingestion
        )
        return vectors.tolist()


def init_embedder(model_name: str) -> Embedder:
    """
    Called once in app.py's create_app().
    Returns a ready-to-use Embedder instance stored on `app.embedder`.
    """
    return Embedder(model_name)
