"""
src/db/chroma_db.py — ChromaDB Vector Store Setup
====================================================
WHY ChromaDB?
- A vector database stores vectors (embeddings) alongside their original text.
- Standard SQL can't do "find text similar to this question" — vector DB can.
- ChromaDB runs locally (no server), is free, and is production-ready.

WHAT IS A VECTOR / EMBEDDING?
- An embedding is a list of numbers (e.g. 384 floats) that represents the
  *meaning* of a piece of text.
- Texts with similar meaning have vectors that are geometrically close.
- "What is ATP?" and "Define adenosine triphosphate" → close vectors.
- "What is ATP?" and "How do I cook pasta?" → far apart vectors.

HOW CHROMA WORKS:
  1. At ingestion time: we embed every NCERT paragraph and store it.
  2. At query time: we embed the student's question and ask ChromaDB
     "which stored vectors are closest to this?" → get top N chunks.

COLLECTION:
  A ChromaDB "collection" is like a table — all NCERT content lives in
  one collection called "ncert_knowledge_base".
"""

import logging
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ncert_knowledge_base"


def init_chroma(chroma_path: str) -> chromadb.ClientAPI:
    """
    Creates (or opens) the persistent ChromaDB client.

    PersistentClient saves the vector store to disk at `chroma_path`.
    On restart, all previously ingested NCERT data is still there.
    """
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(
            anonymized_telemetry=False,   # Don't send usage data to Chroma cloud
        ),
    )
    logger.info("ChromaDB ready at %s", chroma_path)
    return client


def get_or_create_collection(chroma_client: chromadb.ClientAPI):
    """
    Gets the NCERT collection (creates it if first run).

    Called by the ingestion script AND by the retriever.
    Both must use the same collection name so they talk to the same data.
    """
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            # cosine similarity = measures angle between vectors.
            # Better than Euclidean distance for text embeddings.
            "hnsw:space": "cosine"
        },
    )
    return collection
