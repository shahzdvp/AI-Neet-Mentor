"""
src/core/chat_engine.py — Chat Engine (The Orchestrator)
==========================================================
This is the most important file in the backend.

It knows NOTHING about HTTP (no Flask imports).
It knows NOTHING about which LLM or vector DB is being used.
Its only job: coordinate the steps to answer a question.

THE RAG PIPELINE (step by step):
─────────────────────────────────
  Student's Question
        │
        ▼
  [1] EMBED the question
        │  Converts the text → a vector (list of 384 numbers)
        │  e.g. "What is ATP?" → [0.23, -0.11, 0.87, ...]
        ▼
  [2] RETRIEVE relevant NCERT chunks from ChromaDB
        │  ChromaDB finds the N chunks whose vectors are closest to the question vector
        │  "Closest" = most semantically similar
        ▼
  [3] BUILD the prompt
        │  Combines: system instruction + retrieved chunks + conversation history + question
        ▼
  [4] CALL Groq LLM
        │  The LLM sees the NCERT context and answers based on it
        │  This is what makes it "grounded" — it can't make up facts not in the context
        ▼
  [5] SAVE to SQLite
        │  Stores the Q&A pair for chat history
        ▼
  Answer returned to the API layer
"""

from src.rag.retriever import NCERTRetriever
from src.services.groq_service import GroqService
from src.utils.prompt_builder import PromptBuilder
from src.db.sqlite_db import SQLiteDB


class ChatEngine:
    """
    Orchestrates the full RAG pipeline for one question.

    Receives all dependencies injected (passed in) rather than creating them.
    WHY? This is called Dependency Injection.
    - Easier to test: pass in a mock ChromaDB instead of the real one.
    - No global state: each request is self-contained.
    """

    def __init__(self, chroma, embedder, sqlite: SQLiteDB, config: dict):
        self.retriever = NCERTRetriever(chroma=chroma, embedder=embedder, top_k=config["RAG_TOP_K"])
        self.llm       = GroqService(api_key=config["GROQ_API_KEY"], model=config["GROQ_MODEL"],
                                     max_tokens=config["LLM_MAX_TOKENS"], temperature=config["LLM_TEMPERATURE"])
        self.sqlite    = sqlite
        self.config    = config

    def answer(self, message: str, subject: str, session_id: str) -> dict:
        """
        Full pipeline: retrieve → prompt → LLM → save → return.

        Returns:
            {
              "answer": "...",
              "sources": [{"chapter": "...", "text_preview": "..."}]
            }
        """

        # ── Step 1: Retrieve relevant NCERT passages ───────────────────────
        # retriever.get_context() embeds the question and queries ChromaDB.
        # `chunks` is a list of dicts: [{"text": "...", "metadata": {...}}, ...]
        chunks = self.retriever.get_context(query=message, subject=subject)

        # ── Step 2: Load recent conversation history from SQLite ───────────
        # We pass the last 6 exchanges (12 messages) to the LLM.
        # This gives the LLM "memory" within a session.
        history = self.sqlite.get_recent_history(session_id=session_id, limit=6)

        # ── Step 3: Build the final prompt ────────────────────────────────
        # PromptBuilder assembles:
        #   system message (NEET tutor instructions)
        #   + NCERT context (retrieved chunks)
        #   + history (previous Q&A)
        #   + current question
        messages = PromptBuilder.build(
            question=message,
            subject=subject,
            context_chunks=chunks,
            history=history,
        )

        # ── Step 4: Call the LLM ──────────────────────────────────────────
        answer_text = self.llm.complete(messages=messages)

        # ── Step 5: Save to chat history ──────────────────────────────────
        self.sqlite.save_message(session_id=session_id, role="user",    content=message)
        self.sqlite.save_message(session_id=session_id, role="assistant", content=answer_text)

        # ── Step 6: Format sources for the frontend ───────────────────────
        # The frontend can optionally show "Source: NCERT Class 11 Biology, Chapter 13"
        sources = [
            {
                "subject":      chunk.get("metadata", {}).get("subject", ""),
                "class":        chunk.get("metadata", {}).get("class", ""),
                "chapter":      chunk.get("metadata", {}).get("chapter", "NCERT"),
                "chapter_no":   chunk.get("metadata", {}).get("chapter_no", ""),
                "page_number":  chunk.get("metadata", {}).get("page_number", ""),
                "topic":        chunk.get("metadata", {}).get("topic", ""),
                "weightage":    chunk.get("metadata", {}).get("weightage", ""),
                "relevance":    chunk.get("relevance", 0),
                "pdf_filename": chunk.get("metadata", {}).get("pdf_filename", chunk.get("metadata", {}).get("source", "")),
                "text_preview": chunk.get("text", "")[:120] + "...",
            }
            for chunk in chunks
        ]

        return {"answer": answer_text, "sources": sources, "metadata": sources}
