"""
src/api/chat.py — Chat API Endpoint
======================================
This file is the ONLY file that handles HTTP for the chat feature.
It does three things:
  1. Parse and validate the incoming JSON request.
  2. Call the business logic (ChatEngine) — no logic here, just routing.
  3. Return a clean JSON response.

Route: POST /api/chat

Request body (JSON):
  {
    "message":  "What is photosynthesis?",   ← required
    "subject":  "Biology",                   ← optional, default "All"
    "session_id": "abc123"                   ← optional, used for chat history
  }

Response body (JSON):
  {
    "answer":   "Photosynthesis is...",
    "sources":  [{"chapter": "...", "text_preview": "..."}],
    "session_id": "abc123"
  }
"""

import uuid
from flask import Blueprint, request, jsonify, current_app
from pydantic import BaseModel, ValidationError, field_validator

from src.core.chat_engine import ChatEngine

chat_bp = Blueprint("chat", __name__)


# ── Request Model ──────────────────────────────────────────────────────────────
# Pydantic validates incoming JSON automatically.
# If "message" is missing, it raises a ValidationError — we catch it below.
class ChatRequest(BaseModel):
    message: str
    subject: str = "All"       # Default to "All" if not provided
    session_id: str | None = None   # None = new session

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty")
        return v.strip()

    @field_validator("subject")
    @classmethod
    def subject_must_be_valid(cls, v: str) -> str:
        allowed = {"All", "Biology", "Physics", "Chemistry"}
        if v not in allowed:
            return "All"   # Silently default to All for unknown values
        return v


# ── Route Handler ──────────────────────────────────────────────────────────────
@chat_bp.post("/chat")
def chat():
    """
    Handles a student's question and returns an AI-generated answer
    grounded in NCERT content via RAG.
    """

    # ── Step 1: Parse + validate the request ──────────────────────────────
    raw_data = request.get_json(silent=True)
    if not raw_data:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        chat_req = ChatRequest(**raw_data)
    except ValidationError as e:
        # Return a clean error message, not a stack trace.
        errors = [err["msg"] for err in e.errors()]
        return jsonify({"error": "Invalid request", "details": errors}), 422

    # ── Step 2: Assign or reuse a session ID ──────────────────────────────
    # session_id ties multiple messages together into one conversation.
    # If the frontend didn't send one, we create a new UUID.
    session_id = chat_req.session_id or str(uuid.uuid4())

    # ── Step 3: Call the business logic layer ─────────────────────────────
    # ChatEngine does all the heavy lifting:
    #   → retrieves NCERT context (RAG)
    #   → builds the prompt
    #   → calls Groq LLM
    #   → saves the exchange to SQLite
    engine = ChatEngine(
        chroma=current_app.chroma,
        embedder=current_app.embedder,
        sqlite=current_app.sqlite,
        config=current_app.config,
    )

    result = engine.answer(
        message=chat_req.message,
        subject=chat_req.subject,
        session_id=session_id,
    )

    # ── Step 4: Return the response ───────────────────────────────────────
    return jsonify({
        "answer":     result["answer"],
        "sources":    result["sources"],   # Which NCERT chunks were used
        "session_id": session_id,
        "metadata":   result.get("metadata", []),  # Rich metadata for frontend
    }), 200
