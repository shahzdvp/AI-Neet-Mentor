# 🎓 AI NEET Mentor

> A RAG-powered AI tutoring platform built for NEET aspirants — grounded in NCERT textbooks, zero hallucination.

---

## What is This?

**AI NEET Mentor** is a full-stack AI tutoring application that helps students prepare for the NEET (National Eligibility cum Entrance Test) exam. It combines:

- A **conversational AI tutor** that answers Biology, Physics, and Chemistry doubts using actual NCERT content
- A **mock test engine** that generates NEET-style MCQs with AI-powered explanations
- A **RAG pipeline** (Retrieval-Augmented Generation) that grounds every answer in NCERT textbooks — so the AI can't make things up

Think of it as a knowledgeable senior who has read all the NCERT books and is available 24/7.

---

## Live Features

| Feature | Description |
|---|---|
| 💬 AI Chat Tutor | Ask any NEET subject doubt — get NCERT-grounded answers with source citations |
| 📝 Mock Test Generator | Auto-generate 5–180 NEET-style MCQs by topic, subject, and difficulty |
| 🔍 AI Question Explainer | Get per-question explanations on wrong/skipped answers after a mock test |
| 📚 NCERT Source Linking | Every chat answer shows which chapter and page the answer came from |
| 🔁 Session Memory | The AI remembers your last 6 exchanges for multi-turn conversations |
| 🎯 Subject Filtering | Filter chat and mock tests by Biology, Physics, or Chemistry |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Web Framework | Flask 3.0 | Lightweight, application-factory pattern, Blueprints for clean routing |
| LLM | Groq (`llama-3.1-8b-instant`) | Ultra-fast inference on custom LPUs, free tier ~14,400 req/day |
| RAG / Vector Store | ChromaDB 0.5 | Local-first, metadata filtering, no cloud dependency |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers | 22 MB, runs on CPU, 384-dim vectors |
| Chat History | SQLite (stdlib) | Zero-config, persistent, WAL mode for concurrent reads |
| PDF Ingestion | pypdf + LangChain text splitters | Chunk NCERT PDFs into overlapping passages |
| Validation | Pydantic v2 | Clean request/response models, auto error messages |
| Frontend | Vanilla HTML/CSS/JS | Three pages: landing, chat, mock test — no build step needed |

---

## Architecture

```
AI-Neet-Mentor/
│
├── app.py                      # Entry point — Application Factory (create_app)
├── config.py                   # All settings loaded from .env
│
├── frontend/                   # Served as Flask static files
│   ├── index.html              # Landing page
│   ├── chat.html               # Chat interface
│   ├── mock-test.html          # Mock test + results UI
│   └── style.css               # Shared styles
│
├── src/
│   ├── api/                    # HTTP layer — Flask Blueprints, no business logic here
│   │   ├── chat.py             # POST /api/chat
│   │   ├── mock_test.py        # POST /api/mock/generate  |  POST /api/mock/explain
│   │   ├── followups.py        # Follow-up question suggestions
│   │   └── health.py           # GET /api/health
│   │
│   ├── core/
│   │   └── chat_engine.py      # Orchestrates the full RAG pipeline (no HTTP, no SQL)
│   │
│   ├── db/
│   │   ├── sqlite_db.py        # Chat history storage and retrieval
│   │   └── chroma_db.py        # ChromaDB client initialisation
│   │
│   ├── rag/
│   │   ├── embedder.py         # text → vector (singleton, loaded once at startup)
│   │   └── retriever.py        # Query ChromaDB, filter by subject, rank by similarity
│   │
│   ├── services/
│   │   └── groq_service.py     # Groq API wrapper with rate-limit / error handling
│   │
│   └── utils/
│       └── prompt_builder.py   # Assembles system message + history + NCERT context + question
│
├── scripts/
│   └── ingest_ncert.py         # One-time script: PDF/TXT → chunks → embeddings → ChromaDB
│
├── tests/
│   └── test_chat_api.py
│
├── data/                       # Auto-created on first run
│   ├── academic_sanctuary.db   # SQLite file
│   ├── chroma_store/           # ChromaDB vector index
│   └── ncert_chunks/           # Place your NCERT PDFs/TXTs here
│
├── .env.example
└── requirements.txt
```

---

## How It Works — The RAG Pipeline

Every time a student asks a question, this pipeline runs:

```
Student's Question
      │
      ▼
1. EMBED — Convert question text → 384-dimensional vector
      │    (using all-MiniLM-L6-v2, runs locally)
      ▼
2. RETRIEVE — Search ChromaDB for the most similar NCERT passages
      │    (optional filter: only Biology / Physics / Chemistry chunks)
      │    Returns top-K chunks with relevance scores
      ▼
3. BUILD PROMPT — Assemble the LLM context:
      │    [system message: who the AI is + rules]
      │    + [last 6 chat exchanges from SQLite]
      │    + [retrieved NCERT chunks as context]
      │    + [student's current question]
      ▼
4. CALL LLM — Send to Groq (llama-3.1-8b-instant)
      │    The LLM generates an answer grounded in the NCERT context
      ▼
5. SAVE — Store Q&A pair in SQLite for session memory
      ▼
6. RETURN — Answer + source citations sent back to frontend
```

**Why RAG?** Without it, the LLM would answer from its training data — which may be outdated, incorrect, or not aligned with NCERT. With RAG, the model is forced to reason from the actual textbook text you fed it.

---

## Mock Test Engine

The mock test system works independently of the RAG pipeline — it uses the LLM's NEET knowledge directly:

- **Generation**: `POST /api/mock/generate` sends a carefully engineered prompt to Groq asking for MCQs in a strict JSON format. For counts > 20, it batches into multiple calls and merges.
- **Validation**: Each question is parsed and normalised — bad JSON from the LLM is handled gracefully with a regex-based extractor.
- **Explanation**: `POST /api/mock/explain` takes a single question + the student's answer and returns a targeted explanation — addressing *why their wrong choice was wrong*, not just what the right answer is.

MCQ generation supports:
- Topics: any NCERT topic string, or "Full NEET" for syllabus-wide
- Subjects: Biology, Physics, Chemistry, Mixed (with real NEET distribution)
- Difficulty: Easy, Medium, Hard, Mixed (30/50/20 split mirrors real NEET)
- Count: 1 to 180 questions

---

## API Reference

### `POST /api/chat`

```json
// Request
{
  "message": "Explain the sliding filament theory",
  "subject": "Biology",        // "All" | "Biology" | "Physics" | "Chemistry"
  "session_id": "abc-123"      // optional — omit for a new session
}

// Response
{
  "answer": "The sliding filament theory explains...",
  "sources": [
    {
      "subject": "Biology",
      "class": "11",
      "chapter": "Locomotion and Movement",
      "chapter_no": "20",
      "page_number": "317",
      "text_preview": "Actin and myosin filaments...",
      "relevance": 0.87
    }
  ],
  "session_id": "abc-123"
}
```

### `POST /api/mock/generate`

```json
// Request
{
  "topic": "Cell Cycle and Division",
  "subject": "Biology",
  "count": 10,
  "difficulty": "Medium"
}

// Response
{
  "test_id": "uuid",
  "questions": [
    {
      "id": 1,
      "question": "Which phase of mitosis is the longest?",
      "options": ["Prophase", "Metaphase", "Anaphase", "Telophase"],
      "correct_index": 0,
      "explanation": "Prophase is the longest...",
      "subject": "Biology",
      "chapter": "Cell Cycle and Cell Division"
    }
  ]
}
```

### `POST /api/mock/explain`

```json
// Request
{
  "question": "Which phase...",
  "options": ["Prophase", "Metaphase", "Anaphase", "Telophase"],
  "correct_index": 0,
  "student_index": 1,
  "subject": "Biology"
}

// Response
{ "explanation": "You chose Metaphase, but..." }
```

### `GET /api/health`

Returns `{"status": "ok"}` — use this to verify the server is running.

---

## Setup Guide

### Prerequisites

- Python 3.11
- A free Groq API key — get one at [console.groq.com](https://console.groq.com)

### 1 — Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd AI-Neet-Mentor

python -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
# First run downloads the sentence-transformers model (~22 MB). Takes ~2 min.
```

### 3 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
GROQ_API_KEY=gsk_your_actual_key_here
FLASK_SECRET_KEY=any-long-random-string
```

All other settings have working defaults.

### 4 — Ingest NCERT content (one-time)

**Option A — Quick demo (no files needed):**
```bash
python scripts/ingest_ncert.py
# Loads 4 demo chunks. AI works immediately but only knows those 4 topics.
```

**Option B — Real NCERT textbooks (recommended):**
1. Get NCERT PDFs or text exports
2. Name them: `{subject}_{class}_{topic}.txt` (e.g. `biology_11_cell_unit_of_life.txt`)
3. Place in `data/ncert_chunks/`
4. Run: `python scripts/ingest_ncert.py`

The script chunks → embeds → stores everything. Run once; re-run only when you add new content.

### 5 — Start the server

```bash
python app.py
```

Expected output:
```
INFO: SQLite schema ready at data/academic_sanctuary.db
INFO: ChromaDB ready at data/chroma_store
INFO: Loading embedding model: all-MiniLM-L6-v2
INFO: Embedding model ready.
 * Running on http://0.0.0.0:5000
```

### 6 — Open in browser

Visit **http://127.0.0.1:5000**

- Landing page loads automatically
- Click **Start Mentoring** → opens the chat interface
- Click **Mock Test** → opens the test generator

---

## Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use (70b-versatile is more capable but slower) |
| `LLM_MAX_TOKENS` | `1600` | Max tokens in each LLM response |
| `LLM_TEMPERATURE` | `0.3` | Lower = more factual, higher = more creative |
| `FLASK_ENV` | `development` | `development` or `production` |
| `FLASK_SECRET_KEY` | `dev-secret-...` | Change this in production |
| `SQLITE_DB_PATH` | `data/academic_sanctuary.db` | Where chat history is stored |
| `CHROMA_DB_PATH` | `data/chroma_store` | Where NCERT vectors are stored |
| `RAG_TOP_K` | `4` | How many NCERT chunks to retrieve per question |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY not set` | Add your key to `.env` and restart |
| `ChromaDB collection is empty` | Run `python scripts/ingest_ncert.py` |
| `ModuleNotFoundError` | Activate venv and run `pip install -r requirements.txt` |
| CORS error in browser | Ensure Flask is running on port 5000 |
| Slow first request | Normal — embedding model loads once on first use |
| LLM returns malformed JSON (mock test) | Re-generate — the JSON extractor retries gracefully |
| Rate limit error | Groq free tier has per-minute limits; wait 60s and retry |

---

## Project Design Decisions

**Why Flask over FastAPI?** Flask's simplicity and the Application Factory pattern make it easier to reason about at a learning/project level. FastAPI would add async complexity without significant benefit here.

**Why ChromaDB over Pinecone / Weaviate?** ChromaDB runs entirely locally — no cloud account, no API key, no cost. For a NEET project with a fixed corpus (NCERT books), a local vector store is the right call.

**Why Groq over OpenAI?** Speed. Groq's LPU hardware makes llama-3.1 respond in under a second. For a tutoring app where students want instant feedback, latency matters. The free tier (14,400 req/day) is also generous for development.

**Why separate the Chat Engine from the API layer?** The `ChatEngine` class in `src/core/chat_engine.py` has zero Flask imports. It can be called from a CLI, a test, or a different web framework. This is the Dependency Injection principle — the engine receives its dependencies (ChromaDB, embedder, SQLite) rather than creating them, making it fully testable.

**Why SQLite for chat history?** Chat history is session-scoped relational data. SQLite with WAL mode handles concurrent reads fine at this scale. When the project needs to handle thousands of concurrent users, the `SQLiteDB` class interface can be swapped for Postgres with minimal changes.