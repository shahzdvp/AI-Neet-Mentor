# The Academic Sanctuary — Backend

## Architecture Overview

```
academic_sanctuary/
├── src/                        ← All application source code lives here
│   ├── api/                    ← HTTP layer (Flask Blueprints = route handlers)
│   │   ├── __init__.py
│   │   ├── chat.py             ← POST /api/chat  (main chat endpoint)
│   │   └── health.py           ← GET  /api/health (uptime check)
│   │
│   ├── core/                   ← Business logic (no HTTP, no DB — pure logic)
│   │   ├── __init__.py
│   │   └── chat_engine.py      ← Orchestrates RAG + LLM for a response
│   │
│   ├── db/                     ← All database interactions
│   │   ├── __init__.py
│   │   ├── sqlite_db.py        ← User profiles, chat history (SQLite)
│   │   └── chroma_db.py        ← NCERT vector store (ChromaDB)
│   │
│   ├── rag/                    ← Retrieval-Augmented Generation pipeline
│   │   ├── __init__.py
│   │   ├── embedder.py         ← Converts text → vectors (sentence-transformers)
│   │   └── retriever.py        ← Queries ChromaDB, ranks results
│   │
│   ├── services/               ← External service wrappers
│   │   ├── __init__.py
│   │   └── groq_service.py     ← Groq API client (LLM calls)
│   │
│   └── utils/                  ← Shared helpers
│       ├── __init__.py
│       └── prompt_builder.py   ← Assembles the final prompt for the LLM
│
├── scripts/
│   └── ingest_ncert.py         ← One-time script: loads NCERT PDFs into ChromaDB
│
├── tests/                      ← Unit + integration tests
├── data/
│   └── ncert_chunks/           ← Place your NCERT text files / PDFs here
├── frontend/                   ← Copy your index.html, chat.html, style.css here
├── .env.example                ← Template for secrets
├── requirements.txt
├── app.py                      ← Entry point — creates and runs the Flask app
└── config.py                   ← All configuration in one place
```

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Web Framework | Flask | Lightweight, easy to understand, industry standard |
| LLM | Groq (llama-3.1-70b) | Fast inference, free tier available |
| Knowledge Retrieval | RAG + ChromaDB | Keeps answers grounded in NCERT |
| Embeddings | sentence-transformers | Local, free, accurate |
| User Data | SQLite | Simple, no server needed |
| Vector Store | ChromaDB | Local-first, production-ready |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # Fill in your GROQ_API_KEY
python scripts/ingest_ncert.py   # One-time: build the vector store
python app.py               # Start the server
```
