# Setup Guide — The Academic Sanctuary Backend

## Prerequisites
- Python 3.11+
- A free Groq API key from https://console.groq.com

---

## Step 1 — Clone and create virtual environment

```bash
# A virtual environment isolates this project's packages from your system Python.
# Think of it as a clean room for this project.

python -m venv venv

# Activate it:
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Your terminal prompt should now show (venv)
```

---

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt

# This installs Flask, Groq, ChromaDB, sentence-transformers, etc.
# First run takes ~2 minutes (sentence-transformers downloads the model).
```

---

## Step 3 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` in any text editor and fill in:
```
GROQ_API_KEY=gsk_your_actual_key_here
```
Everything else has sensible defaults for local development.

---

## Step 4 — Ingest NCERT content (one-time setup)

### Option A — Quick start with demo data (no files needed):
```bash
python scripts/ingest_ncert.py
# Will load 4 demo chunks (photosynthesis, mitochondria, Newton's law, Le Chatelier)
# The AI will work immediately but only knows these 4 topics.
```

### Option B — Real NCERT content (recommended):
1. Get NCERT textbooks as PDF or text files.
2. Name them: `{subject}_{class}_{chapter_hint}.txt`
   Examples:
   - `biology_11_cell_unit_of_life.txt`
   - `chemistry_12_electrochemistry.pdf`
   - `physics_11_laws_of_motion.txt`
3. Place them in `data/ncert_chunks/`
4. Run: `python scripts/ingest_ncert.py`

The script will chunk → embed → store all content. Takes ~1 min per 100 pages.
You only run this when you add new content.

---

## Step 5 — Start the server

```bash
python app.py
```

You should see:
```
INFO: SQLite schema ready at data/academic_sanctuary.db
INFO: ChromaDB ready at data/chroma_store
INFO: Loading embedding model: all-MiniLM-L6-v2
INFO: Embedding model ready.
 * Running on http://0.0.0.0:5000
```

---

## Step 6 — Open the app

Open your browser and go to: **http://127.0.0.1:5000**

The landing page (index.html) loads from Flask's static file server.
Click "Start Free" to open chat.html and ask your first question!

---

## Step 7 — Test the API directly (optional)

```bash
# Health check
curl http://127.0.0.1:5000/api/health

# Send a chat message
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain photosynthesis", "subject": "Biology"}'
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Common Issues

| Problem | Fix |
|---|---|
| `GROQ_API_KEY not set` | Add your key to `.env` |
| `ChromaDB collection is empty` | Run `python scripts/ingest_ncert.py` |
| `ModuleNotFoundError` | Make sure your venv is activated and `pip install -r requirements.txt` ran |
| CORS error in browser | Make sure Flask is running on port 5000 |
| Slow first request | Normal — embedding model loads on first use |
