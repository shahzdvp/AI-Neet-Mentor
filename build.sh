# ─────────────────────────────────────────────────────────────────
# Render Build Script
# ─────────────────────────────────────────────────────────────────
# Render runs this script during every deploy.
# It installs dependencies and runs NCERT ingestion (if needed).
# ─────────────────────────────────────────────────────────────────

set -o errexit  # Exit on any error

echo "── Installing Python dependencies ──"
pip install --upgrade pip
pip install -r requirements.txt

echo "── Creating data directories ──"
mkdir -p data/ncert_chunks

echo "── Checking ChromaDB vector store ──"
if [ -f "data/chroma_store/chroma.sqlite3" ]; then
    echo "ChromaDB vector database already exists. Skipping ingestion."
else
    echo "Vector store not found. Running NCERT ingestion..."
    python scripts/ingest_ncert.py
fi

echo "── Build complete ──"
