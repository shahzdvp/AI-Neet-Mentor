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

echo "── Running NCERT ingestion (loads demo content if no files present) ──"
python scripts/ingest_ncert.py

echo "── Build complete ──"
