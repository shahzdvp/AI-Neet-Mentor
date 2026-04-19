"""
scripts/ingest_ncert.py — NCERT Data Ingestion Script
=======================================================
RUN THIS ONCE (or whenever you add new NCERT content):
  python scripts/ingest_ncert.py

WHAT IT DOES:
  1. Reads NCERT text files or PDFs from data/ncert_chunks/
  2. Splits them into smart overlapping chunks
  3. Embeds each chunk (text → vector)
  4. Stores the vector + original text + metadata in ChromaDB

WHY CHUNKING MATTERS:
  NCERT chapters can be 5,000+ words. We can't embed a whole chapter as one vector
  because important details get "averaged out" and retrieval becomes inaccurate.

  Instead we split into chunks of ~400 words with 50-word overlap.
  The overlap ensures a sentence that falls on a chunk boundary isn't lost.

  Example:
    Chapter text: "...mitochondria produce ATP. This process is called..."
    Chunk 1 ends: "...mitochondria produce ATP."
    Chunk 2 starts: "...mitochondria produce ATP. This process is called..."  ← overlap keeps context

METADATA:
  Each chunk is stored with metadata so we can filter by subject/chapter later:
  {
    "subject":  "Biology",
    "class":    "11",
    "chapter":  "Cell: The Unit of Life",
    "source":   "bio_class11_ch08.txt"
  }

HOW TO PROVIDE NCERT CONTENT:
  Option A — Text files (simplest):
    Place .txt files in data/ncert_chunks/
    Filename format: {subject}_{class}_{chapter_hint}.txt
    e.g. biology_11_cell_the_unit_of_life.txt

  Option B — PDFs:
    Place .pdf files in data/ncert_chunks/
    The script will extract text automatically using pypdf.

  The metadata is parsed from the filename. You can also edit the
  FILE_METADATA dict at the bottom of this script to add exact chapter names.
"""

import sys
import os
import logging

# Add project root to path so we can import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from config import get_config
from src.db.chroma_db import init_chroma, get_or_create_collection
from src.rag.embedder import init_embedder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Chunking Parameters ────────────────────────────────────────────────────────
# chunk_size: maximum characters per chunk (~400 words ≈ 2000 chars)
# chunk_overlap: characters shared between consecutive chunks
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200

DATA_DIR = "data/ncert_chunks"

# ── Optional: Map filename → exact chapter metadata ───────────────────────────
# If a file isn't listed here, metadata is parsed from the filename automatically.
FILE_METADATA = {
    "Chemistry_11_01_Some_Basic_Concepts_of_Chemistry.pdf": {
        "subject":    "Chemistry",
        "class":      "11",
        "chapter":    "Some Basic Concepts of Chemistry",
        "chapter_no": "01",
        "topic":      "Mole Concept, Atomic Mass, Molecular Mass, Stoichiometry",
        "weightage":  "high",
        "source":       "Chemistry_11_01_Some_Basic_Concepts_of_Chemistry.pdf",
        "pdf_filename": "Chemistry_11_01_Some_Basic_Concepts_of_Chemistry.pdf"
    },
    "Biology_11_01_The_Living_World.pdf": {
        "subject":    "Biology",
        "class":      "11",
        "chapter":    "The Living World",
        "chapter_no": "01",
        "topic":      "Taxonomy, Biodiversity, Nomenclature, Classification",
        "weightage":  "medium",
        "source":     "Biology_11_01_The_Living_World.pdf",
        "pdf_filename": "Biology_11_01_The_Living_World.pdf"
    },
    "Physics_11_01_Physical_World.pdf": {
        "subject":    "Physics",
        "class":      "11",
        "chapter":    "Physical World",
        "chapter_no": "01",
        "topic":      "Scope of Physics, Fundamental Forces, Nature of Science",
        "weightage":  "low",
        "source":     "Physics_11_01_Physical_World.pdf",
        "pdf_filename": "Physics_11_01_Physical_World.pdf"
    },
}


def extract_text_from_file(filepath: str) -> str:
    """Reads text from .txt or .pdf files."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        reader = PdfReader(filepath)
        pages  = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    else:
        logger.warning("Skipping unsupported file type: %s", filepath)
        return ""


def parse_metadata_from_filename(filename: str) -> dict:
    """
    Parses metadata from filename.
    Expected format: {subject}_{class}_{chapter_hint}.txt
    e.g. biology_11_photosynthesis.txt → subject=Biology, class=11, chapter=photosynthesis
    """
    name = os.path.splitext(filename)[0]     # Remove extension
    parts = name.split("_")

    subject = parts[0].capitalize() if len(parts) > 0 else "Unknown"
    cls     = parts[1] if len(parts) > 1 else ""
    chapter = " ".join(parts[2:]).replace("_", " ").title() if len(parts) > 2 else name

    return {"subject": subject, "class": cls, "chapter": chapter, "source": filename, "pdf_filename": filename}

def ingest_pdf_with_pages(filepath: str, meta: dict, collection, embedder, splitter) -> int:
    """
    Ingests a PDF file page by page and stores page number in each chunk's metadata.
    """
    reader      = PdfReader(filepath)
    filename    = os.path.basename(filepath)
    all_chunks    = []
    all_metadatas = []
    all_ids       = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        page_chunks = splitter.split_text(text)

        for i, chunk in enumerate(page_chunks):
            chunk_meta = meta.copy()
            chunk_meta["page_number"] = str(page_num)
            chunk_meta["pdf_filename"] = filename

            all_chunks.append(chunk)
            all_metadatas.append(chunk_meta)
            all_ids.append(f"{filename}_page{page_num}_chunk{i}")

    if not all_chunks:
        return 0

    vectors = embedder.embed_batch(all_chunks)
    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=vectors,
        metadatas=all_metadatas,
    )
    return len(all_chunks)

def ingest():
    """Main ingestion function."""
    cfg       = get_config()
    chroma    = init_chroma(cfg.CHROMA_DB_PATH)
    embedder  = init_embedder(cfg.EMBEDDING_MODEL)
    collection = get_or_create_collection(chroma)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Split on paragraph → sentence → word boundaries (in that priority order)
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.warning(
            "Created empty data directory: %s\n"
            "Place your NCERT .txt or .pdf files there and re-run this script.",
            DATA_DIR,
        )
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".pdf"))]
    if not files:
        logger.warning(
            "No .txt or .pdf files found in %s.\n"
            "Add NCERT content files and re-run.", DATA_DIR
        )
        _ingest_demo_content(collection, embedder, splitter)
        return

    total_chunks = 0
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        logger.info("Processing: %s", filename)

        meta = FILE_METADATA.get(filename) or parse_metadata_from_filename(filename)
        ext  = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            count = ingest_pdf_with_pages(filepath, meta, collection, embedder, splitter)
            logger.info("  → %d chunks (with page numbers)", count)
            total_chunks += count

        elif ext == ".txt":
            text = open(filepath, encoding="utf-8").read()
            if not text.strip():
                logger.warning("Empty content in %s, skipping.", filename)
                continue
            chunks  = splitter.split_text(text)
            vectors = embedder.embed_batch(chunks)
            ids     = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
            metas   = []
            for i in range(len(chunks)):
                m = meta.copy()
                m["page_number"] = "—"
                metas.append(m)
            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=vectors,
                metadatas=metas,
            )
            logger.info("  → %d chunks", len(chunks))
            total_chunks += len(chunks)   


    logger.info("✅ Ingestion complete. Total chunks stored: %d", total_chunks)
    logger.info("   ChromaDB collection '%s' now has %d documents.", 
                collection.name, collection.count())


def _ingest_demo_content(collection, embedder, splitter):
    """
    Ingests a tiny demo dataset so the app works out-of-the-box
    even without real NCERT files. Replace with real NCERT content.
    """
    logger.info("Loading demo NCERT content for development testing...")

    demo_docs = [
        {
            "text": (
                "Photosynthesis is the process by which green plants manufacture food "
                "from carbon dioxide and water using light energy. "
                "The overall equation for photosynthesis is: "
                "6CO2 + 6H2O + light energy → C6H12O6 + 6O2. "
                "Photosynthesis occurs in the chloroplasts. "
                "It has two stages: the light reactions (in thylakoid membranes) "
                "and the Calvin cycle (in the stroma). "
                "The oxygen released during photosynthesis comes from the splitting "
                "of water molecules in the light reactions — this is called photolysis."
            ),
            "meta": {"subject": "Biology", "class": "11", "chapter": "Photosynthesis in Higher Plants", "source": "demo"},
        },
        {
            "text": (
                "Mitochondria are the powerhouses of the cell. "
                "They produce ATP (adenosine triphosphate) through cellular respiration. "
                "Mitochondria have a double membrane structure: "
                "an outer membrane and a highly folded inner membrane called cristae. "
                "The matrix is the space enclosed by the inner membrane and contains "
                "enzymes for the Krebs cycle. "
                "ATP synthesis occurs at the inner membrane via ATP synthase (F0F1 particles). "
                "Mitochondria contain their own circular DNA and ribosomes — "
                "evidence for the endosymbiotic theory."
            ),
            "meta": {"subject": "Biology", "class": "11", "chapter": "Cell: The Unit of Life", "source": "demo"},
        },
        {
            "text": (
                "Newton's Second Law of Motion states that the force acting on an object "
                "is equal to the mass of the object multiplied by its acceleration: F = ma. "
                "Force is measured in Newtons (N), mass in kilograms (kg), "
                "and acceleration in m/s². "
                "If the net force on an object is zero, the object is in equilibrium — "
                "it either remains at rest or continues moving with constant velocity "
                "(Newton's First Law). "
                "Impulse is the change in momentum: J = F × t = Δp."
            ),
            "meta": {"subject": "Physics", "class": "11", "chapter": "Laws of Motion", "source": "demo"},
        },
        {
            "text": (
                "Le Chatelier's Principle states that if a dynamic equilibrium is disturbed "
                "by changing the conditions (concentration, temperature, pressure), "
                "the system responds to partially counteract the change "
                "and restore a new equilibrium. "
                "For example, in the Haber process N2 + 3H2 ⇌ 2NH3 (exothermic): "
                "increasing pressure shifts equilibrium to the right (fewer moles of gas). "
                "Increasing temperature shifts equilibrium to the left (endothermic direction)."
            ),
            "meta": {"subject": "Chemistry", "class": "11", "chapter": "Equilibrium", "source": "demo"},
        },
    ]

    ids       = [f"demo_{i}" for i in range(len(demo_docs))]
    texts     = [d["text"] for d in demo_docs]
    metadatas = [d["meta"] for d in demo_docs]
    vectors   = embedder.embed_batch(texts)

    collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
    logger.info("✅ Demo content loaded. %d chunks stored.", len(demo_docs))
    logger.info(
        "   Replace with real NCERT .txt/.pdf files in data/ncert_chunks/ "
        "and re-run for full accuracy."
    )


if __name__ == "__main__":
    ingest()
