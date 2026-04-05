"""
config.py — Central Configuration
===================================
WHY have a dedicated config file?
- All settings are in ONE place. You never hunt through 10 files to change a timeout.
- The app reads from environment variables (.env), so secrets are never in code.
- python-dotenv loads the .env file automatically when the app starts.

HOW IT WORKS:
  os.getenv("GROQ_API_KEY")         → reads from environment
  os.getenv("RAG_TOP_K", "4")       → reads from environment, falls back to "4" if missing
"""

import os
from dotenv import load_dotenv

# load_dotenv() reads the .env file and puts every key=value into os.environ.
# This must run BEFORE we read any os.getenv() calls below.
load_dotenv()


class Config:
    """
    Base configuration.
    All other config classes inherit from this.
    """

    # ── Flask ──────────────────────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-prod")

    # ── Groq LLM ───────────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # ── Database Paths ─────────────────────────────────────────────────────
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/academic_sanctuary.db")
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "data/chroma_store")

    # ── RAG ────────────────────────────────────────────────────────────────
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))

    # ── Embedding Model ────────────────────────────────────────────────────
    # This runs locally — no API key needed.
    # "all-MiniLM-L6-v2" is fast, small (~22MB), and very accurate for Q&A.
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── CORS ───────────────────────────────────────────────────────────────
    # Which origins (domains) are allowed to call our API.
    # "null" covers file:// (opening HTML directly in browser during dev).
    CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:5500", "null", "*"]


class DevelopmentConfig(Config):
    """Settings for local development."""
    DEBUG = True


class ProductionConfig(Config):
    """Settings for production deployment."""
    DEBUG = False
    # In production, CORS_ORIGINS should list only your actual domain.


# ── Config selector ────────────────────────────────────────────────────────────
# The app factory (app.py) calls get_config() to get the right class.
_config_map = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
}

def get_config() -> Config:
    env = os.getenv("FLASK_ENV", "development")
    return _config_map.get(env, DevelopmentConfig)()
