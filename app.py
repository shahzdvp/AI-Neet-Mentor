"""
app.py — Application Entry Point
==================================
WHY use the Application Factory Pattern?
-----------------------------------------
Instead of creating `app = Flask(__name__)` at the top of a file (simple but
rigid), we wrap app creation inside a function called `create_app()`.

Benefits:
1. TESTABILITY — tests can call create_app() with a test config, no side effects.
2. FLEXIBILITY — you can have dev/prod/test versions of the app.
3. NO CIRCULAR IMPORTS — blueprints are registered after the app is fully built.

HOW IT WORKS:
  1. create_app() creates a Flask instance.
  2. It loads config (env vars via config.py).
  3. It initialises shared services (DB connections, ChromaDB, embedder).
  4. It registers Blueprints (route groups).
  5. It returns the ready-to-run app.
"""

import os
from flask import Flask, send_file, abort
from werkzeug.utils import safe_join
from flask_cors import CORS

from config import get_config
from src.db.sqlite_db import init_sqlite
from src.db.chroma_db import init_chroma
from src.rag.embedder import init_embedder
from src.api.chat import chat_bp
from src.api.health import health_bp
from src.api.mock_test import mock_bp
from src.api.followups import followups_bp


def create_app() -> Flask:
    """
    Application Factory.
    
    """
    app = Flask(
        __name__,
        # Serve the frontend HTML/CSS files from the /frontend folder.
        # Flask's static_folder lets us serve static files without extra config.
        static_folder="frontend",
        static_url_path="",
    )

    # ── 1. Load config ──────────────────────────────────────────────────────
    cfg = get_config()
    app.config.from_object(cfg)

    # ── 2. Enable CORS ──────────────────────────────────────────────────────
    # CORS = Cross-Origin Resource Sharing.
    # Browsers block JS from calling a different domain/port by default.
    # Our frontend (port 5500) calls our API (port 5000) → we MUST allow this.
    CORS(app, origins=cfg.CORS_ORIGINS)

    # ── 3. Ensure data directories exist ────────────────────────────────────
    os.makedirs("data", exist_ok=True)

    # ── 4. Initialise databases ─────────────────────────────────────────────
    # We store the DB clients on `app` so every request can access them
    # via `current_app` without re-creating connections.
    app.sqlite = init_sqlite(cfg.SQLITE_DB_PATH)
    app.chroma = init_chroma(cfg.CHROMA_DB_PATH)
    app.embedder = init_embedder(cfg.EMBEDDING_MODEL)

    # ── 5. Register Blueprints (route groups) ───────────────────────────────
    # A Blueprint is a group of related routes.
    # url_prefix means /api/chat, /api/health — all API routes share /api/.
    app.register_blueprint(chat_bp,   url_prefix="/api")
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(mock_bp,   url_prefix="/api")
    app.register_blueprint(followups_bp, url_prefix="/api")

    # ── 6. Root route → serve the landing page ──────────────────────────────
    @app.route("/")
    def index():
        return app.send_static_file("index.html")
    
    @app.route("/mock-test")
    def mock_test():
        return app.send_static_file("mock-test.html")

    @app.route("/api/pdf/<path:filename>")
    def serve_pdf(filename):
        pdf_dir = os.path.abspath("data/ncert_chunks")
        try:
            safe_path = safe_join(pdf_dir, filename)
        except Exception:
            abort(400)
        if not os.path.isfile(safe_path):
            abort(404)
        return send_file(safe_path, mimetype="application/pdf")

    return app


# ── Run directly ────────────────────────────────────────────────────────────
# `python app.py` triggers this block.
# In production, use: gunicorn "app:create_app()"
if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(
        host="0.0.0.0",   # Accept connections from any network interface
        port=5000,
        debug=flask_app.config.get("DEBUG", True),
    )
