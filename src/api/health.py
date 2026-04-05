"""
src/api/health.py — Health Check Endpoint
==========================================
WHY have a health endpoint?
- Lets you verify the server is running with a simple browser visit.
- Production tools (Docker, Kubernetes, load balancers) ping /api/health
  to know if the service is alive before sending real traffic.

Route: GET /api/health
"""

from flask import Blueprint, jsonify, current_app

# Blueprint = a group of related routes.
# name="health" is just an internal label Flask uses.
health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health_check():
    """
    Returns a JSON object showing which components are ready.
    200 = everything OK.
    500 = something is broken.
    """
    status = {
        "status": "ok",
        "services": {
            "sqlite":  _check_sqlite(),
            "chroma":  _check_chroma(),
            "embedder": _check_embedder(),
        },
    }

    # If any service is down, return HTTP 500.
    all_ok = all(v == "ready" for v in status["services"].values())
    http_code = 200 if all_ok else 500
    return jsonify(status), http_code


def _check_sqlite() -> str:
    try:
        current_app.sqlite.execute("SELECT 1")
        return "ready"
    except Exception:
        return "error"


def _check_chroma() -> str:
    try:
        # heartbeat() is ChromaDB's built-in alive check.
        current_app.chroma.heartbeat()
        return "ready"
    except Exception:
        return "error"


def _check_embedder() -> str:
    try:
        # The embedder is ready if the model object exists on app.
        return "ready" if current_app.embedder else "error"
    except Exception:
        return "error"
