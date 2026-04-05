"""
tests/test_chat_api.py — API Tests
====================================
WHY write tests?
- They verify the API contract: if the endpoint breaks, tests catch it instantly.
- They let you refactor confidently — run tests, see green, deploy.
- Industry standard: no production code ships without tests.

HOW FLASK TESTING WORKS:
- Flask has a built-in test client that simulates HTTP requests.
- app.testing = True disables error propagation (shows real errors).
- We pass a test config so tests use an in-memory SQLite DB (not the real one).

RUN TESTS:
  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from app import create_app


@pytest.fixture
def client():
    """
    Pytest fixture: sets up a fresh Flask test client before each test.
    A fixture runs once per test function automatically.
    """
    # Patch the heavy services so tests run fast without real models/APIs
    with patch("app.init_embedder") as mock_emb, \
         patch("app.init_chroma")   as mock_chroma, \
         patch("app.init_sqlite")   as mock_sqlite:

        # Return lightweight mock objects
        mock_emb.return_value    = MagicMock()
        mock_chroma.return_value = MagicMock()
        mock_sqlite.return_value = MagicMock()

        # Also mock ChromaDB heartbeat so health check passes
        mock_chroma.return_value.heartbeat.return_value = 1

        app = create_app()
        app.config["TESTING"] = True

        with app.test_client() as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """GET /api/health should return 200 when all services are mocked."""
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_json_structure(self, client):
        """Health response must have 'status' and 'services' keys."""
        data = client.get("/api/health").get_json()
        assert "status" in data
        assert "services" in data


class TestChatEndpoint:
    def test_missing_body_returns_400(self, client):
        """Sending no body should return 400."""
        resp = client.post("/api/chat", content_type="application/json")
        assert resp.status_code == 400

    def test_empty_message_returns_422(self, client):
        """Sending an empty message string should return 422 (validation error)."""
        resp = client.post(
            "/api/chat",
            json={"message": "   "},   # whitespace only
        )
        assert resp.status_code == 422

    def test_valid_request_calls_engine(self, client):
        """A valid request should reach ChatEngine and return 200."""
        with patch("src.api.chat.ChatEngine") as MockEngine:
            # Make ChatEngine.answer() return a fake result
            mock_instance = MockEngine.return_value
            mock_instance.answer.return_value = {
                "answer":  "Photosynthesis is the process...",
                "sources": [],
            }

            resp = client.post(
                "/api/chat",
                json={"message": "Explain photosynthesis", "subject": "Biology"},
            )

        assert resp.status_code == 200
        data = resp.get_json()
        assert "answer" in data
        assert "session_id" in data
        assert "sources" in data

    def test_session_id_preserved(self, client):
        """If client sends a session_id, it should be echoed back."""
        with patch("src.api.chat.ChatEngine") as MockEngine:
            MockEngine.return_value.answer.return_value = {"answer": "...", "sources": []}

            resp = client.post(
                "/api/chat",
                json={"message": "What is ATP?", "session_id": "test-session-42"},
            )

        data = resp.get_json()
        assert data["session_id"] == "test-session-42"

    def test_invalid_subject_defaults_to_all(self, client):
        """An unrecognised subject should default to 'All' without error."""
        with patch("src.api.chat.ChatEngine") as MockEngine:
            MockEngine.return_value.answer.return_value = {"answer": "...", "sources": []}

            resp = client.post(
                "/api/chat",
                json={"message": "Explain DNA", "subject": "InvalidSubject"},
            )

        # Should still succeed (subject silently defaults to "All")
        assert resp.status_code == 200
