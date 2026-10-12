"""
tests/test_api.py
==================
Integration tests for the FastAPI backend (api/app.py) using TestClient.

Tests cover:
  - GET /health  → 200 with expected fields
  - GET /runs    → 200 with list response
  - GET /history → 200 with list response
  - POST /runs/enqueue without API key → 401 when API_KEY is set
  - POST /runs/enqueue with correct API key → enqueued (mocked)

Uses in-memory SQLite so no real DB file is created.
No LLM calls are made.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Patch database before importing the app so init_db doesn't touch disk
import db.connection as _db_conn


class TestAPIHealth(unittest.TestCase):
    """Tests for GET /health."""

    def setUp(self):
        # Patch init_db so no real DB is created
        with patch.object(_db_conn, "init_db", return_value=None):
            from fastapi.testclient import TestClient
            from api.app import app
            self.client = TestClient(app, raise_server_exceptions=True)

    def test_health_returns_200(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_health_has_status_ok(self):
        response = self.client.get("/health")
        data = response.json()
        self.assertEqual(data.get("status"), "ok")

    def test_health_has_required_fields(self):
        response = self.client.get("/health")
        data = response.json()
        for field in ("status", "db_path", "dataset_version", "prompt_version",
                      "retriever_version", "embedding_model", "llm_model"):
            self.assertIn(field, data, f"Missing field: {field}")


class TestAPIHistory(unittest.TestCase):
    """Tests for GET /history."""

    def setUp(self):
        with patch.object(_db_conn, "init_db", return_value=None):
            from fastapi.testclient import TestClient
            from api.app import app
            self.client = TestClient(app, raise_server_exceptions=True)

    @patch("api.app.os.path.exists", return_value=False)
    def test_history_returns_empty_list_when_no_file(self, _):
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])


class TestAPIRuns(unittest.TestCase):
    """Tests for GET /runs (list)."""

    def setUp(self):
        with patch.object(_db_conn, "init_db", return_value=None):
            from fastapi.testclient import TestClient
            from api.app import app
            self.client = TestClient(app, raise_server_exceptions=True)

    def test_runs_returns_200(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(_db_conn, "get_db_connection", return_value=mock_conn):
            response = self.client.get("/runs")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)


class TestAPIEnqueueAuth(unittest.TestCase):
    """Tests for POST /runs/enqueue API-key authentication."""

    def _make_client(self, api_key_env=None):
        with patch.object(_db_conn, "init_db", return_value=None):
            from fastapi.testclient import TestClient
            import api.app as app_module
            # Patch the module-level _API_KEY_ENV
            original = app_module._API_KEY_ENV
            app_module._API_KEY_ENV = api_key_env
            from api.app import app
            client = TestClient(app, raise_server_exceptions=False)
            app_module._API_KEY_ENV = original
            return client, app_module

    def test_no_api_key_env_allows_unauthenticated(self):
        """When API_KEY env is unset, no auth is required (local dev mode)."""
        with patch.object(_db_conn, "init_db", return_value=None):
            import api.app as app_module
            original = app_module._API_KEY_ENV
            app_module._API_KEY_ENV = None
            try:
                with patch("core.queue.enqueue_run", return_value="run-test-123"), \
                     patch("core.queue.process_queue"):
                    from fastapi.testclient import TestClient
                    client = TestClient(app_module.app, raise_server_exceptions=False)
                    response = client.post("/runs/enqueue", json={"mode": "smoke", "no_judge": True})
                    # Should not be 401
                    self.assertNotEqual(response.status_code, 401)
            finally:
                app_module._API_KEY_ENV = original

    def test_wrong_api_key_returns_401(self):
        """When API_KEY env is set, wrong key → 401."""
        with patch.object(_db_conn, "init_db", return_value=None):
            import api.app as app_module
            original = app_module._API_KEY_ENV
            app_module._API_KEY_ENV = "correct-key"
            try:
                from fastapi.testclient import TestClient
                client = TestClient(app_module.app, raise_server_exceptions=False)
                response = client.post(
                    "/runs/enqueue",
                    json={"mode": "smoke"},
                    headers={"X-API-Key": "wrong-key"},
                )
                self.assertEqual(response.status_code, 401)
            finally:
                app_module._API_KEY_ENV = original

    def test_correct_api_key_proceeds(self):
        """Correct X-API-Key header passes authentication."""
        with patch.object(_db_conn, "init_db", return_value=None):
            import api.app as app_module
            original = app_module._API_KEY_ENV
            app_module._API_KEY_ENV = "secret-key"
            try:
                with patch("core.queue.enqueue_run", return_value="run-123"), \
                     patch("core.queue.process_queue"):
                    from fastapi.testclient import TestClient
                    client = TestClient(app_module.app, raise_server_exceptions=False)
                    response = client.post(
                        "/runs/enqueue",
                        json={"mode": "smoke", "no_judge": True},
                        headers={"X-API-Key": "secret-key"},
                    )
                    self.assertNotEqual(response.status_code, 401)
            finally:
                app_module._API_KEY_ENV = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
