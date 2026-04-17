"""
Integration tests for the system health check endpoint.

This module verifies that the FastAPI application can be correctly initialized
and that the /health endpoint responds appropriately to verify system availability.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_health(tmp_path, monkeypatch):
    """
    Test that the /health endpoint returns a 200 OK status.

    Verifies that the FastAPI application starts up correctly and the basic
    routing is functional.
    """
    # Configure a temporary data directory and a placeholder API key for initialization
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")

    import app.config

    # Reset configuration to ensure environment variables are picked up
    app.config._settings = None
    app = create_app()

    # Use TestClient to simulate a request to the running FastAPI application
    with TestClient(app) as client:
        response = client.get("/health")

    # Assert that the integration between the router and the handler is working
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
