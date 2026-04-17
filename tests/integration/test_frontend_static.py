from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_root_serves_updated_frontend_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "x")

    import app.config

    app.config._settings = None
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/")

    html = response.text
    assert response.status_code == 200
    assert '<textarea' in html
    assert 'id="q"' in html
    assert 'id="citeTooltip"' in html
    assert "marked.min.js" in html
    assert "purify.min.js" in html
    assert "est. MMXXVI" not in html
