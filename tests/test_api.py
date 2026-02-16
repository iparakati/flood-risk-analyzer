"""Tests for the FastAPI endpoints."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_unknown_dataset():
    resp = client.get("/data/nonexistent?max_records=100")
    assert resp.status_code == 404
