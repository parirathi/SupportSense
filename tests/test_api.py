"""
Tests for SupportSense Flask Application & API Routes.
"""

import json
import pytest
from app import app


@pytest.fixture
def client():
    """Configures test client for Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """GET / should render the dashboard page with status 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"SupportSense" in response.data
    assert b"Analyze Incoming Feedback" in response.data


def test_health_endpoint(client):
    """GET /health should return status 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_metrics_endpoint(client):
    """GET /api/metrics should return held-out test evaluation metrics."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "held_out_test_metrics" in data
    assert "accuracy" in data["held_out_test_metrics"]
    assert "macro_f1" in data["held_out_test_metrics"]
    assert "limitations" in data


def test_analyze_valid_input(client):
    """POST /api/analyze with valid feedback should return predictions and priority."""
    payload = {"text": "I was overcharged on my account by fifty dollars and need an explanation."}
    response = client.post(
        "/api/analyze",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    result = data["result"]
    assert result["sentiment"] in ["negative", "neutral", "positive"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["priority"] in ["Urgent", "Normal"]
    assert isinstance(result["probabilities"], dict)


def test_analyze_empty_input_rejected(client):
    """POST /api/analyze with empty or whitespace-only text should return 400."""
    payload = {"text": "   "}
    response = client.post(
        "/api/analyze",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "too short" in data["error"].lower()


def test_analyze_too_short_input(client):
    """POST /api/analyze with less than minimum characters should return 400."""
    payload = {"text": "hi"}
    response = client.post(
        "/api/analyze",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "too short" in data["error"].lower()


def test_analyze_excessive_length_rejected(client):
    """POST /api/analyze with text exceeding 2000 characters should return 400."""
    excessive_text = "urgent issue! " * 200  # > 2500 chars
    payload = {"text": excessive_text}
    response = client.post(
        "/api/analyze",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "exceeds maximum limit" in data["error"].lower()


def test_analyze_non_string_type_rejected(client):
    """POST /api/analyze with non-string payload should return 400."""
    payload = {"text": 12345}
    response = client.post(
        "/api/analyze",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "must be a valid string" in data["error"].lower()


def test_analyze_missing_field_rejected(client):
    """POST /api/analyze missing 'text' field should return 400."""
    payload = {"message": "Hello"}
    response = client.post(
        "/api/analyze",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "missing required field" in data["error"].lower()


def test_analyze_non_json_payload_rejected(client):
    """POST /api/analyze with non-JSON content type should return 400."""
    response = client.post(
        "/api/analyze",
        data="raw plain text string",
        content_type="text/plain",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "invalid request" in data["error"].lower()


def test_clear_session_route(client):
    """POST /api/clear-session should clear session history."""
    # First submit an item
    client.post(
        "/api/analyze",
        data=json.dumps({"text": "Where can I find the docs?"}),
        content_type="application/json",
    )
    # Then clear session
    clear_resp = client.post("/api/clear-session")
    assert clear_resp.status_code == 200
    clear_data = json.loads(clear_resp.data)
    assert clear_data["status"] == "success"
