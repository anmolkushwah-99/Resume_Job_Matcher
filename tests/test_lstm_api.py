"""
API Integration Tests for Supervised LSTM Endpoints (Step 8).
Tests POST /api/matches/lstm and POST /api/resumes/<resume_id>/lstm-matches
including request validation, error handling, and privacy protection.
"""

import pytest
from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestLSTMAPI:
    """Tests LSTM Flask API routes."""

    def test_match_lstm_success(self, client):
        response = client.post(
            "/api/matches/lstm",
            json={"resume_id": "RES001", "job_id": "JOB001"}
        )
        assert response.status_code == 200
        data = response.get_json()

        assert data["success"] is True
        assert data["resume_id"] == "RES001"
        assert data["job_id"] == "JOB001"
        assert data["model"] == "embedding_lstm"
        assert data["predicted_label"] in [0, 1]
        assert "lstm_score" in data
        assert 0.0 <= data["lstm_score"] <= 1.0
        assert "threshold" in data

        # Privacy test: sensitive personal data should NOT be present
        for sensitive_key in ["raw_text", "email", "phone", "address", "database", "password"]:
            assert sensitive_key not in data

    def test_match_lstm_missing_json(self, client):
        response = client.post("/api/matches/lstm", data="not json")
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False

    def test_match_lstm_missing_parameters(self, client):
        res1 = client.post("/api/matches/lstm", json={"resume_id": "RES001"})
        assert res1.status_code == 400

        res2 = client.post("/api/matches/lstm", json={"job_id": "JOB001"})
        assert res2.status_code == 400

    def test_match_lstm_nonexistent_resume(self, client):
        response = client.post(
            "/api/matches/lstm",
            json={"resume_id": "NON_EXISTENT_RES", "job_id": "JOB001"}
        )
        assert response.status_code == 404

    def test_match_lstm_nonexistent_job(self, client):
        response = client.post(
            "/api/matches/lstm",
            json={"resume_id": "RES001", "job_id": "NON_EXISTENT_JOB"}
        )
        assert response.status_code == 404

    def test_resume_all_lstm_matches(self, client):
        response = client.post("/api/resumes/RES001/lstm-matches?limit=3")
        assert response.status_code == 200
        data = response.get_json()

        assert data["success"] is True
        assert data["resume_id"] == "RES001"
        assert data["model"] == "embedding_lstm"
        assert len(data["matches"]) <= 3

        scores = [m["lstm_score"] for m in data["matches"]]
        assert scores == sorted(scores, reverse=True)
