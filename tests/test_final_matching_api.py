"""
API integration and privacy tests for Step 10 endpoints.
"""

import pytest
import json
from app import create_app


@pytest.fixture
def client():
    """Provides Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_health_endpoint(client):
    """Verify health endpoint reports operational database and primary model readiness."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] in ["healthy", "degraded"]
    assert "models" in data
    assert data["models"]["primary_model"] == "tfidf_svm"
    assert data["models"]["primary_model_ready"] is True


def test_api_root_endpoint(client):
    """Verify root endpoint reflects Step 10 phase and primary model."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.get_json()
    assert "Step 10" in data["phase"]
    assert data["primary_model"] == "tfidf_svm"



def test_api_match_svm_single_success(client):
    """Verify POST /api/matches/svm single-pair evaluation."""
    payload = {"resume_id": "RES001", "job_id": "JOB001"}
    res = client.post("/api/matches/svm", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["resume_id"] == "RES001"
    assert data["job_id"] == "JOB001"
    assert data["model_name"] == "tfidf_svm"
    assert "decision_score" in data
    assert "is_match" in data
    assert "matched_skills" in data
    assert "missing_skills" in data


def test_api_match_svm_validation_errors(client):
    """Verify 400 Bad Request on invalid payloads."""
    # Missing resume_id
    res1 = client.post("/api/matches/svm", json={"job_id": "JOB001"})
    assert res1.status_code == 400

    # Missing job_id
    res2 = client.post("/api/matches/svm", json={"resume_id": "RES001"})
    assert res2.status_code == 400

    # Non-JSON payload
    res3 = client.post("/api/matches/svm", data="not json", content_type="text/plain")
    assert res3.status_code == 400


def test_api_match_svm_not_found(client):
    """Verify 404 Not Found on nonexistent records."""
    res = client.post("/api/matches/svm", json={"resume_id": "NON_EXISTENT_RES", "job_id": "JOB001"})
    assert res.status_code == 404


def test_api_resume_all_svm_matches_ranking(client):
    """Verify POST /api/resumes/<resume_id>/svm-matches ranking and top_k."""
    payload = {"job_ids": ["JOB001", "JOB002", "JOB003"], "top_k": 2}
    res = client.post("/api/resumes/RES001/svm-matches", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["total_returned"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["rank"] == 1
    assert data["results"][0]["decision_score"] >= data["results"][1]["decision_score"]


def test_api_batch_svm_matches(client):
    """Verify POST /api/matches/svm/batch with error isolation."""
    payload = {
        "resume_id": "RES001",
        "job_ids": ["JOB001", "NON_EXISTENT_JOB", "JOB002"]
    }
    res = client.post("/api/matches/svm/batch", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["total_submitted"] == 3
    assert data["successful_count"] == 2
    assert data["failed_count"] == 1


def test_api_generic_matches_endpoint(client):
    """Verify POST /api/matches generic endpoint defaults to tfidf_svm."""
    payload = {"resume_id": "RES001", "job_id": "JOB001"}
    res = client.post("/api/matches", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["model_name"] == "tfidf_svm"


def test_api_privacy_protection_in_responses(client):
    """Verify sensitive candidate details (email, phone, raw resume text) are NOT leaked in match responses."""
    res = client.post("/api/matches/svm", json={"resume_id": "RES001", "job_id": "JOB001"})
    assert res.status_code == 200
    data_str = json.dumps(res.get_json())

    sensitive_keys = ["raw_text", "extracted_text", "email", "phone", "candidate_name", "password", "db_password"]
    for key in sensitive_keys:
        assert f'"{key}"' not in data_str
