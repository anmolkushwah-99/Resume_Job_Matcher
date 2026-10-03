"""
Unit and integration tests for Step 11 frontend routes and supporting APIs.
Ensures zero regression for existing API endpoints while validating new UI routes.
"""

import json
import pytest
from app import create_app


@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_root_html_negotiation(client):
    """Browser requesting HTML from root gets the dashboard UI."""
    response = client.get("/", headers={"Accept": "text/html"})
    assert response.status_code == 200
    assert b"ResumeMatch AI" in response.data
    assert b"<!DOCTYPE html>" in response.data


def test_root_json_backward_compatibility(client):
    """API clients requesting JSON from root get the system manifest (Step 1-10 contract)."""
    response = client.get("/", headers={"Accept": "application/json"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "operational"
    assert "primary_model" in data
    assert data["primary_model"] == "tfidf_svm"
    assert "POST /api/resumes/upload" in data["endpoints"]
    assert "POST /api/resumes/<resume_id>/svm-matches" in data["endpoints"]


def test_dashboard_and_ui_aliases(client):
    """Explicit UI routes render index.html."""
    for route in ["/dashboard", "/app", "/ui"]:
        response = client.get(route)
        assert response.status_code == 200
        assert b"ResumeMatch AI" in response.data


def test_api_stats_endpoint(client):
    """GET /api/stats returns counts and model metadata."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "total_resumes" in data
    assert "total_jobs" in data
    assert "total_matches_evaluated" in data
    assert "primary_model" in data
    assert data["primary_model"] == "tfidf_svm"


def test_api_resumes_list_endpoint(client):
    """GET /api/resumes returns list of resumes with parsed fields."""
    response = client.get("/api/resumes")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "resumes" in data
    assert "total_resumes" in data
    assert isinstance(data["resumes"], list)
    if data["resumes"]:
        first = data["resumes"][0]
        assert "resume_id" in first
        assert "skills" in first


def test_api_jobs_list_endpoint(client):
    """GET /api/jobs returns benchmark jobs list."""
    response = client.get("/api/jobs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "jobs" in data
    assert "total_jobs" in data
    assert isinstance(data["jobs"], list)
    if data["jobs"]:
        first = data["jobs"][0]
        assert "job_id" in first
        assert "job_title" in first
        assert "skills" in first or "required_skills" in first


def test_api_single_job_endpoint(client):
    """GET /api/jobs/<id> returns specific job or 404."""
    response = client.get("/api/jobs/JOB_01")
    if response.status_code == 200:
        data = response.get_json()
        assert data["success"] is True
        job = data.get("job", data)
        assert job["job_id"] == "JOB_01"
        assert "job_title" in job or "title" in job
    else:
        assert response.status_code == 404


def test_api_matches_history_endpoint(client):
    """GET /api/matches/history returns evaluation history array."""
    response = client.get("/api/matches/history")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "history" in data
    assert "total_records" in data
    assert isinstance(data["history"], list)
