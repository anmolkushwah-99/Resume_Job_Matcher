"""
Flask API Route Integration Tests.
Tests the /api/resumes/upload, /api/health, and root endpoints.
"""

import io
from unittest.mock import patch, MagicMock
import pytest
from app import create_app
from tests.pdf_test_utils import get_standard_synthetic_resume_pdf, generate_synthetic_pdf_bytes


@pytest.fixture
def client():
    """Create a Flask test client."""
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


class TestFlaskAPI:
    """Test suite for Flask API routes."""

    def test_root_endpoint(self, client):
        """Verify root endpoint returns system information and operational status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.get_json()
        assert "Step 9" in data["phase"]
        assert "POST /api/resumes/upload" in data["endpoints"]
        assert "POST /api/resumes/<resume_id>/process" in data["endpoints"]
        assert "POST /api/matches/calculate" in data["endpoints"]
        assert "POST /api/resumes/<resume_id>/matches" in data["endpoints"]
        assert "POST /api/matches/svm" in data["endpoints"]
        assert "POST /api/resumes/<resume_id>/svm-matches" in data["endpoints"]
        assert "POST /api/matches/ann" in data["endpoints"]
        assert "POST /api/resumes/<resume_id>/ann-matches" in data["endpoints"]
        assert "POST /api/matches/lstm" in data["endpoints"]
        assert "POST /api/resumes/<resume_id>/lstm-matches" in data["endpoints"]

    def test_health_endpoint(self, client):
        """Verify health check endpoint returns 200 and database status object."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert "database" in data

    def test_upload_missing_file_key(self, client):
        """Verify 400 when 'file' key is missing from multipart/form-data."""
        response = client.post("/api/resumes/upload", data={})
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "No file part in request" in data["error"]

    def test_upload_empty_filename(self, client):
        """Verify 400 when file part is submitted with empty filename."""
        data = {
            "file": (io.BytesIO(b"%PDF-1.4 test"), "")
        }
        response = client.post(
            "/api/resumes/upload",
            data=data,
            content_type="multipart/form-data"
        )
        assert response.status_code == 400
        json_data = response.get_json()
        assert json_data["success"] is False
        assert "empty" in json_data["error"].lower() or "missing" in json_data["error"].lower()

    def test_upload_unsupported_file_extension(self, client):
        """Verify 400 when non-PDF file (e.g. .docx, .txt) is uploaded."""
        data = {
            "file": (io.BytesIO(b"Candidate resume content"), "resume.docx")
        }
        response = client.post(
            "/api/resumes/upload",
            data=data,
            content_type="multipart/form-data"
        )
        assert response.status_code == 400
        json_data = response.get_json()
        assert json_data["success"] is False
        assert "Only PDF files are supported" in json_data["error"]

    def test_upload_empty_pdf_file(self, client):
        """Verify 400 when 0-byte PDF is uploaded."""
        data = {
            "file": (io.BytesIO(b""), "empty_resume.pdf")
        }
        response = client.post(
            "/api/resumes/upload",
            data=data,
            content_type="multipart/form-data"
        )
        assert response.status_code == 400
        json_data = response.get_json()
        assert json_data["success"] is False
        assert "empty" in json_data["error"].lower()

    def test_upload_pdf_with_no_text_returns_422(self, client):
        """Verify 422 when PDF has no extractable text."""
        blank_pdf_bytes = generate_synthetic_pdf_bytes([[]])
        data = {
            "file": (io.BytesIO(blank_pdf_bytes), "blank_scanned.pdf")
        }
        response = client.post(
            "/api/resumes/upload",
            data=data,
            content_type="multipart/form-data"
        )
        assert response.status_code == 422
        json_data = response.get_json()
        assert json_data["success"] is False
        assert "does not contain any extractable text" in json_data["error"]

    def test_upload_valid_resume_pdf_success(self, client):
        """Verify 201 Created and proper JSON response for valid PDF upload."""
        pdf_bytes = get_standard_synthetic_resume_pdf()
        data = {
            "file": (io.BytesIO(pdf_bytes), "john_doe_resume.pdf")
        }
        response = client.post(
            "/api/resumes/upload",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 201
        json_data = response.get_json()
        assert json_data["success"] is True
        assert json_data["message"] == "Resume uploaded and processed successfully"
        assert "resume_id" in json_data
        assert json_data["filename"] == "john_doe_resume.pdf"
        assert json_data["text_length"] > 0
        # Personal extracted text MUST NOT be returned in API response
        assert "extracted_text" not in json_data
        assert "text" not in json_data

    def test_upload_oversize_file_returns_413(self, client):
        """Verify 413 Payload Too Large when file exceeds 10MB."""
        large_bytes = b"0" * (10 * 1024 * 1024 + 100)
        data = {
            "file": (io.BytesIO(large_bytes), "huge_resume.pdf")
        }
        response = client.post(
            "/api/resumes/upload",
            data=data,
            content_type="multipart/form-data"
        )
        assert response.status_code == 413
        json_data = response.get_json()
        assert json_data["success"] is False
        assert "10 MB" in json_data["error"]

    def test_process_resume_endpoint_success(self, client):
        """Verify 200 OK and structured features from POST /api/resumes/<resume_id>/process."""
        # 1. Upload a valid resume first
        pdf_bytes = get_standard_synthetic_resume_pdf()
        upload_resp = client.post(
            "/api/resumes/upload",
            data={"file": (io.BytesIO(pdf_bytes), "api_test_candidate.pdf")},
            content_type="multipart/form-data"
        )
        assert upload_resp.status_code == 201
        resume_id = upload_resp.get_json()["resume_id"]

        # 2. Process NLP features
        process_resp = client.post(f"/api/resumes/{resume_id}/process")
        assert process_resp.status_code == 200
        data = process_resp.get_json()
        assert data["success"] is True
        assert data["resume_id"] == resume_id
        assert data["skills_detected"] > 0
        assert "Python" in data["skills"]
        assert data["degree"] == "B.Tech"
        assert data["field_of_study"] == "Computer Science"
        assert data["experience_years"] == 2.0
        # Personal extracted text MUST NOT be returned in API response
        assert "extracted_text" not in data
        assert "text" not in data

    def test_process_nonexistent_resume_returns_404(self, client):
        """Verify 404 Not Found when processing a non-existent resume ID."""
        response = client.post("/api/resumes/00000000-0000-0000-0000-000000000000/process")
        assert response.status_code == 404
        data = response.get_json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    def test_calculate_match_endpoint_success(self, client):
        """Verify 200 OK and valid match response from POST /api/matches/calculate."""
        payload = {
            "resume_id": "RES001",
            "job_id": "JOB001"
        }
        response = client.post("/api/matches/calculate", json=payload)
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["resume_id"] == "RES001"
        assert data["job_id"] == "JOB001"
        assert data["model"] == "tfidf_cosine_baseline"
        assert 0.0 <= data["similarity_score"] <= 1.0
        assert data["similarity_percentage"] == round(data["similarity_score"] * 100, 2)
        assert isinstance(data["shared_skills"], list)
        # Verify privacy: no raw text returned
        assert "extracted_text" not in data
        assert "job_description" not in data

    def test_calculate_match_endpoint_missing_body(self, client):
        """Verify 400 Bad Request when request body or parameters are missing."""
        resp1 = client.post("/api/matches/calculate", json={})
        assert resp1.status_code == 400

        resp2 = client.post("/api/matches/calculate", json={"resume_id": "RES001"})
        assert resp2.status_code == 400

        resp3 = client.post("/api/matches/calculate", json={"job_id": "JOB001"})
        assert resp3.status_code == 400

    def test_calculate_match_endpoint_nonexistent_resume_returns_404(self, client):
        """Verify 404 Not Found when resume does not exist."""
        payload = {
            "resume_id": "NONEXISTENT_RESUME_9999",
            "job_id": "JOB001"
        }
        response = client.post("/api/matches/calculate", json=payload)
        assert response.status_code == 404
        data = response.get_json()
        assert data["success"] is False

    def test_get_resume_all_matches_endpoint(self, client):
        """Verify 200 OK and ranked matches list from POST /api/resumes/<resume_id>/matches."""
        response = client.post("/api/resumes/RES001/matches?limit=5")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["resume_id"] == "RES001"
        assert len(data["matches"]) <= 5
        scores = [m["similarity_score"] for m in data["matches"]]
        assert scores == sorted(scores, reverse=True)

    def test_get_resume_all_matches_nonexistent_returns_404(self, client):
        """Verify 404 Not Found when resume ID does not exist."""
        response = client.post("/api/resumes/NONEXISTENT_RESUME_9999/matches")
        assert response.status_code == 404

    def test_calculate_svm_match_endpoint_success(self, client):
        """Verify 200 OK and valid prediction response from POST /api/matches/svm."""
        payload = {
            "resume_id": "RES001",
            "job_id": "JOB001"
        }
        response = client.post("/api/matches/svm", json=payload)
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["resume_id"] == "RES001"
        assert data["job_id"] == "JOB001"
        assert data["model"] == "tfidf_svm"
        assert data["predicted_label"] in [0, 1]
        assert isinstance(data["is_match"], bool)
        assert isinstance(data["decision_score"], (float, int))
        assert isinstance(data["shared_skills"], list)
        # Privacy check: no raw text
        assert "extracted_text" not in data
        assert "job_description" not in data

    def test_calculate_svm_match_endpoint_missing_body(self, client):
        """Verify 400 Bad Request when parameters are missing for SVM match."""
        resp1 = client.post("/api/matches/svm", json={})
        assert resp1.status_code == 400

        resp2 = client.post("/api/matches/svm", json={"resume_id": "RES001"})
        assert resp2.status_code == 400

        resp3 = client.post("/api/matches/svm", json={"job_id": "JOB001"})
        assert resp3.status_code == 400

    def test_calculate_svm_match_endpoint_nonexistent_resume_returns_404(self, client):
        """Verify 404 Not Found when resume does not exist."""
        payload = {
            "resume_id": "NONEXISTENT_RESUME_9999",
            "job_id": "JOB001"
        }
        response = client.post("/api/matches/svm", json=payload)
        assert response.status_code == 404
        data = response.get_json()
        assert data["success"] is False

    def test_get_resume_all_svm_matches_endpoint(self, client):
        """Verify 200 OK and ranked matches list from POST /api/resumes/<resume_id>/svm-matches."""
        response = client.post("/api/resumes/RES001/svm-matches?limit=5")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["resume_id"] == "RES001"
        assert data["model"] == "tfidf_svm"
        assert len(data["matches"]) <= 5
        scores = [m["decision_score"] for m in data["matches"]]
        assert scores == sorted(scores, reverse=True)

    def test_get_resume_all_svm_matches_nonexistent_returns_404(self, client):
        """Verify 404 Not Found when resume ID does not exist for SVM matches."""
        response = client.post("/api/resumes/NONEXISTENT_RESUME_9999/svm-matches")
        assert response.status_code == 404



