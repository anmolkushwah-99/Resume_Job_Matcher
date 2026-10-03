"""
Unit Tests for SVM Service Module.
Verifies single-pair prediction, MySQL persistence, idempotency,
model coexistence with Step 5, and multi-job ranking.
"""

import pytest
from src.models.svm_service import (
    predict_and_store_svm_match,
    predict_svm_matches_for_resume,
    get_svm_classifier,
)
from src.matching.match_service import calculate_and_store_match, ResumeNotFoundError, JobNotFoundError
from src.database.mysql_client import get_db_cursor, seed_benchmark_jobs, seed_benchmark_resumes


@pytest.fixture(scope="module", autouse=True)
def setup_benchmark_db():
    """Ensure benchmark data is seeded in MySQL for SVM service tests."""
    seed_benchmark_jobs()
    seed_benchmark_resumes()


class TestSVMService:

    def test_predict_and_store_svm_match_success(self):
        result = predict_and_store_svm_match(
            resume_id="RES001",
            job_id="JOB001",
            model_name="tfidf_svm",
            persist=True,
        )
        assert result["success"] is True
        assert result["resume_id"] == "RES001"
        assert result["job_id"] == "JOB001"
        assert result["model"] == "tfidf_svm"
        assert result["predicted_label"] in [0, 1]
        assert isinstance(result["is_match"], bool)
        assert isinstance(result["decision_score"], (float, int))
        assert isinstance(result["shared_skills"], list)

    def test_svm_and_step5_baseline_coexistence_in_db(self):
        # 1. Store Step 5 match
        s5_res = calculate_and_store_match("RES001", "JOB001", model_name="tfidf_cosine_baseline", persist=True)
        # 2. Store Step 6 SVM match
        s6_res = predict_and_store_svm_match("RES001", "JOB001", model_name="tfidf_svm", persist=True)

        # 3. Query DB to verify both records exist simultaneously
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT model_name, similarity_score, overall_score
                FROM matches
                WHERE resume_id = 'RES001' AND job_id = 'JOB001'
                """
            )
            rows = cur.fetchall()
            models_in_db = {r["model_name"] for r in rows}
            assert "tfidf_cosine_baseline" in models_in_db
            assert "tfidf_svm" in models_in_db

    def test_predict_svm_matches_for_resume_ranked(self):
        result = predict_svm_matches_for_resume(resume_id="RES001", limit=5, persist=False)
        assert result["success"] is True
        assert result["resume_id"] == "RES001"
        assert len(result["matches"]) <= 5

        # Check ranking order (descending decision score)
        scores = [m["decision_score"] for m in result["matches"]]
        assert scores == sorted(scores, reverse=True)

    def test_predict_nonexistent_resume_raises_error(self):
        with pytest.raises(ResumeNotFoundError):
            predict_and_store_svm_match("NONEXISTENT_RESUME_999", "JOB001")

    def test_predict_nonexistent_job_raises_error(self):
        with pytest.raises(JobNotFoundError):
            predict_and_store_svm_match("RES001", "NONEXISTENT_JOB_999")
