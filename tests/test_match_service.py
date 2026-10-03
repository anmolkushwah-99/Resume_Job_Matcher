"""
Unit Tests for Match Service.
Verifies resume-job match calculation, MySQL persistence, idempotency,
error handling, and ranked match generation.
"""

import pytest
from src.matching.match_service import (
    calculate_and_store_match,
    calculate_matches_for_resume,
    ResumeNotFoundError,
    JobNotFoundError,
)
from src.database.mysql_client import get_db_cursor, seed_benchmark_jobs, seed_benchmark_resumes


@pytest.fixture(scope="module", autouse=True)
def setup_benchmark_db():
    """Ensure benchmark data is seeded in MySQL for service tests."""
    seed_benchmark_jobs()
    seed_benchmark_resumes()


class TestMatchService:

    def test_calculate_and_store_match_success(self):
        result = calculate_and_store_match(
            resume_id="RES001",
            job_id="JOB001",
            model_name="tfidf_cosine_baseline",
            persist=True,
        )
        assert result["success"] is True
        assert result["resume_id"] == "RES001"
        assert result["job_id"] == "JOB001"
        assert result["model"] == "tfidf_cosine_baseline"
        assert 0.0 <= result["similarity_score"] <= 1.0
        assert result["similarity_percentage"] == round(result["similarity_score"] * 100, 2)
        assert isinstance(result["shared_skills"], list)

    def test_idempotent_match_persistence(self):
        # Calculate twice
        res1 = calculate_and_store_match("RES001", "JOB001", persist=True)
        res2 = calculate_and_store_match("RES001", "JOB001", persist=True)
        assert res1["similarity_score"] == res2["similarity_score"]

        # Verify only one row exists for this pair and model in DB
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS cnt FROM matches
                WHERE resume_id = 'RES001' AND job_id = 'JOB001' AND model_name = 'tfidf_cosine_baseline'
                """
            )
            count = cur.fetchone()["cnt"]
            assert count == 1

    def test_calculate_match_nonexistent_resume_raises_error(self):
        with pytest.raises(ResumeNotFoundError):
            calculate_and_store_match("NONEXISTENT_RESUME_999", "JOB001")

    def test_calculate_match_nonexistent_job_raises_error(self):
        with pytest.raises(JobNotFoundError):
            calculate_and_store_match("RES001", "NONEXISTENT_JOB_999")

    def test_calculate_match_empty_inputs_raises_value_error(self):
        with pytest.raises(ValueError):
            calculate_and_store_match("", "JOB001")
        with pytest.raises(ValueError):
            calculate_and_store_match("RES001", "")

    def test_calculate_matches_for_resume_ranked(self):
        result = calculate_matches_for_resume(resume_id="RES001", limit=5, persist=False)
        assert result["success"] is True
        assert result["resume_id"] == "RES001"
        assert len(result["matches"]) <= 5

        # Check ranking order (descending similarity)
        scores = [m["similarity_score"] for m in result["matches"]]
        assert scores == sorted(scores, reverse=True)
