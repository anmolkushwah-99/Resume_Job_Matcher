"""
Unit and functional tests for FinalMatchEngine (src/matching/final_match_engine.py).
"""

import pytest
from src.matching.final_match_engine import FinalMatchEngine, get_final_match_engine
from src.matching.match_service import ResumeNotFoundError, JobNotFoundError


@pytest.fixture(autouse=True)
def clean_engine_singleton():
    """Ensure a clean singleton instance for each test."""
    FinalMatchEngine.reset_instance()
    yield
    FinalMatchEngine.reset_instance()


def test_engine_initialization_and_singleton():
    """Verify FinalMatchEngine singleton creation and readiness."""
    engine1 = FinalMatchEngine.get_instance()
    engine2 = get_final_match_engine()
    assert engine1 is engine2
    assert engine1.is_ready is True
    assert engine1.model_name == "tfidf_svm"
    assert engine1.threshold == 0.0


def test_engine_match_one_success():
    """Verify single-pair matching produces valid decision scores and predictions."""
    engine = FinalMatchEngine.get_instance()
    res = engine.match_one(resume_id="RES001", job_id="JOB001", persist=False)

    assert res["success"] is True
    assert res["resume_id"] == "RES001"
    assert res["job_id"] == "JOB001"
    assert res["model_name"] == "tfidf_svm"
    assert isinstance(res["decision_score"], float)
    assert res["threshold"] == 0.0
    assert res["predicted_label"] in [0, 1]
    assert res["is_match"] == (res["decision_score"] >= 0.0)
    assert isinstance(res["matched_skills"], list)
    assert isinstance(res["missing_skills"], list)
    assert "explanation_note" in res


def test_engine_match_one_invalid_inputs():
    """Verify invalid resume or job IDs raise appropriate exceptions."""
    engine = FinalMatchEngine.get_instance()

    with pytest.raises(ValueError, match="resume_id"):
        engine.match_one(resume_id="", job_id="JOB001", persist=False)

    with pytest.raises(ValueError, match="job_id"):
        engine.match_one(resume_id="RES001", job_id="", persist=False)

    with pytest.raises(ResumeNotFoundError):
        engine.match_one(resume_id="NON_EXISTENT_RESUME_999", job_id="JOB001", persist=False)

    with pytest.raises(JobNotFoundError):
        engine.match_one(resume_id="RES001", job_id="NON_EXISTENT_JOB_999", persist=False)


def test_engine_match_many_deterministic_ranking():
    """Verify multi-job matching orders strictly by decision_score DESC, tie-breaker job_id ASC."""
    engine = FinalMatchEngine.get_instance()
    job_subset = ["JOB001", "JOB002", "JOB003", "JOB004", "JOB005"]

    res = engine.match_many(resume_id="RES001", job_ids=job_subset, persist=False)
    assert res["success"] is True
    assert res["total_jobs_evaluated"] == len(job_subset)
    results = res["results"]
    assert len(results) == len(job_subset)

    # Check rank ordering
    for i in range(len(results) - 1):
        score_curr = results[i]["decision_score"]
        score_next = results[i + 1]["decision_score"]
        assert score_curr >= score_next, f"Ordering violation: {score_curr} < {score_next}"
        if score_curr == score_next:
            assert results[i]["job_id"] < results[i + 1]["job_id"], "Tie-breaker violation"
        assert results[i]["rank"] == i + 1


def test_engine_match_many_top_k():
    """Verify top_k parameter correctly truncates ranked results."""
    engine = FinalMatchEngine.get_instance()
    job_subset = ["JOB001", "JOB002", "JOB003", "JOB004", "JOB005"]

    res = engine.match_many(resume_id="RES001", job_ids=job_subset, top_k=2, persist=False)
    assert len(res["results"]) == 2
    assert res["total_returned"] == 2
    assert res["total_jobs_evaluated"] == len(job_subset)
    assert res["results"][0]["rank"] == 1
    assert res["results"][1]["rank"] == 2


def test_engine_match_many_match_only():
    """Verify match_only filter retains only pairs with decision_score >= 0.0."""
    engine = FinalMatchEngine.get_instance()
    job_subset = ["JOB001", "JOB002", "JOB003", "JOB004", "JOB005"]

    res = engine.match_many(resume_id="RES001", job_ids=job_subset, match_only=True, persist=False)
    for item in res["results"]:
        assert item["is_match"] is True
        assert item["decision_score"] >= 0.0


def test_engine_match_batch_error_isolation():
    """Verify batch matching isolates failures for missing jobs while succeeding on valid jobs."""
    engine = FinalMatchEngine.get_instance()
    mixed_jobs = ["JOB001", "INVALID_JOB_XYZ", "JOB002"]

    res = engine.match_batch(resume_id="RES001", job_ids=mixed_jobs, persist=False)
    assert res["success"] is True
    assert res["total_submitted"] == 3
    assert res["successful_count"] == 2
    assert res["failed_count"] == 1
    assert len(res["failed_jobs"]) == 1
    assert res["failed_jobs"][0]["job_id"] == "INVALID_JOB_XYZ"
