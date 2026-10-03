"""
Unit and Integration Tests for LSTM Prediction Service (Step 8).
Tests single-pair prediction, resume-level ranking, MySQL persistence,
and coexistence with Step 5, Step 6, and Step 7 match records.
"""

import pytest
from src.models.lstm_service import (
    predict_and_store_lstm_match,
    predict_lstm_matches_for_resume,
    get_lstm_classifier,
)
from src.database.mysql_client import get_db_cursor


class TestLSTMService:
    """Tests LSTM service inference and persistence."""

    def test_predict_and_store_lstm_match_success(self):
        result = predict_and_store_lstm_match(
            resume_id="RES001",
            job_id="JOB001",
            model_name="embedding_lstm",
            persist=True
        )

        assert result["success"] is True
        assert result["resume_id"] == "RES001"
        assert result["job_id"] == "JOB001"
        assert result["model"] == "embedding_lstm"
        assert result["predicted_label"] in [0, 1]
        assert isinstance(result["is_match"], bool)
        assert 0.0 <= result["lstm_score"] <= 1.0
        assert "shared_skills" in result

    def test_lstm_coexistence_with_previous_steps_in_db(self):
        """Verifies that tfidf_cosine_baseline, tfidf_svm, tfidf_ann, and embedding_lstm coexist in matches table."""
        # Ensure Step 5 record
        from src.matching.match_service import calculate_and_store_match
        calculate_and_store_match("RES001", "JOB001", model_name="tfidf_cosine_baseline", persist=True)

        # Ensure Step 6 record
        from src.models.svm_service import predict_and_store_svm_match
        predict_and_store_svm_match("RES001", "JOB001", model_name="tfidf_svm", persist=True)

        # Ensure Step 7 record
        from src.models.ann_service import predict_and_store_ann_match
        predict_and_store_ann_match("RES001", "JOB001", model_name="tfidf_ann", persist=True)

        # Ensure Step 8 record
        predict_and_store_lstm_match("RES001", "JOB001", model_name="embedding_lstm", persist=True)

        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT model_name, overall_score, similarity_score
                FROM matches
                WHERE resume_id = %s AND job_id = %s
                ORDER BY model_name ASC
                """,
                ("RES001", "JOB001")
            )
            rows = cur.fetchall()
            models_in_db = {r["model_name"] for r in rows}

            assert "tfidf_cosine_baseline" in models_in_db
            assert "tfidf_svm" in models_in_db
            assert "tfidf_ann" in models_in_db
            assert "embedding_lstm" in models_in_db

    def test_predict_lstm_matches_for_resume_ranked(self):
        result = predict_lstm_matches_for_resume(
            resume_id="RES001",
            model_name="embedding_lstm",
            limit=5,
            persist=False
        )

        assert result["success"] is True
        assert result["resume_id"] == "RES001"
        assert result["model"] == "embedding_lstm"
        assert len(result["matches"]) <= 5

        # Check ranking order (descending by lstm_score)
        scores = [m["lstm_score"] for m in result["matches"]]
        assert scores == sorted(scores, reverse=True)

    def test_predict_nonexistent_resume_raises_error(self):
        with pytest.raises(Exception):
            predict_and_store_lstm_match("NONEXISTENT_RES", "JOB001", persist=False)

    def test_predict_nonexistent_job_raises_error(self):
        with pytest.raises(Exception):
            predict_and_store_lstm_match("RES001", "NONEXISTENT_JOB", persist=False)
