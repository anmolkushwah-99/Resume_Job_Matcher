"""
Unit and Integration Tests for NLP Service & MySQL Persistence.
Tests MySQL database transactional updates, resume_skills junction records, and idempotency.
"""

import uuid
import pytest
from src.database.mysql_client import get_db_cursor
from src.preprocessing.nlp_service import (
    process_resume_by_id,
    ResumeNotFoundError,
    ResumeProcessingError,
)


class TestNLPService:
    """Test suite for NLP service and MySQL integration."""

    def test_process_resume_by_id_e2e(self):
        """Verify fetching resume from MySQL, extracting features, and updating tables."""
        test_id = str(uuid.uuid4())
        test_text = """
Jane Doe

SUMMARY
Machine Learning Engineer with 2.5 years of experience in Python, PyTorch, and NLP.

EDUCATION
B.Tech in Computer Science from State University

EXPERIENCE
Machine Learning Engineer at AI Corp
2.5 years of experience
"""
        # 1. Insert initial resume record into MySQL
        with get_db_cursor(commit=True) as cur:
            cur.execute("""
                INSERT INTO resumes (id, filename, file_path, extracted_text)
                VALUES (%s, %s, %s, %s);
            """, (test_id, "jane_doe.pdf", f"{test_id}/jane_doe.pdf", test_text))

        try:
            # 2. Run NLP Service
            res = process_resume_by_id(test_id)

            assert res["success"] is True
            assert res["resume_id"] == test_id
            assert res["skills_detected"] > 0
            assert "Python" in res["skills"]
            assert "PyTorch" in res["skills"]
            assert "Natural Language Processing" in res["skills"]
            assert res["degree"] == "B.Tech"
            assert res["field_of_study"] == "Computer Science"
            assert res["experience_years"] == 2.5

            # 3. Verify MySQL database state
            with get_db_cursor(commit=False) as cur:
                cur.execute("SELECT degree, field_of_study, experience_years FROM resumes WHERE id = %s;", (test_id,))
                db_row = cur.fetchone()
                assert db_row["degree"] == "B.Tech"
                assert db_row["field_of_study"] == "Computer Science"
                assert float(db_row["experience_years"]) == 2.5

                cur.execute("SELECT COUNT(*) AS cnt FROM resume_skills WHERE resume_id = %s;", (test_id,))
                skill_count = cur.fetchone()["cnt"]
                assert skill_count == res["skills_detected"]

            # 4. Verify Idempotency (Processing again does not duplicate rows)
            res2 = process_resume_by_id(test_id)
            with get_db_cursor(commit=False) as cur:
                cur.execute("SELECT COUNT(*) AS cnt FROM resume_skills WHERE resume_id = %s;", (test_id,))
                skill_count2 = cur.fetchone()["cnt"]
                assert skill_count2 == skill_count

        finally:
            # Cleanup test record
            with get_db_cursor(commit=True) as cur:
                cur.execute("DELETE FROM resumes WHERE id = %s;", (test_id,))

    def test_nonexistent_resume_raises_not_found(self):
        """Verify non-existent resume ID raises ResumeNotFoundError."""
        fake_id = str(uuid.uuid4())
        with pytest.raises(ResumeNotFoundError) as exc:
            process_resume_by_id(fake_id)
        assert "not found" in str(exc.value).lower()

    def test_empty_text_resume_raises_processing_error(self):
        """Verify resume record with empty text raises ResumeProcessingError."""
        test_id = str(uuid.uuid4())
        with get_db_cursor(commit=True) as cur:
            cur.execute("""
                INSERT INTO resumes (id, filename, file_path, extracted_text)
                VALUES (%s, %s, %s, %s);
            """, (test_id, "empty_text.pdf", f"{test_id}/empty_text.pdf", "   "))

        try:
            with pytest.raises(ResumeProcessingError) as exc:
                process_resume_by_id(test_id)
            assert "no extracted text" in str(exc.value).lower()
        finally:
            with get_db_cursor(commit=True) as cur:
                cur.execute("DELETE FROM resumes WHERE id = %s;", (test_id,))
