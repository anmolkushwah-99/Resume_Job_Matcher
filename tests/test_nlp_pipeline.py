"""
Integration Tests for Central NLP Preprocessing Pipeline.
Tests end-to-end extraction on complete synthetic resumes, minimal resumes, and artifact generation.
"""

import json
from pathlib import Path
import pytest
from src.preprocessing.nlp_pipeline import process_resume_text, get_processed_nlp_dir


class TestNLPPipeline:
    """Test suite for central NLP pipeline."""

    def test_complete_realistic_synthetic_resume(self):
        """Verify end-to-end feature extraction on a realistic synthetic resume."""
        resume_text = """
John Doe

SUMMARY
Software developer interested in machine learning and scalable web services.

SKILLS
Python, Java, SQL, Machine Learning, Natural Language Processing, TensorFlow,
Scikit-learn, Git, MySQL, Flask, Docker

EDUCATION
B.Tech in Computer Science and Engineering
ABC University

EXPERIENCE
Software Developer Intern
XYZ Technologies
1.5 years of experience

PROJECTS
Resume Screening System: End-to-end ML matching engine

CERTIFICATIONS
AWS Certified Solutions Architect Associate
"""
        result = process_resume_text(resume_text, resume_id="TEST_SYNTHETIC_001")

        # 1. Check basic metadata
        assert result["resume_id"] == "TEST_SYNTHETIC_001"
        assert result["tokens_count"] > 0
        assert len(result["lemmas"]) > 0

        # 2. Check sections
        assert "Software developer" in result["sections"]["summary"]
        assert "B.Tech in Computer Science" in result["sections"]["education"]
        assert "1.5 years" in result["sections"]["experience"]
        assert "Resume Screening System" in result["sections"]["projects"]
        assert "AWS Certified" in result["sections"]["certifications"]

        # 3. Check skills
        skill_names = result["skill_names"]
        assert "Python" in skill_names
        assert "Java" in skill_names
        assert "SQL" in skill_names
        assert "Machine Learning" in skill_names
        assert "Natural Language Processing" in skill_names
        assert "TensorFlow" in skill_names
        assert "Scikit-learn" in skill_names
        assert "Git" in skill_names
        assert "MySQL" in skill_names
        assert "Flask" in skill_names
        assert "Docker" in skill_names

        # 4. Check education
        assert result["degree"] == "B.Tech"
        assert result["field_of_study"] == "Computer Science"

        # 5. Check experience
        assert result["experience_years"] == 1.5

        # 6. Verify JSON artifact exists
        artifact_file = get_processed_nlp_dir() / "TEST_SYNTHETIC_001.json"
        assert artifact_file.exists()
        with open(artifact_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
            assert saved_data["resume_id"] == "TEST_SYNTHETIC_001"
            assert "Python" in saved_data["skill_names"]

        # Cleanup artifact
        artifact_file.unlink(missing_ok=True)

    def test_minimal_resume_handling(self):
        """Verify minimal resume text extracts whatever is present without crashing."""
        minimal_text = "Python developer with 3 years experience."
        result = process_resume_text(minimal_text)

        assert "Python" in result["skill_names"]
        assert result["experience_years"] == 3.0
        assert result["degree"] is None

    def test_empty_resume_raises_value_error(self):
        """Verify empty resume text raises ValueError."""
        with pytest.raises(ValueError) as exc:
            process_resume_text("")
        assert "empty or invalid" in str(exc.value)

    def test_pipeline_repeatability(self):
        """Verify calling pipeline multiple times produces identical output."""
        text = "Frontend Engineer with React, TypeScript, and 2 years of experience."
        res1 = process_resume_text(text)
        res2 = process_resume_text(text)

        assert res1["skill_names"] == res2["skill_names"]
        assert res1["experience_years"] == res2["experience_years"]
        assert res1["tokens"] == res2["tokens"]
