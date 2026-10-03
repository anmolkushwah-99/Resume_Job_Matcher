"""
Unit Tests for Experience Extractor.
Tests explicit duration detection, months-to-years conversion, fresher detection, and role parsing.
"""

from src.preprocessing.experience_extractor import extract_experience


class TestExperienceExtractor:
    """Test suite for professional experience extraction."""

    def test_explicit_two_years(self):
        """Verify '2 years of experience' maps to 2.0."""
        text = "Software Engineer with 2 years of experience in backend development."
        exp = extract_experience(text)
        assert exp["experience_years"] == 2.0
        assert exp["is_fresher"] is False

    def test_plus_years_pattern(self):
        """Verify '3+ years experience' maps to 3.0."""
        text = "Senior Python Developer with 3+ years experience in Flask and AWS."
        exp = extract_experience(text)
        assert exp["experience_years"] == 3.0

    def test_decimal_years(self):
        """Verify '1.5 years of experience' maps to 1.5."""
        text = "Machine Learning Engineer with 1.5 years of experience building NLP models."
        exp = extract_experience(text)
        assert exp["experience_years"] == 1.5

    def test_months_conversion(self):
        """Verify '6 months of experience' maps to 0.5 years."""
        text = "Completed 6 months of internship at ABC Technologies as a web developer."
        exp = extract_experience(text)
        assert exp["experience_years"] == 0.5

    def test_fresher_detection(self):
        """Verify entry-level/fresher candidates default to 0.0 years."""
        text = "Recent graduate and fresher seeking entry-level software engineer roles."
        exp = extract_experience(text)
        assert exp["experience_years"] == 0.0
        assert exp["is_fresher"] is True

    def test_role_extraction(self):
        """Verify role keywords are identified from text."""
        text = """
EXPERIENCE
Software Developer Intern at XYZ Corp
Machine Learning Engineer at DataCo
"""
        exp = extract_experience(text)
        role_titles = [r["title"] for r in exp["roles"]]
        assert any("Software Developer" in t for t in role_titles)
        assert any("Machine Learning Engineer" in t for t in role_titles)

    def test_empty_input(self):
        """Verify empty string returns 0.0 years."""
        exp = extract_experience("")
        assert exp["experience_years"] == 0.0
        assert exp["roles"] == []
