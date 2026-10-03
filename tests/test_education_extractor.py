"""
Unit Tests for Education Extractor.
Tests degree normalization, field of study extraction, institution parsing, and missing education.
"""

from src.preprocessing.education_extractor import extract_education


class TestEducationExtractor:
    """Test suite for education extraction."""

    def test_btech_computer_science(self):
        """Verify B.Tech and Computer Science detection."""
        text = "Education: B.Tech in Computer Science and Engineering from ABC University."
        edu = extract_education(text)

        assert edu is not None
        assert edu["degree"] == "B.Tech"
        assert edu["field_of_study"] == "Computer Science"
        assert "ABC University" in edu["institution"]

    def test_be_electronics(self):
        """Verify B.E. and Electronics detection."""
        text = "Graduated with B.E. in Electronics and Communication Engineering from Delhi College."
        edu = extract_education(text)

        assert edu is not None
        assert edu["degree"] == "B.E."
        assert edu["field_of_study"] == "Electronics and Communication"

    def test_mtech_software_engineering(self):
        """Verify M.Tech and Software Engineering detection."""
        text = "Academics: M.Tech in Software Engineering."
        edu = extract_education(text)

        assert edu is not None
        assert edu["degree"] == "M.Tech"
        assert edu["field_of_study"] == "Software Engineering"

    def test_mca_extraction(self):
        """Verify Master of Computer Applications is normalized to MCA."""
        text = "Completed Master of Computer Applications (MCA) from National Institute of Technology."
        edu = extract_education(text)

        assert edu is not None
        assert edu["degree"] == "MCA"
        assert "National Institute of Technology" in edu["institution"]

    def test_bsc_statistics(self):
        """Verify B.Sc and Statistics detection."""
        text = "Bachelor of Science in Statistics."
        edu = extract_education(text)

        assert edu is not None
        assert edu["degree"] == "B.Sc"
        assert edu["field_of_study"] == "Statistics"

    def test_missing_education_returns_none(self):
        """Verify text without education mentions returns None."""
        text = "Experienced software developer with 4 years building REST APIs."
        edu = extract_education(text)
        assert edu is None

    def test_empty_input(self):
        """Verify empty string returns None."""
        assert extract_education("") is None
        assert extract_education(None) is None
