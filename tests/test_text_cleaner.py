"""
Unit Tests for Text Cleaner.
Verifies conservative cleaning, whitespace normalization, blank line reduction,
and preservation of technical terms, numbers, and punctuation.
"""

from src.preprocessing.text_cleaner import clean_resume_text


class TestTextCleaner:
    """Test suite for clean_resume_text function."""

    def test_whitespace_normalization(self):
        """Verify normalization of irregular horizontal whitespace and tabs."""
        raw_text = "John   Doe  \t \t Python    Developer   "
        expected = "John Doe Python Developer"
        assert clean_resume_text(raw_text) == expected

    def test_blank_lines_reduction(self):
        """Verify reduction of 3+ consecutive newlines down to 2 newlines."""
        raw_text = "John Doe\n\n\n\nPython Developer\n\n\n\nSkills: Python, SQL"
        expected = "John Doe\n\nPython Developer\n\nSkills: Python, SQL"
        assert clean_resume_text(raw_text) == expected

    def test_carriage_returns_and_control_chars(self):
        """Verify normalization of carriage returns (\\r\\n, \\r) and form feeds."""
        raw_text = "Line One\r\nLine Two\rLine Three\x0c\nLine Four\x00"
        cleaned = clean_resume_text(raw_text)
        assert "Line One" in cleaned
        assert "Line Two" in cleaned
        assert "Line Three" in cleaned
        assert "Line Four" in cleaned
        assert "\r" not in cleaned
        assert "\x0c" not in cleaned
        assert "\x00" not in cleaned

    def test_technical_terms_preservation(self):
        """Verify that technical symbols (+, #, ., -, /) are strictly preserved."""
        raw_text = (
            "Skills: Python, C++, C#, Java, .NET, Node.js, React.js, "
            "SQL, MongoDB, MySQL, TensorFlow, PyTorch, Scikit-learn, "
            "Flask, FastAPI, AWS, REST API, CI/CD, Next.js"
        )
        cleaned = clean_resume_text(raw_text)
        assert "C++" in cleaned
        assert "C#" in cleaned
        assert ".NET" in cleaned
        assert "Node.js" in cleaned
        assert "React.js" in cleaned
        assert "Scikit-learn" in cleaned
        assert "REST API" in cleaned
        assert "CI/CD" in cleaned
        assert "Next.js" in cleaned

    def test_preservation_of_numbers_and_punctuation(self):
        """Verify numbers, percentages, dates, and common punctuation are kept intact."""
        raw_text = "Graduated: 2026 | GPA: 3.8/4.0 | Experience: 2.5+ years (100% committed)"
        cleaned = clean_resume_text(raw_text)
        assert "2026" in cleaned
        assert "3.8/4.0" in cleaned
        assert "2.5+" in cleaned
        assert "(100% committed)" in cleaned
        assert "|" in cleaned

    def test_empty_and_falsy_inputs(self):
        """Verify handling of empty string, None, and whitespace-only text."""
        assert clean_resume_text("") == ""
        assert clean_resume_text(None) == ""
        assert clean_resume_text("   \n\n   \t  ") == ""
