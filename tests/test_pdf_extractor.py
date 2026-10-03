"""
Unit Tests for PDF Text Extractor.
Tests single-page, multi-page, empty-page, corrupted PDFs, and technical term extraction.
"""

import io
import tempfile
from pathlib import Path
import pytest
from src.extraction.pdf_extractor import extract_text_from_pdf, PDFExtractionError
from tests.pdf_test_utils import generate_synthetic_pdf_bytes, get_standard_synthetic_resume_pdf


class TestPDFExtractor:
    """Test suite for extract_text_from_pdf function."""

    def test_extract_valid_single_page(self):
        """Verify extraction from a single-page valid PDF."""
        pdf_bytes = generate_synthetic_pdf_bytes([
            ["John Doe", "Senior Python Developer", "Education: B.Tech in CS"]
        ])
        text = extract_text_from_pdf(pdf_bytes)
        assert "John Doe" in text
        assert "Senior Python Developer" in text
        assert "Education: B.Tech in CS" in text

    def test_extract_multiple_pages(self):
        """Verify extraction across multiple pages preserving order."""
        page1 = ["Page One Header", "Candidate Name: Jane Smith"]
        page2 = ["Page Two Header", "Experience: 5 years in Data Science"]
        page3 = ["Page Three Header", "Skills: Python, TensorFlow, PyTorch"]

        pdf_bytes = generate_synthetic_pdf_bytes([page1, page2, page3])
        text = extract_text_from_pdf(pdf_bytes)

        assert "Page One Header" in text
        assert "Jane Smith" in text
        assert "Page Two Header" in text
        assert "Data Science" in text
        assert "Page Three Header" in text
        assert "PyTorch" in text

        # Verify page order
        pos1 = text.find("Page One Header")
        pos2 = text.find("Page Two Header")
        pos3 = text.find("Page Three Header")
        assert pos1 < pos2 < pos3

    def test_extract_with_empty_page(self):
        """Verify handling of PDF containing an empty page without crashing."""
        page1 = ["First Page Content"]
        page2 = []  # Empty page
        page3 = ["Third Page Content"]

        pdf_bytes = generate_synthetic_pdf_bytes([page1, page2, page3])
        text = extract_text_from_pdf(pdf_bytes)

        assert "First Page Content" in text
        assert "Third Page Content" in text

    def test_technical_terms_preservation(self):
        """Verify that technical terms with symbols are cleanly extracted."""
        tech_lines = [
            "Technical Skills:",
            "Python, C++, C#, Java, .NET, Node.js, React.js, SQL",
            "MongoDB, MySQL, TensorFlow, PyTorch, Scikit-learn",
            "Flask, FastAPI, AWS, REST API, Git",
        ]
        pdf_bytes = generate_synthetic_pdf_bytes([tech_lines])
        text = extract_text_from_pdf(pdf_bytes)

        terms_to_check = [
            "Python", "C++", "C#", "Java", ".NET", "Node.js", "React.js",
            "SQL", "MongoDB", "MySQL", "TensorFlow", "PyTorch", "Scikit-learn",
            "Flask", "FastAPI", "AWS", "REST API", "Git"
        ]
        for term in terms_to_check:
            assert term in text, f"Expected technical term '{term}' to be preserved in extracted text."

    def test_extract_from_file_path(self):
        """Verify extraction when a file path (str/Path) is provided."""
        pdf_bytes = get_standard_synthetic_resume_pdf()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(pdf_bytes)

        try:
            text = extract_text_from_pdf(tmp_path)
            assert "John Doe" in text
            assert "Python Developer" in text
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_extract_from_bytes_io(self):
        """Verify extraction from io.BytesIO stream."""
        pdf_bytes = get_standard_synthetic_resume_pdf()
        stream = io.BytesIO(pdf_bytes)
        text = extract_text_from_pdf(stream)
        assert "John Doe" in text
        assert "B.Tech in Computer Science" in text

    def test_invalid_corrupted_pdf_raises_error(self):
        """Verify that corrupted bytes raise PDFExtractionError."""
        corrupt_bytes = b"This is not a PDF file at all, just plain corrupted bytes."
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_from_pdf(corrupt_bytes)
        assert "Invalid or corrupted PDF file" in str(exc_info.value) or "error" in str(exc_info.value).lower()

    def test_empty_bytes_raises_error(self):
        """Verify that empty byte input raises PDFExtractionError."""
        with pytest.raises(PDFExtractionError):
            extract_text_from_pdf(b"")

    def test_none_input_raises_error(self):
        """Verify that None input raises PDFExtractionError."""
        with pytest.raises(PDFExtractionError):
            extract_text_from_pdf(None)

    def test_nonexistent_file_path_raises_error(self):
        """Verify that non-existent file path raises PDFExtractionError."""
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_from_pdf("non_existent_file_path_12345.pdf")
        assert "not found" in str(exc_info.value).lower()
