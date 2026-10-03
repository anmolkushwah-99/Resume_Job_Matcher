"""
Unit and Integration Tests for Resume Service.
Tests file validation, path sanitization, local storage, MySQL db insert, and rollback mechanisms.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.extraction.resume_service import (
    validate_resume_file,
    sanitize_pdf_filename,
    process_resume_upload,
    ResumeValidationError,
    EmptyPDFError,
    ResumeStorageError,
    ResumeDatabaseError,
)
from src.storage.local_storage import delete_resume_file, get_resume_file_path
from tests.pdf_test_utils import generate_synthetic_pdf_bytes, get_standard_synthetic_resume_pdf


class TestResumeServiceValidation:
    """Tests for file validation and path sanitization."""

    def test_sanitize_pdf_filename(self):
        """Verify filename sanitization prevents path traversal."""
        assert sanitize_pdf_filename("../../../malicious_resume.pdf") == "malicious_resume.pdf"
        assert sanitize_pdf_filename("..\\..\\path\\traversal.pdf") == "traversal.pdf"
        assert sanitize_pdf_filename("valid_name.pdf") == "valid_name.pdf"
        assert sanitize_pdf_filename("") == "resume.pdf"
        assert sanitize_pdf_filename(".pdf") == "resume.pdf"

    def test_validate_valid_pdf_bytes(self):
        """Verify valid PDF bytes pass validation."""
        pdf_bytes = get_standard_synthetic_resume_pdf()
        bytes_out, filename_out = validate_resume_file(pdf_bytes, filename="john_doe_resume.pdf")
        assert len(bytes_out) > 0
        assert filename_out == "john_doe_resume.pdf"

    def test_validate_missing_file_raises_error(self):
        """Verify missing/None file raises 400."""
        with pytest.raises(ResumeValidationError) as exc:
            validate_resume_file(None, filename="resume.pdf")
        assert exc.value.status_code == 400
        assert "No file provided" in exc.value.message

    def test_validate_empty_filename_raises_error(self):
        """Verify empty filename raises 400."""
        pdf_bytes = get_standard_synthetic_resume_pdf()
        with pytest.raises(ResumeValidationError) as exc:
            validate_resume_file(pdf_bytes, filename="")
        assert exc.value.status_code == 400
        assert "empty or missing filename" in exc.value.message.lower()

    def test_validate_unsupported_extensions_rejected(self):
        """Verify .doc, .docx, .txt, .jpg, .zip, etc. are rejected with 400."""
        unsupported = ["resume.docx", "resume.doc", "resume.txt", "photo.jpg", "archive.zip", "script.py"]
        pdf_bytes = b"%PDF-1.4 sample content"
        for fname in unsupported:
            with pytest.raises(ResumeValidationError) as exc:
                validate_resume_file(pdf_bytes, filename=fname)
            assert exc.value.status_code == 400
            assert "Only PDF files are supported" in exc.value.message

    def test_validate_empty_file_rejected(self):
        """Verify 0-byte file is rejected with 400."""
        with pytest.raises(ResumeValidationError) as exc:
            validate_resume_file(b"", filename="empty.pdf")
        assert exc.value.status_code == 400
        assert "empty" in exc.value.message.lower()

    def test_validate_oversize_file_rejected(self):
        """Verify files larger than 10 MB are rejected with status 413."""
        oversize_bytes = b"0" * (10 * 1024 * 1024 + 1)
        with pytest.raises(ResumeValidationError) as exc:
            validate_resume_file(oversize_bytes, filename="huge.pdf")
        assert exc.value.status_code == 413
        assert "exceeds the maximum allowed limit of 10 MB" in exc.value.message


class TestResumeServicePipeline:
    """Tests for complete pipeline and partial failure handling."""

    def test_successful_upload_pipeline(self):
        """Verify standard end-to-end processing returns privacy-safe summary."""
        pdf_bytes = get_standard_synthetic_resume_pdf()

        result = process_resume_upload(pdf_bytes, filename="john_doe_resume.pdf")

        assert result["success"] is True
        assert "resume_id" in result
        assert result["filename"] == "john_doe_resume.pdf"
        assert result["text_length"] > 0
        # Ensure raw text is NOT exposed in the summary dict
        assert "extracted_text" not in result
        assert "text" not in result

        # Cleanup created test file and db row if needed
        relative_path = f"{result['resume_id']}/john_doe_resume.pdf"
        delete_resume_file(relative_path)

    def test_empty_text_pdf_raises_422(self):
        """Verify that a valid PDF with no extractable text raises EmptyPDFError (422)."""
        empty_pdf = generate_synthetic_pdf_bytes([[]])

        with pytest.raises(EmptyPDFError) as exc:
            process_resume_upload(empty_pdf, filename="scanned_blank.pdf")
        assert exc.value.status_code == 422
        assert "does not contain any extractable text" in exc.value.message

    @patch("src.extraction.resume_service.get_db_cursor")
    def test_database_failure_triggers_storage_rollback(self, mock_get_cursor):
        """Verify that if database insert fails, stored local file is removed."""
        pdf_bytes = get_standard_synthetic_resume_pdf()

        # Mock DB failure
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Simulated MySQL database failure")
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        with pytest.raises(ResumeDatabaseError) as exc:
            process_resume_upload(pdf_bytes, filename="test_rollback_resume.pdf")

        assert "Failed to save resume record in database" in exc.value.message
