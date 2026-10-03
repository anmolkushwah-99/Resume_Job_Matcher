"""
Resume Processing Service.
Orchestrates PDF validation, text extraction, cleaning, local file storage,
and MySQL database persistence with transactional rollback on partial failure.
"""

import os
import re
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from werkzeug.utils import secure_filename

from src.database.mysql_client import get_db_cursor
from src.storage.local_storage import save_resume_file, delete_resume_file
from src.extraction.pdf_extractor import extract_text_from_pdf, PDFExtractionError
from src.preprocessing.text_cleaner import clean_resume_text

logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf"}


class ResumeValidationError(Exception):
    """Raised when file validation fails (bad format, empty file, oversize)."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class EmptyPDFError(Exception):
    """Raised when PDF does not contain any extractable text."""
    def __init__(self, message: str = "PDF contains no extractable text."):
        super().__init__(message)
        self.message = message
        self.status_code = 422


class ResumeStorageError(Exception):
    """Raised when upload or local file storage operations fail."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        self.status_code = 500


class ResumeDatabaseError(Exception):
    """Raised when database insertion fails."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        self.status_code = 500


def sanitize_pdf_filename(filename: str) -> str:
    """
    Sanitizes filename and prevents directory traversal attacks.
    """
    if not filename:
        return "resume.pdf"

    # Normalize backslashes to forward slashes and take the basename
    base_name = Path(filename.replace("\\", "/")).name

    # Use werkzeug's secure_filename as second pass
    base = secure_filename(base_name)
    if not base or base.lower() in ("pdf", ".pdf", "pdf.pdf"):
        base = "resume.pdf"

    # Ensure extension is .pdf
    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf"

    # Truncate filename if excessively long
    if len(base) > 200:
        base = base[:196] + ".pdf"

    return base


def validate_resume_file(file_storage_or_bytes: Any, filename: Optional[str] = None) -> Tuple[bytes, str]:
    """
    Validates uploaded resume file:
    1. Checks presence of file and filename
    2. Validates .pdf extension
    3. Checks file size does not exceed 10 MB
    4. Validates byte content is non-empty

    :param file_storage_or_bytes: Werkzeug FileStorage or bytes.
    :param filename: Optional filename override.
    :return: Tuple of (file_bytes, safe_filename).
    :raises ResumeValidationError: If any validation rule is violated.
    """
    if file_storage_or_bytes is None:
        raise ResumeValidationError("No file provided for upload.")

    actual_filename = filename

    if hasattr(file_storage_or_bytes, "filename"):
        actual_filename = file_storage_or_bytes.filename or actual_filename

    if not actual_filename or actual_filename.strip() == "":
        raise ResumeValidationError("Uploaded file has an empty or missing filename.")

    # Check extension
    ext = Path(actual_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ResumeValidationError(
            f"Only PDF files are supported. Received file with extension: '{ext or 'unknown'}'."
        )

    # Read bytes
    if hasattr(file_storage_or_bytes, "read"):
        file_bytes = file_storage_or_bytes.read()
    elif isinstance(file_storage_or_bytes, bytes):
        file_bytes = file_storage_or_bytes
    else:
        raise ResumeValidationError("Unsupported file format or stream.")

    if not file_bytes or len(file_bytes) == 0:
        raise ResumeValidationError("Uploaded file is empty (0 bytes).")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ResumeValidationError(
            f"File size ({len(file_bytes)} bytes) exceeds the maximum allowed limit of 10 MB.",
            status_code=413
        )

    safe_name = sanitize_pdf_filename(actual_filename)
    return file_bytes, safe_name


def process_resume_upload(
    file_input: Any,
    filename: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete Resume Upload and Text Extraction Pipeline with MySQL persistence.

    Workflow:
    1. Validate PDF file (type, size, filename).
    2. Extract text from PDF using pypdf.
    3. Validate extracted text is non-empty.
    4. Clean and normalize extracted text.
    5. Generate unique UUID and safe local file path.
    6. Save PDF to local file storage.
    7. Insert resume record into MySQL 'resumes' table.
    8. Roll back local file if database insertion fails.
    9. Return privacy-safe summary response.

    :param file_input: FileStorage, file stream, or bytes.
    :param filename: Original filename if input is raw bytes.
    :param user_id: Optional UUID of candidate/user.
    :return: Dict containing success status, resume_id, filename, and text_length.
    """
    # Step 1: Validate file
    file_bytes, safe_name = validate_resume_file(file_input, filename)

    # Step 2: Extract text from PDF
    try:
        raw_text = extract_text_from_pdf(file_bytes)
    except PDFExtractionError as pe:
        logger.warning("PDF extraction failed: %s", str(pe))
        raise ResumeValidationError(f"Unable to parse PDF file: {str(pe)}", status_code=422) from pe

    # Step 3: Clean extracted text
    clean_text = clean_resume_text(raw_text)

    # Step 4: Verify usable text content
    if not clean_text or len(clean_text.strip()) == 0:
        logger.warning("Uploaded PDF '%s' yielded no extractable text.", safe_name)
        raise EmptyPDFError("The uploaded PDF does not contain any extractable text.")

    # Step 5: Prepare Storage Path and UUID
    upload_uuid = str(uuid.uuid4())
    relative_storage_path = f"{upload_uuid}/{safe_name}"

    # Step 6: Save PDF to Local Storage
    saved_file_path = None
    try:
        saved_file_path = save_resume_file(file_bytes, relative_storage_path)
        logger.info("Successfully stored resume PDF locally: %s", saved_file_path)
    except Exception as se:
        logger.error("Local file storage failed for '%s': %s", relative_storage_path, str(se))
        raise ResumeStorageError(f"Failed to save resume file to storage: {str(se)}") from se

    # Step 7: Insert into MySQL Database ('resumes' table)
    try:
        query = """
            INSERT INTO resumes (id, user_id, filename, file_path, extracted_text)
            VALUES (%s, %s, %s, %s, %s);
        """
        params = (
            upload_uuid,
            user_id if user_id else None,
            safe_name,
            relative_storage_path,
            clean_text
        )

        with get_db_cursor(commit=True) as cur:
            cur.execute(query, params)

        logger.info("Successfully saved resume record in MySQL database: %s", upload_uuid)

    except Exception as de:
        logger.error("MySQL Database insert failed for '%s': %s", upload_uuid, str(de))

        # Step 8: Transactional Rollback (Delete local file if DB insert fails)
        if saved_file_path:
            try:
                logger.info("Rolling back local resume file: %s", saved_file_path)
                delete_resume_file(saved_file_path)
            except Exception as cleanup_err:
                logger.critical(
                    "Orphaned file cleanup failed for path '%s': %s",
                    saved_file_path,
                    str(cleanup_err)
                )

        raise ResumeDatabaseError(f"Failed to save resume record in database: {str(de)}") from de

    # Step 9: Return privacy-safe response (DO NOT include complete text)
    return {
        "success": True,
        "message": "Resume uploaded and processed successfully",
        "resume_id": upload_uuid,
        "filename": safe_name,
        "text_length": len(clean_text)
    }
