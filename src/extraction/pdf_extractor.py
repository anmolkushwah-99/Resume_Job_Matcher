"""
PDF Text Extraction Module.
Uses pypdf to parse and extract raw textual content from PDF files.
"""

import io
from pathlib import Path
from typing import Union, BinaryIO
import pypdf
from pypdf.errors import PdfReadError


class PDFExtractionError(Exception):
    """Custom exception raised when PDF parsing or extraction fails."""
    pass


def extract_text_from_pdf(file_input: Union[str, Path, bytes, BinaryIO]) -> str:
    """
    Extracts all textual content from a PDF document page by page.

    Handles multiple input types:
    - File path (str or Path)
    - Raw bytes
    - File-like stream (io.BytesIO, io.BufferedReader, Flask FileStorage.stream)

    Key properties:
    - Preserves sequential page order.
    - Gracefully handles empty pages or pages with None text return.
    - Preserves technical symbols (e.g. C++, C#, .NET, React.js).
    - Raises PDFExtractionError for corrupted, unreadable, or password-protected PDFs.

    :param file_input: PDF path, bytes, or file stream.
    :return: Combined extracted text from all pages.
    :raises PDFExtractionError: If the PDF cannot be opened or parsed.
    """
    if file_input is None:
        raise PDFExtractionError("No file input provided for PDF extraction.")

    stream = None

    try:
        if isinstance(file_input, (str, Path)):
            path_obj = Path(file_input)
            if not path_obj.exists():
                raise PDFExtractionError(f"PDF file not found at path: {file_input}")
            stream = open(path_obj, "rb")
        elif isinstance(file_input, bytes):
            if len(file_input) == 0:
                raise PDFExtractionError("PDF byte stream is empty.")
            stream = io.BytesIO(file_input)
        elif hasattr(file_input, "read"):
            # File-like object (e.g., BytesIO or Werkzeug FileStorage stream)
            # Ensure stream is seeked to start if possible
            if hasattr(file_input, "seek"):
                try:
                    file_input.seek(0)
                except Exception:
                    pass
            stream = file_input
        else:
            raise PDFExtractionError(f"Unsupported file input type: {type(file_input)}")

        try:
            reader = pypdf.PdfReader(stream)
        except (PdfReadError, Exception) as pe:
            raise PDFExtractionError(f"Invalid or corrupted PDF file: {str(pe)}") from pe

        if reader.is_encrypted:
            try:
                # Attempt to decrypt with empty password for unencrypted access
                reader.decrypt("")
            except Exception as de:
                raise PDFExtractionError(f"PDF is password protected and cannot be read: {str(de)}") from de

        total_pages = len(reader.pages)
        if total_pages == 0:
            return ""

        page_texts = []
        for page_idx, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text()
                if extracted:
                    page_texts.append(extracted)
                else:
                    # Page has no extractable text (e.g., scanned image or empty page)
                    page_texts.append("")
            except Exception as page_err:
                # If a specific page fails, capture partial text and log warning
                page_texts.append("")

        combined_text = "\n\n".join(t for t in page_texts if t.strip())
        return combined_text

    except PDFExtractionError:
        raise
    except Exception as e:
        raise PDFExtractionError(f"Unexpected error while extracting text from PDF: {str(e)}") from e
    finally:
        # If we opened a file from a string path, ensure it is closed
        if isinstance(file_input, (str, Path)) and stream and hasattr(stream, "close"):
            stream.close()
