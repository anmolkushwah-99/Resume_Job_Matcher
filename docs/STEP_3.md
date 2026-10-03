# Step 3 — Resume Upload and Text Extraction Documentation

## 1. Objective

The objective of **Step 3** is to build an end-to-end, privacy-conscious PDF resume ingestion pipeline for the **AI-Based Resume Screening and Job Matching System**. This pipeline enables candidates or recruiters to upload resume PDF documents, performs robust file validation, extracts raw multi-page textual content while strictly preserving technical terms and symbols, applies conservative text cleaning, stores the original PDF securely on the local filesystem (`data/resumes/`), and persists extracted metadata and text records into the MySQL database (`resume_job.resumes` table).

---

## 2. Architecture & Pipeline Workflow

```
Candidate / Recruiter
        │  (Uploads Resume PDF)
        ▼
Flask API Endpoint (POST /api/resumes/upload)
        │
        ├── 1. File Validation
        │      • Check file presence and valid filename
        │      • Enforce allowed extension (.pdf only)
        │      • Limit maximum file size (10 MB)
        │      • Sanitize filename against directory traversal attacks
        │
        ├── 2. PDF Text Extraction (src/extraction/pdf_extractor.py)
        │      • Sequentially parses pages via pypdf
        │      • Preserves page ordering and handles empty/unparseable pages
        │      • Preserves technical vocabulary and symbols (C++, C#, .NET, etc.)
        │
        ├── 3. Text Normalization (src/preprocessing/text_cleaner.py)
        │      • Strips non-printable and control characters
        │      • Normalizes line endings (\r\n -> \n) and unicode (NFKC)
        │      • Collapses excessive horizontal whitespace and blank lines
        │      • Retains numbers, casing, and section structures
        │
        ├── 4. Local File Storage (src/storage/local_storage.py)
        │      • Stores file at path: data/resumes/<UUID>/<sanitized_filename>
        │      • Prevents path traversal vulnerabilities
        │
        ├── 5. Database Persistence (MySQL Database: resume_job, Table: resumes)
        │      • Inserts: id, user_id, filename, file_path, extracted_text
        │      • Transactional rollback: deletes local PDF file if DB insert fails
        │
        ▼
JSON Success Response (201 Created)
        • Returns: success, message, resume_id, filename, text_length
        • Privacy: Raw extracted text is NEVER exposed in the API response
```

---

## 3. Files Created and Modified

| File Path | Action | Description |
|---|---|---|
| `requirements.txt` | Modified | Added `pymysql>=1.1.0`, `cryptography>=41.0.0`, `supabase>=2.0.0`, and `python-dotenv>=1.0.0` |
| `data/mysql_schema.sql` | Created | Complete MySQL schema definition with all 7 relational tables, indexes, and constraints |
| `src/__init__.py` | Created | Source package root initializer |
| `src/database/__init__.py` | Created | Database module exports (MySQL & Supabase) |
| `src/database/mysql_client.py` | Created | MySQL connection pooling, cursor context manager, schema initialization, and health checks |
| `src/database/supabase_client.py` | Created | Supabase client singleton, URL normalizer, and safe health checker |
| `src/storage/__init__.py` | Created | Local storage module exports |
| `src/storage/local_storage.py` | Created | Local file storage manager for saving, retrieving, and safely cleaning up resume PDFs |
| `src/preprocessing/__init__.py` | Created | Preprocessing module exports |
| `src/preprocessing/text_cleaner.py` | Created | Conservative text cleaning and normalization preserving technical terms |
| `src/extraction/__init__.py` | Created | Extraction module exports |
| `src/extraction/pdf_extractor.py` | Created | PDF text parser built with `pypdf` |
| `src/extraction/resume_service.py` | Created | Upload orchestrator, validation, local storage, MySQL persistence & rollback |
| `app/__init__.py` | Created | Flask application factory with CORS, 10MB limits, and JSON error handlers |
| `app/routes.py` | Created | Flask route handlers for `/`, `/api/health`, and `POST /api/resumes/upload` |
| `conftest.py` | Created | Pytest path resolution configuration |
| `tests/__init__.py` | Created | Test package initializer |
| `tests/pdf_test_utils.py` | Created | Synthetic valid PDF byte generator for isolated automated testing |
| `tests/test_pdf_extractor.py` | Created | Unit tests for single/multi-page, empty page, corrupt, and technical term extraction |
| `tests/test_text_cleaner.py` | Created | Unit tests for whitespace, blank line, and technical term normalization |
| `tests/test_mysql.py` | Created | MySQL connectivity and cursor execution tests |
| `tests/test_supabase.py` | Created | Safe connectivity, singleton, and URL normalization tests |
| `tests/test_resume_service.py` | Created | Unit & rollback tests for validation, local storage, and database persistence |
| `tests/test_api.py` | Created | Flask integration test suite covering upload validation, error codes, and responses |
| `README.md` | Modified | Updated with Step 3 specifications, usage instructions, and MySQL configuration |
| `docs/STEP_3.md` | Created | Step 3 architectural documentation |

---

## 4. PDF Upload & Validation

### Validation Rules
1. **File Presence:** The multipart form body must contain a file item with field name `file`.
2. **Filename Integrity:** Rejects empty filenames or malformed extensions.
3. **Allowed Format:** Strictly `.pdf` files are accepted. Formats like `.doc`, `.docx`, `.txt`, `.jpg`, `.png`, and `.zip` are rejected with HTTP 400.
4. **File Size Limit:** Files larger than 10 MB (`10 * 1024 * 1024` bytes) are rejected with HTTP 413.
5. **Path Traversal Protection:** Base names are extracted and sanitized via `werkzeug.utils.secure_filename` to neutralize `../` and directory traversal vectors.

---

## 5. PDF Text Extraction

- Built on `pypdf.PdfReader`.
- Iterates sequentially through all document pages.
- Gracefully handles pages with empty or unparseable text by mapping `None` returns to empty strings.
- Detects corrupted, unreadable, or encrypted PDFs and raises a typed `PDFExtractionError`.
- Strictly maintains character symbols needed for technical keywords (e.g. `C++`, `C#`, `.NET`, `Node.js`, `React.js`, `REST API`, `Scikit-learn`).

---

## 6. Conservative Text Cleaning

Cleaning is deliberately conservative to prepare text for future NLP stages without destroying domain-specific grammar or vocabulary:
- Replaces control characters (null bytes `\x00`, form feeds `\x0c`, vertical tabs).
- Normalizes carriage returns (`\r\n` and `\r` to `\n`).
- Normalizes unicode via `unicodedata.normalize("NFKC")`.
- Normalizes multiple spaces/tabs into a single space on each line.
- Collapses 3 or more consecutive newlines into 2 (`\n\n`) to preserve distinct paragraph and section boundaries.
- **Strictly preserves:** Technical terms, casing, numbers, dates, GPA/scores, bullet points, colons, and section names.
- **Excludes:** No stopword removal, stemming, lemmatization, or TF-IDF vectorization in Step 3.

---

## 7. Storage Integration (Local File Storage)

- Base Directory: `data/resumes/`
- File Naming Scheme: `<UUID>/<sanitized_filename>` (e.g., `data/resumes/550e8400-e29b-41d4-a716-446655440000/john_doe_resume.pdf`)
- Security: Absolute path traversal protection preventing directory escapes.

---

## 8. Database Integration (MySQL: `resume_job`)

- Database: `resume_job`
- Target Table: `resumes`
- Columns Populated:
  - `id`: CHAR(36) UUID string primary key
  - `user_id`: Optional CHAR(36) foreign key referencing `users(id)`
  - `filename`: Original sanitized filename (VARCHAR(255))
  - `file_path`: Relative storage path (TEXT)
  - `extracted_text`: Cleaned resume text (LONGTEXT)
  - `uploaded_at`: Default `CURRENT_TIMESTAMP`

### Transactional Rollback Guarantee
If the database insertion fails after the PDF file has already been saved to local storage, the service immediately removes the local file (`delete_resume_file(saved_file_path)`) to prevent orphaned files.

---

## 9. API Specification

### Endpoint: `POST /api/resumes/upload`

**Request Headers:**
`Content-Type: multipart/form-data`

**Request Body:**
- `file` *(required)*: Binary PDF file.
- `user_id` *(optional)*: UUID string referencing candidate user record.

**Success Response (`201 Created`):**
```json
{
  "success": true,
  "message": "Resume uploaded and processed successfully",
  "resume_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "filename": "sample_resume.pdf",
  "text_length": 1420
}
```

**Error Responses:**
- `400 Bad Request`: Missing file, unsupported format, or empty file.
- `413 Request Entity Too Large`: File exceeds 10 MB.
- `422 Unprocessable Entity`: Corrupted PDF or PDF containing no usable text.
- `500 Internal Server Error`: Storage or database insertion failure (without exposing stack traces).

---

## 10. Verification and Testing

44 automated tests pass across unit and integration suites:
- `tests/test_pdf_extractor.py`: 10 tests passing
- `tests/test_text_cleaner.py`: 6 tests passing
- `tests/test_mysql.py`: 3 tests passing
- `tests/test_supabase.py`: 6 tests passing
- `tests/test_resume_service.py`: 10 tests passing
- `tests/test_api.py`: 9 tests passing

To run the complete test suite:
```bash
python -m pytest -v
```

---

## 11. Security and Privacy Guidelines

1. **Credential Safety:** Secrets (`MYSQL_PASSWORD`, `SECRET_KEY`) are read exclusively from `.env` and never committed or exposed to the client.
2. **Privacy by Design:** Extracted resume text contains Personally Identifiable Information (PII) and is **never** echoed in API JSON responses or debug logs.
3. **Injection & Traversal Protection:** Parameterized SQL queries are strictly used for all MySQL operations. Filenames are sanitized and namespaced under UUID directories.
4. **Payload Size Guardrails:** 10 MB request limit enforced at both Flask server and service layers.

---

## 12. Future Work (Step 4+)

- **Step 4:** NLP Preprocessing and Resume Information Extraction (Named Entity Recognition, Skill Taxonomy matching, Education & Experience parsing).
- **Step 5:** TF-IDF & Cosine Similarity baseline scoring.
- **Step 6–8:** Machine Learning Classification (SVM, ANN, LSTM).
