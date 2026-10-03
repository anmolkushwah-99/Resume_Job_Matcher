# Step 4 — NLP Preprocessing & Resume Entity/Feature Extraction

## 1. Objective

The objective of **Step 4** is to implement a deterministic, explainable Natural Language Processing (NLP) pipeline for the **AI-Based Resume Screening and Job Matching System**. This subsystem converts raw, cleaned resume text into structured candidate features—including tokenization with technical keyword preservation, section segmentation, master taxonomy skill extraction with alias mapping, education parsing, experience duration extraction, and MySQL database persistence.

---

## 2. NLP Pipeline Architecture

```
Raw / Cleaned Resume Text
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Central NLP Pipeline Orchestrator           │
│                 (src/preprocessing/nlp_pipeline.py)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┼────────────────────────┐
       ▼                       ▼                        ▼
1. Tokenization & Lemmas 2. Section Detection    3. Entity Extraction
   • spaCy en_core_web_sm   • Summary / Profile     • 100-Skill Master Taxonomy
   • Technical Protection   • Technical Skills      • Alias Normalization
   • Stopwords Filtered     • Education             • Education (Degree, Major)
   • Normalized Tokens      • Work Experience       • Experience (Years, Roles)
                            • Projects / Certs
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Structured Candidate Features               │
│                 (Saved to data/processed/resume_nlp/*.json) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 MySQL Persistence (resume_job)              │
│  • UPDATE resumes (education, degree, field_of_study, yrs)  │
│  • INSERT INTO resume_skills (resume_id, skill_id)          │
│  • Transaction Safe with Rollback                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed NLP Components

### A. Tokenization (`src/preprocessing/tokenizer.py`)
- Powered by `spaCy` (`en_core_web_sm`).
- **Technical Symbol Protection:** Registers custom tokenizer special cases to prevent splitting symbols in terms like `C++`, `C#`, `.NET`, `Node.js`, `React.js`, `Scikit-learn`, `CI/CD`, `REST API`, `ASP.NET`, etc.
- Provides raw token sequences and lowercased normalized tokens.

### B. Controlled Stopword Handling
- Standard English grammatical stopwords are removed **only** from the NLP feature representation.
- **Original Text Preservation:** The stored `extracted_text` remains 100% unaltered.
- **Protected Technical Terms:** Terms like `c`, `r`, `go`, `ai`, `ml`, `dl`, `it`, `git` are protected and never filtered.

### C. Lemmatization
- Applies spaCy linguistic lemmatization (e.g. `developing` -> `develop`, `applications` -> `application`).
- Preserves technical proper nouns and uppercase acronyms (e.g., `AWS`, `SQL`, `REST`, `API`).

### D. Resume Section Detection (`src/preprocessing/section_parser.py`)
- Uses regex pattern matching on heading lines.
- Case-insensitive recognition of multi-word variants:
  - `summary` (SUMMARY, CAREER OBJECTIVE, PROFESSIONAL SUMMARY, ABOUT ME, PROFILE)
  - `skills` (SKILLS, TECHNICAL SKILLS, CORE COMPETENCIES, TECH STACK, TECHNOLOGIES)
  - `education` (EDUCATION, ACADEMIC QUALIFICATIONS, EDUCATIONAL BACKGROUND, ACADEMICS)
  - `experience` (EXPERIENCE, WORK EXPERIENCE, PROFESSIONAL EXPERIENCE, INTERNSHIPS)
  - `projects` (PROJECTS, PERSONAL PROJECTS, KEY PROJECTS)
  - `certifications` (CERTIFICATIONS, CERTIFICATES, LICENSES & CERTIFICATIONS)
  - `achievements`, `languages`, `interests`

### E. Skill Extraction & Taxonomy Normalization (`src/preprocessing/skill_extractor.py`)
- Maps text against the **100-Skill Master Taxonomy** (`data/taxonomy/skills.csv`).
- **Alias Resolution:** Normalized aliases map to canonical skill names (e.g., `sklearn` -> `Scikit-learn`, `js` -> `JavaScript`, `ts` -> `TypeScript`, `golang` -> `Go`, `k8s` -> `Kubernetes`, `postgres` -> `PostgreSQL`, `tf` -> `TensorFlow`, `genai` -> `Large Language Models`).
- **Word Boundary Matching:** Strict regex boundary rules prevent false positives (e.g., "R" in "Developer", "Go" in "good", "C" in "Computer", "AI" in "Email").
- **Deduplication:** Multiple mentions of a skill or its aliases produce a single canonical entry.

### F. Education Extraction (`src/preprocessing/education_extractor.py`)
- Identifies degrees (`B.Tech`, `B.E.`, `B.Sc`, `MCA`, `BCA`, `M.Tech`, `M.S.`, `MBA`, `PhD`, `Diploma`, `Bachelor`, `Master`).
- Extracts field of study (`Computer Science`, `Information Technology`, `Data Science`, `Artificial Intelligence`, `Software Engineering`, `Statistics`, etc.).
- Identifies university / institution names where detectable.

### G. Experience Extraction (`src/preprocessing/experience_extractor.py`)
- Extracts numeric experience duration (e.g., "2 years of experience" -> `2.0`, "1.5 years" -> `1.5`, "6 months" -> `0.5`).
- Detects entry-level / fresher candidates (defaults to `0.0`).
- Identifies role titles (e.g., `Software Developer`, `Machine Learning Engineer`, `Data Analyst`, `Intern`).

---

## 4. MySQL Database Integration

- **Database:** `resume_job`
- **Tables Updated:**
  1. `resumes`: Updates `education`, `degree`, `field_of_study`, and `experience_years`.
  2. `resume_skills`: Inserts junction records `(resume_id, skill_id)`.
- **Idempotency:** Re-processing a resume replaces previous `resume_skills` records for that resume, preventing duplicates.
- **Transaction Safety:** Commits only upon total success; rolls back on failure.

---

## 5. API Endpoint

### `POST /api/resumes/<resume_id>/process`
Extracts structured features and persists them into MySQL.

**Example Request:**
```bash
curl -X POST http://127.0.0.1:5000/api/resumes/550e8400-e29b-41d4-a716-446655440000/process
```

**Success Response (`200 OK`):**
```json
{
  "success": true,
  "message": "Resume NLP processing completed successfully",
  "resume_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "sample_resume.pdf",
  "skills_detected": 7,
  "skills": [
    "Docker",
    "Flask",
    "Git",
    "PostgreSQL",
    "Python",
    "REST API",
    "Redis"
  ],
  "education_detected": true,
  "degree": "B.Tech",
  "field_of_study": "Computer Science",
  "experience_years": 3.0,
  "sections_detected": [
    "summary",
    "skills",
    "education",
    "experience",
    "projects"
  ]
}
```

---

## 6. Processed NLP JSON Output

When a resume is processed, an inspectable artifact is persisted to `data/processed/resume_nlp/<resume_id>.json`:
```json
{
  "resume_id": "RES001",
  "processed_at": "2026-10-01T11:00:00.000000Z",
  "tokens_count": 42,
  "tokens": [...],
  "normalized_tokens": [...],
  "filtered_tokens": [...],
  "lemmas": [...],
  "sections": { ... },
  "skills_count": 7,
  "skills": [
    {
      "skill_id": "SK001",
      "skill_name": "Python",
      "category": "Programming Language",
      "matched_alias": "Python"
    }
  ],
  "degree": "B.Tech",
  "field_of_study": "Computer Science",
  "experience_years": 3.0
}
```

---

## 7. Testing & Verification

**78 automated tests** passing across 10 test modules (`python -m pytest -v`):
- `tests/test_tokenizer.py`: 6 tests
- `tests/test_section_parser.py`: 5 tests
- `tests/test_skill_extractor.py`: 6 tests
- `tests/test_education_extractor.py`: 7 tests
- `tests/test_experience_extractor.py`: 7 tests
- `tests/test_nlp_pipeline.py`: 4 tests
- `tests/test_nlp_service.py`: 3 tests
- `tests/test_mysql.py`: 3 tests
- `tests/test_pdf_extractor.py`: 10 tests
- `tests/test_text_cleaner.py`: 6 tests
- `tests/test_resume_service.py`: 10 tests
- `tests/test_api.py`: 11 tests

**Validation Script:**
```bash
python src/nlp_validation.py
```
- Total resumes processed: 25/25
- Successful extractions: 25 (100%)
- Average skills detected: 4.92 per resume
- Repeatability check: 100% idempotent

---

## 8. Limitations & Scope Constraints

1. **Complex Academic Formats:** Multi-paragraph narrative transcripts or non-standard international qualifications may require fuzzy fallback rules.
2. **Deterministic Rules:** Uses pattern matching and spaCy linguistic rules; does not hallucinate entities.
3. **Step 5 Boundary:** Matching algorithms (TF-IDF, Cosine Similarity) and ML classifiers (SVM, ANN) are strictly deferred to subsequent phases.
