# Dataset Card: Resume-Job Matching Prototype Dataset

## 1. Dataset Summary

* **Project Title:** AI-Based Resume Screening and Job Matching System
* **Dataset Version:** `1.0.0-dev` (Phase 2 Development Prototype)
* **Dataset Nature:** **Curated Synthetic / Development Data**
* **Primary Task:** Binary Classification (`0` = Poor/No Match, `1` = Qualified Match)

---

## 2. Dataset Purpose & Intended Use

This dataset was designed and authored specifically for the engineering, testing, validation, and benchmarking of NLP and machine learning pipelines in an educational course project environment. It provides structured text profiles, skill vectors, job requirement specifications, and ground-truth compatibility annotations.

**Intended Uses:**
* Development of text extraction and preprocessing routines.
* Training and benchmarking baseline NLP models (TF-IDF + Cosine Similarity, SVM, ANN).
* Validation of candidate-level train/validation/test splitting logic.

**Out-of-Scope Uses:**
* Production hiring or commercial employment decision-making.
* High-stakes candidate filtering without human-in-the-loop oversight.

---

## 3. Dataset Composition & Statistics

| Component | File Path | Count | Description |
| :--- | :--- | :--- | :--- |
| **Resumes** | `data/raw/resumes/resumes.csv` | **25** | Structured candidate profiles spanning 10 tech domains & non-tech controls |
| **Jobs** | `data/raw/jobs/jobs.csv` | **12** | Distinct job postings with required & preferred qualifications |
| **Pairs** | `data/raw/raw_pairs.csv` | **116** | Annotated resume-job pairs with binary labels (Balanced 0/1) |
| **Taxonomy** | `data/taxonomy/skills.csv` | **100** | Canonical technical skills, category taxonomy, and common aliases |

---

## 4. Data Source & Synthetic Data Disclosure

> **Transparency Disclosure:** All candidate profiles, resumes, and job postings in this Phase 2 dataset are **synthetic records** created to model real-world resume and job requisition patterns. No personally identifiable information (PII), proprietary corporate data, or scraped profiles from LinkedIn, Indeed, or personal websites were utilized.

If external public datasets (e.g., Kaggle Resume Dataset) are incorporated in future phases, the sources and licenses will be appended to this document.

---

## 5. Data Schema

### 5.1 Resumes (`resumes.csv`)
* `resume_id` (string, PK)
* `resume_text` (string, raw summary)
* `education` (string, e.g., B.Tech, M.S.)
* `degree` (string, e.g., Bachelor, Master)
* `field_of_study` (string, e.g., Computer Science)
* `experience_years` (float)
* `skills` (string, semicolon-separated canonical skills)
* `certifications` (string, optional)
* `projects` (string, sample project titles)

### 5.2 Jobs (`jobs.csv`)
* `job_id` (string, PK)
* `job_title` (string)
* `company` (string)
* `job_description` (string)
* `job_category` (string)
* `required_skills` (string, semicolon-separated)
* `preferred_skills` (string, semicolon-separated)
* `minimum_experience` (float)
* `education_requirement` (string)

### 5.3 Resume-Job Pairs (`raw_pairs.csv`)
* `pair_id` (string, PK)
* `resume_id` (string, FK)
* `job_id` (string, FK)
* `match_label` (integer: `0` or `1`)

---

## 6. Limitations & Known Biases

1. **Vocabulary Coverage:** Prototype contains a curated subset of 100 skills. Niche or emerging libraries may not be indexed in this version.
2. **Synthetic Simplification:** Text descriptions are concise and well-formed; real-world resumes often exhibit erratic formatting, OCR artifacts, and ambiguous phrasing.
3. **Class Balance:** Labels were assigned via deterministic rubric, which may reflect annotator assumptions regarding seniority thresholds.

---

## 7. Privacy & Ethical Considerations

* No live personal data or candidate contact details (phone numbers, addresses, emails) are stored.
* Adheres strictly to non-scraping policies and reproducible academic standards.
