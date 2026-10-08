# AI-Based Resume Screening and Job Matching System

An academic Natural Language Processing (NLP) and Machine Learning (ML) system designed for automated resume screening, structured profile extraction, candidate-job compatibility classification, deterministic ranking, and transparent skill gap explanation.

---

## 1. Overview

The **AI-Based Resume Screening and Job Matching System** automates the recruitment screening process by analyzing unstructured resume PDFs, extracting technical competencies and experience, and evaluating candidate compatibility against job descriptions.

### What the System Produces:
- **Structured Profile Extraction:** Automated extraction of skills, education, and years of experience from PDF resumes.
- **ML-Based Compatibility Scoring:** Supervised classification and continuous decision scoring using **TF-IDF + Linear Support Vector Machine (LinearSVC)**.
- **Deterministic Job Ranking:** Multi-job evaluation ranking opportunities by decision score with deterministic tie-breaking.
- **Skill Gap Breakdown:** Direct identification of matched skills and missing prerequisites.
- **Interactive Web Interface:** A responsive Single-Page Application (SPA) for recruiters and candidates to inspect profiles, run matching, and compare evaluations.

---

## 2. Problem Statement & Objectives

### Problem Statement
Manual resume screening is time-consuming, prone to subjective bias, and difficult to scale across high volumes of applicant profiles. Furthermore, many automated screening tools function as opaque "black boxes" without providing interpretable justifications. This project implements an explainable, automated matching pipeline with zero candidate-level evaluation data leakage.

### Objectives
- Automate text extraction from PDF resumes.
- Extract canonical skills using a curated 100-skill taxonomy across 7 categories.
- Represent candidate and job text numerically using TF-IDF unigram and bigram features.
- Evaluate multiple ML paradigms (Baseline Cosine Similarity, LinearSVC, ANN/MLP, and LSTM).
- Select and integrate an optimal classifier based on generalization, specificity, and computational efficiency.
- Provide transparent skill explanations (Matched Competencies vs. Missing Requirements).
- Deliver a responsive web interface and RESTful APIs backed by a MySQL database.

---

## 3. Key Features

- **PDF Resume Ingestion:** Robust text extraction using `pypdf`.
- **NLP Preprocessing:** Custom tokenization preserving technical terms (`C++`, `C#`, `.NET`, `Node.js`, `CI/CD`), stopword filtering, and spaCy lemmatization.
- **Taxonomy-Driven Skill Extraction:** Boundary-enforced regex matching mapped to a 100-skill dictionary.
- **Education & Experience Parsing:** Heuristic extraction of degrees, majors, and total experience duration.
- **Zero-Leakage Evaluation:** Grouped candidate-aware partitioning (`GroupShuffleSplit`) to prevent candidate overlap across train/test splits.
- **Supervised ML Classification:** LinearSVC model predicting match (`decision_score >= 0.0`) vs. non-match (`decision_score < 0.0`).
- **Deterministic Job Ranking:** Evaluates a candidate across all benchmark jobs, sorted by `decision_score` descending (`job_id` ascending as tie-breaker).
- **Explainable Skill Matrix:** Computes matched skills, missing skills, and overlap percentage.
- **Persistent Match History:** Relational logging of evaluation transactions in MySQL.
- **Responsive Web Dashboard:** Fast client-side hash routing (`#/dashboard`, `#/resumes`, `#/jobs`, `#/matcher`, `#/history`, `#/how-it-works`).

---

## 4. System Workflow & Architecture

### End-to-End Workflow
```text
Candidate Resume (PDF)
        │
        ▼
PDF Text Extraction (pypdf)
        │
        ▼
NLP Preprocessing & Cleaning (spaCy + NLTK)
        │
        ▼
Information & Skill Extraction (100-Skill Master Taxonomy)
        │
        ▼
Joint Pair Representation (Resume Text + Job Description)
        │
        ▼
TF-IDF Vectorization (1,661 Unigram & Bigram Features)
        │
        ▼
LinearSVC Model Inference ──► Continuous Decision Score d(x)
        │
        ▼
Classification: Match (d(x) >= 0.0) | Non-Match (d(x) < 0.0)
        │
        ├──► Skill Gap Explainer (Matched: R ∩ J | Missing: J \ R)
        └──► Deterministic Ranking (Score DESC, Job ID ASC)
        │
        ▼
Web Dashboard & MySQL Persistence
```

### Layered System Architecture
```text
┌────────────────────────────────────────────────────────────────────────┐
│                      1. Frontend Presentation Layer                    │
│    Responsive Single-Page Application (HTML5 / CSS3 / JavaScript ES6+) │
│         Dashboard | Resumes | Jobs | Match Engine | History | Docs     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP REST (JSON)
┌───────────────────────────────────▼────────────────────────────────────┐
│                       2. Flask Application Backend                     │
│               REST API Endpoints (app/routes.py)                       │
│           Resume Service  │  Match Service  │  NLP Service             │
└───────────────────┬───────────────────────────────┬────────────────────┘
                    │                               │
┌───────────────────▼───────────────┐   ┌───────────▼────────────────────┐
│      3. NLP & Matching Engine     │   │      4. Persistence Layer      │
│   • pypdf Text Extractor          │   │   • MySQL 8.x (resume_job)     │
│   • spaCy & NLTK Preprocessing    │   │   • Local Resume Storage       │
│   • 100-Skill Master Taxonomy     │   └────────────────────────────────┘
│   • LinearSVC Inference Engine    │
│   • Set-Theoretic Skill Explainer │
└───────────────────┬───────────────┘
                    │ Loads Frozen Artifacts
┌───────────────────▼────────────────────────────────────────────────────┐
│                       5. Machine Learning Model Store                  │
│    • tfidf_svm_model.joblib (LinearSVC Classifier)                     │
│    • tfidf_svm_vectorizer.joblib (TF-IDF N-gram Vectorizer)            │
│    • Model Metadata & Evaluation Benchmarks (JSON)                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.11.9 | Core application runtime |
| **Backend** | Flask $\ge$ 3.0.0, Flask-CORS | RESTful API and web application server |
| **Database** | MySQL 8.x, PyMySQL | Relational data persistence and querying |
| **NLP** | spaCy (`en_core_web_sm`), NLTK | Tokenization, protected terms, lemmatization |
| **Machine Learning** | scikit-learn $\ge$ 1.3.0 | TF-IDF vectorization, LinearSVC, metrics |
| **Deep Learning** | TensorFlow / Keras $\ge$ 2.15.0 | Comparative ANN and LSTM baseline models |
| **PDF Extraction** | pypdf $\ge$ 4.0.0 | PDF document parsing and text stream decoding |
| **Model Storage** | joblib | Serialization of models and vectorizers |
| **Frontend** | HTML5, CSS3, JavaScript ES6+ | Responsive Single-Page Application interface |
| **Testing** | pytest | Unit, API, and integration test suites |

---

## 6. Dataset & Methodology

- **Dataset Composition:** Academic benchmark consisting of **25 candidate resumes** (across 10 domains), **12 benchmark jobs**, and **120 annotated pairs** (38 positive matches, 82 negative non-matches).
- **Skill Taxonomy:** 100 canonical skills categorized into: *Programming Languages*, *AI / ML*, *Web Development*, *Databases*, *Cloud & DevOps*, *Data Engineering*, and *Tools/Platforms*.
- **Candidate-Aware Data Split:** To prevent data leakage, dataset partitioning is strictly candidate-grouped (`GroupShuffleSplit` on `resume_id`):
  - **Training Set (70%):** 17 candidates (82 pairs: 26 positive, 56 negative)
  - **Validation Set (15%):** 4 candidates (20 pairs: 7 positive, 13 negative)
  - **Held-Out Test Set (15%):** 4 candidates (18 pairs: 5 positive, 13 negative)

---

## 7. Model Evaluation & Selection

Four paradigms were evaluated on the identical held-out test benchmark (18 pairs: 5 positive, 13 negative):

### Primary Benchmark Results

| Model | Accuracy | Precision | Recall | Specificity | F1 Score | Balanced Acc. | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + Cosine Baseline** | 0.6667 | 0.4545 | 1.0000 | 0.5385 | 0.6250 | 0.7692 | 0.8308 |
| **TF-IDF + LinearSVC (Selected)** | **0.6111** | **0.3750** | **0.6000** | **0.6154** | **0.4615** | **0.6077** | **0.6154** |
| **TF-IDF + ANN / MLP** | 0.5556 | 0.3333 | 0.6000 | 0.5385 | 0.4286 | 0.5692 | 0.5538 |
| **Embedding + LSTM** | 0.2778 | 0.2778 | 1.0000 | 0.0000 | 0.4348 | 0.5000 | 0.3385 |

### Selection Rationale:
- **TF-IDF + LinearSVC (`tfidf_svm`)** was selected as the primary matching model.
- It provides the highest **specificity (0.6154)** among supervised classifiers, preventing unqualified false-positive recommendations.
- Offers deterministic signed decision margins, fast sub-millisecond inference, and clean mathematical interpretability.

---

## 8. Final Matching Engine & Explainability

### Mathematical Formulation
The classifier computes the signed continuous decision margin $d(x)$ from the separating hyperplane:

$$d(x) = w^T \phi(x) + b$$

- **Decision Rule:** $\text{Match}$ if $d(x) \ge 0.0$; $\text{Non-Match}$ if $d(x) < 0.0$.
- **Ranking Order:** Primary: `decision_score` descending; Secondary: `job_id` ascending (tie-breaker).

*(Note: The SVM decision score is a continuous distance margin, not a probability percentage).*

### Transparent Skill Explanations
Alongside the ML score, the system computes exact set-theoretic skill breakdowns:
- **Matched Skills:** $\text{Skills}_{\text{resume}} \cap \text{Skills}_{\text{job}}$
- **Missing Skills:** $\text{Skills}_{\text{job}} \setminus \text{Skills}_{\text{resume}}$
- **Skill Overlap Ratio:** $\frac{|\text{Skills}_{\text{resume}} \cap \text{Skills}_{\text{job}}|}{\max(1, |\text{Skills}_{\text{job}}|)}$

---

## 9. Database Design

The relational database `resume_job` consists of 7 structured tables:

| Table Name | Primary Key | Description | Key Relationships |
|---|---|---|---|
| `users` | `id` (CHAR 36) | User accounts (candidates, recruiters, admins) | 1:N with `resumes`, 1:N with `jobs` |
| `resumes` | `id` (CHAR 36) | Uploaded resumes, extracted text, degrees, experience | N:1 with `users`, 1:N with `resume_skills` |
| `jobs` | `id` (CHAR 36) | Benchmark jobs, descriptions, requirements | N:1 with `users`, 1:N with `job_skills` |
| `skills` | `id` (CHAR 36) | 100-skill canonical taxonomy with categories & aliases | Referenced by junction tables |
| `resume_skills` | `(resume_id, skill_id)` | Extracted skills associated with a resume | N:M junction (`resumes` $\leftrightarrow$ `skills`) |
| `job_skills` | `(job_id, skill_id)` | Required/preferred skills for a job | N:M junction (`jobs` $\leftrightarrow$ `skills`) |
| `matches` | `id` (CHAR 36) | Persistent evaluation results, scores, and timestamps | N:1 with `resumes`, N:1 with `jobs` |

---

## 10. REST API Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health status and model readiness check |
| `GET` | `/api/stats` | System metrics (total resumes, jobs, matches) |
| `GET` | `/api/resumes` | List candidate resumes with extracted details |
| `GET` | `/api/resumes/<id>` | Retrieve detailed parsed profile for a resume |
| `POST` | `/api/resumes/upload` | Upload a new PDF resume |
| `POST` | `/api/resumes/<id>/process` | Trigger NLP feature extraction on an uploaded resume |
| `GET` | `/api/jobs` | List benchmark jobs with skill requirements |
| `POST` | `/api/matches/svm` | Run single resume-job SVM evaluation |
| `POST`/`GET` | `/api/resumes/<id>/svm-matches` | Multi-job batch matching and ranked recommendations |
| `POST` | `/api/matches/svm/batch` | Batch evaluate multiple resumes against multiple jobs |
| `GET` | `/api/matches/history` | Retrieve historical match logs from MySQL |

---

## 11. Project Directory Structure

```text
Resume_Job_Matcher/
├── app/
│   ├── static/                         # Frontend UI (CSS and modular JavaScript)
│   │   ├── css/                        # dashboard.css, matching.css, style.css
│   │   └── js/                         # api.js, app.js, matching.js, ui.js
│   ├── templates/                      # index.html (Single-Page Application root)
│   ├── __init__.py                     # Flask application factory
│   └── routes.py                       # RESTful API route definitions
├── data/
│   ├── processed/resume_nlp/           # Serialized NLP JSON candidate profiles
│   ├── raw/                            # jobs.csv, resumes.csv, raw_pairs.csv
│   ├── taxonomy/skills.csv             # 100-skill master taxonomy and aliases
│   └── mysql_schema.sql                # Relational MySQL DDL schema
├── models/                             # Frozen model artifacts (.joblib, .keras)
│   ├── tfidf_svm_model.joblib          # Trained LinearSVC model (Primary Engine)
│   └── tfidf_svm_vectorizer.joblib     # Fitted TF-IDF vectorizer (1,661 features)
├── src/
│   ├── database/mysql_client.py        # PyMySQL connection pool and helpers
│   ├── extraction/                     # PDF text extractor and resume service
│   ├── matching/                       # FinalMatchEngine, explanations, builders
│   ├── models/                         # Model wrapper classes (SVM, ANN, LSTM)
│   └── preprocessing/                  # NLP pipeline, tokenizers, skill extractors
├── tests/                              # Automated test suite (41 pytest modules)
├── app.py                              # Application entry point
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 12. Installation & Quickstart

### Prerequisites
- Python `3.11.x`
- MySQL Server `8.x`
- Git

### Setup Steps

1. **Clone Repository & Create Virtual Environment:**
   ```powershell
   git clone https://github.com/anmolkushwah-99/Resume_Job_Matcher.git
   cd Resume_Job_Matcher
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies & NLP Model:**
   ```powershell
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Database Configuration:**
   Create the database and import the schema:
   ```sql
   CREATE DATABASE resume_job CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```
   ```powershell
   mysql -u root -p resume_job < data\mysql_schema.sql
   ```
   Create a `.env` file in the root directory:
   ```env
   DB_HOST=127.0.0.1
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=your_mysql_password
   DB_NAME=resume_job
   SECRET_KEY=academic_project_key
   ```

4. **Run the Application:**
   ```powershell
   python app.py
   ```
   Open browser at: `http://127.0.0.1:5000/`

5. **Run Automated Tests:**
   ```powershell
   pytest -q
   ```

---

## 13. Project Summary

| Field | Detail |
|---|---|
| **Project Title** | AI-Based Resume Screening and Job Matching System |
| **Domain** | Natural Language Processing & Machine Learning |
| **Primary Model** | TF-IDF + Linear Support Vector Machine (`LinearSVC`) |
| **Classification Rule** | $d(x) \ge 0.0 \rightarrow \text{Match}$, $d(x) < 0.0 \rightarrow \text{Non-Match}$ |
| **Ranking Logic** | `decision_score` DESC, `job_id` ASC (Deterministic) |
| **Evaluation Strategy** | Candidate-Aware Group Split (`GroupShuffleSplit`) |
| **Stack** | Python 3.11, Flask, MySQL, scikit-learn, spaCy, pypdf, Vanilla HTML/CSS/JS |
