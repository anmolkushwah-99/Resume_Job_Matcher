# AI-Based Resume Screening and Job Matching System

## Project Description

The **AI-Based Resume Screening and Job Matching System** is an intelligent natural language processing (NLP) and machine learning (ML) platform designed to automate and streamline the recruitment screening process. The system analyzes candidate resumes (PDF format), extracts structured information and key competencies, compares candidates against job descriptions, identifies matched and missing skill gaps, and produces explainable compatibility scores and candidate rankings.

---

## Current Phase

**Phase 11 — Recruiter / Candidate Dashboard & Frontend Integration (Completed)**

> **Status:** Steps 1 through 11 are fully implemented, tested, and validated.
> - **Step 1:** Foundation & Environment
> - **Step 2:** Dataset Design & Data Validation
> - **Step 3:** Resume PDF Upload & Text Extraction
> - **Step 4:** NLP Preprocessing & Feature Extraction
> - **Step 5:** TF-IDF & Cosine Similarity Baseline Matching
> - **Step 6:** Supervised SVM Classification
> - **Step 7:** Supervised ANN / MLP Classification
> - **Step 8:** Supervised LSTM Sequence Modeling
> - **Step 9:** Comprehensive Model Evaluation & Model Selection
> - **Step 10:** Final Resume–Job Matching Engine & Integration
> - **Step 11:** Recruiter / Candidate Dashboard & Frontend Integration

---

## STEP 11 — Recruiter / Candidate Dashboard & Frontend Integration

### 1. Purpose
Provides a responsive, academic and professional user interface consuming the validated Step 10 REST APIs without altering the frozen machine-learning pipeline, decision thresholds ($\theta = 0.0$), or ranking semantics.

### 2. Frontend Features
- **Unified Navigation & SPA Layout**: Fast client-side routing between Dashboard, Candidate Resumes, Available Jobs, Find Matches, Evaluation History, and System Architecture documentation.
- **Resume Management & PDF Upload**: Upload candidate PDF resumes, parse text with `pypdf`, and inspect extracted skills, education, and experience summary.
- **Multi-Job Matching Workflow**: Select a candidate resume and run one multi-job SVM inference request (`POST /api/resumes/<resume_id>/svm-matches`) against all benchmark jobs with zero one-request-per-job latency.
- **Deterministic Ranked Display**: Displays ranked results with signed LinearSVC `Decision Score` (strictly never displayed as fake probability/percentage), `MATCH` / `NON-MATCH` badge, matched competencies, and missing skill gaps.
- **Filtering & Search**: Client-side filtering by All, Matches Only, Non-Matches, and live title keyword search.
- **Job Comparison Modal**: Select and compare up to 3 job evaluations side-by-side with matched/missing skill breakdowns.
- **Match History & Stats**: Live metrics tracking total resumes, benchmark jobs, database evaluations, and model health.

### 3. Running the Application

1. **Activate Virtual Environment**:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. **Start the Flask Backend & UI Server**:
   ```powershell
   python app.py
   ```
   Or via Flask CLI:
   ```powershell
   flask run --port=5000
   ```

3. **Access the Application**:
   Open your web browser and navigate to:
   - **Dashboard**: [http://127.0.0.1:5000/dashboard](http://127.0.0.1:5000/dashboard) (or [http://127.0.0.1:5000/](http://127.0.0.1:5000/))
   - **Health Check API**: [http://127.0.0.1:5000/api/health](http://127.0.0.1:5000/api/health)
   - **API Stats**: [http://127.0.0.1:5000/api/stats](http://127.0.0.1:5000/api/stats)

### 4. Verification & Testing

Run all unit, functional, regression, and frontend integration test suites:
```powershell
pytest -q
```
**Results:** `226 passed, 42 warnings in ~164s` (100% pass rate).

---

## Development Roadmap

* **Phase 1:** Environment and Project Structure *(Completed)*
* **Phase 2:** Dataset Design and Collection for Resume-Job Matching *(Completed)*
* **Phase 3:** Resume Upload and Text Extraction *(Completed)*
* **Phase 4:** NLP Preprocessing and Information Extraction *(Completed)*
* **Phase 5:** TF-IDF & Cosine Similarity Baseline *(Completed)*
* **Phase 6:** Supervised SVM Classification *(Completed)*
* **Phase 7:** Supervised ANN / MLP Classification *(Completed)*
* **Phase 8:** LSTM / RNN-Based Sequence Modeling *(Completed)*
* **Phase 9:** Comprehensive Model Evaluation & Final Model Selection *(Completed)*
* **Phase 10:** Final Resume–Job Matching Engine *(Completed)*
* **Phase 11:** Recruiter / Candidate Dashboard & Frontend Integration *(Completed)*
* **Phase 12:** Production Deployment, Auth & Monitoring *(Future Step)*
