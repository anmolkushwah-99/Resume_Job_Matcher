# AI-Based Resume Screening and Job Matching System

## Project Description

The **AI-Based Resume Screening and Job Matching System** is an intelligent natural language processing (NLP) and machine learning (ML) platform designed to automate and streamline the recruitment screening process. The system analyzes candidate resumes (PDF format), extracts structured information and key competencies, compares candidates against job descriptions, identifies matched and missing skill gaps, and produces explainable compatibility scores and candidate rankings.

---

## Current Phase

**Phase 1 — Project Foundation and Environment Setup**

> **Note:** In this foundational phase, only the project directory structure, isolated virtual environment, core dependencies, and NLP baseline resources are configured and verified. Machine learning models, extraction pipelines, matching algorithms, and user interfaces are **NOT** implemented yet and will be built incrementally in subsequent phases.

---

## Planned Technology Stack

* **Programming Language:** Python 3.11+
* **Backend API:** Flask, Flask-CORS
* **Data Processing & Manipulation:** Pandas, NumPy
* **Machine Learning & Preprocessing:** Scikit-learn
* **Natural Language Processing (NLP):** NLTK, spaCy (`en_core_web_sm`)
* **Document & PDF Parsing:** pypdf
* **Data Visualization & Analytics:** Matplotlib, Seaborn
* **Interactive Experimentation:** Jupyter Notebooks

---

## Planned Machine Learning Approaches (Future Components)

The following modeling strategies will be explored, implemented, and compared in future phases:

1. **TF-IDF + Cosine Similarity (Baseline):** Statistical text similarity and term-frequency matching for baseline scoring.
2. **Support Vector Machine (SVM):** Supervised classification of resume-job fit categories.
3. **Artificial Neural Network (ANN):** Multi-layer perceptron (MLP) trained on extracted feature vectors and skill embeddings.
4. **Recurrent Neural Network / LSTM (Optional / Experimental):** Sequential modeling for contextual resume evaluation.

---

## Planned Features

* **Resume PDF Upload:** Secure upload handling for single and batch candidate resumes.
* **Resume Text Extraction:** Parsing raw text from PDF documents using `pypdf`.
* **NLP Preprocessing Pipeline:** Tokenization, stop-word removal, lemmatization, and text normalization.
* **Skill & Entity Extraction:** Named entity recognition and rule-based extraction for candidate skills, education, and experience.
* **Job Description Analysis:** Parsing and structured representation of job requirements and target qualifications.
* **Resume-Job Compatibility Analysis:** Automated semantic and keyword-based similarity calculation.
* **Match Score Calculation:** Normalized overall match index between candidate and job profile.
* **Skill Gap Analysis:** Clear breakdown of matched skills vs. missing candidate skills.
* **Candidate Ranking:** Sorting and filtering candidate profiles for recruiters.
* **Explainable Results:** Transparent, human-understandable justification for match ratings.

---

## Project Architecture

```
User (Recruiter / Candidate)
         │
         ▼
    Resume PDF
         │
         ▼
[Future] PDF Text Extraction (pypdf)
         │
         ▼
[Future] NLP Preprocessing (NLTK & spaCy)
         │
         ▼
[Future] Feature & Skill Extraction
         │
         ▼
[Future] Matching & Machine Learning Models (TF-IDF / SVM / ANN / LSTM)
         │
         ▼
[Future] Compatibility Analysis & Skill Gap Detection
         │
         ▼
[Future] Results & Explainable Match Score
```

*(All downstream processing modules marked `[Future]` are scheduled for implementation in subsequent project phases.)*

---

## Project Structure

```
Resume_Job_Matcher/
│
├── app/                  # Flask backend application and templates (Future UI/API)
│   ├── templates/        # HTML templates
│   └── static/           # Static assets (CSS, JS, images)
│
├── data/                 # Project data storage
│   ├── resumes/          # Raw candidate resume PDFs
│   ├── jobs/             # Job descriptions and postings
│   └── processed/        # Cleaned datasets and extracted features
│
├── models/               # Saved trained ML models (.pkl, .keras, etc.)
│
├── notebooks/            # Jupyter notebooks for EDA and model experiments
│
├── src/                  # Core source code modules
│   ├── preprocessing/    # Text cleaning, tokenization, lemmatization
│   ├── extraction/       # PDF parsing and skill extraction
│   ├── matching/         # Similarity metrics and scoring algorithms
│   └── models/           # Model definitions (SVM, ANN, LSTM)
│
├── tests/                # Automated unit and integration tests
│
├── .gitignore            # Git exclusion rules for virtualenv, models, and data
├── README.md             # Project documentation and roadmap
├── requirements.txt      # Pinned project dependencies
└── test_setup.py         # Environment and dependency verification script
```

---

## Development Roadmap & Future Phases

* **Phase 1:** Environment and Project Structure *(Completed)*
* **Phase 2:** Dataset Design and Collection for Resume-Job Matching
* **Phase 3:** Resume and Job Text Extraction
* **Phase 4:** NLP Preprocessing Pipeline
* **Phase 5:** TF-IDF & Cosine Similarity Baseline
* **Phase 6:** SVM Model Implementation
* **Phase 7:** ANN Model Implementation
* **Phase 8:** Optional LSTM Experiment
* **Phase 9:** Model Evaluation, Cross-Validation & Benchmark Comparison
* **Phase 10:** Flask Backend API Development
* **Phase 11:** Interactive Web Dashboard & Final System Integration
