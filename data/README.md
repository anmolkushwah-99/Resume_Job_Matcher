# Data Directory Overview & Architecture

This directory organizes all datasets, taxonomies, annotations, and database documentation for the **AI-Based Resume Screening and Job Matching System**.

---

## 1. Directory Structure

```
data/
├── raw/                      # Unmodified source data (Never overwritten)
│   ├── resumes/              # Raw resume profiles & text (resumes.csv)
│   ├── jobs/                 # Raw job requisitions & specs (jobs.csv)
│   └── raw_pairs.csv         # Labeled Resume + Job training pairs
│
├── processed/                # Normalized, tokenized, feature-engineered data (Generated in future phases)
│
├── taxonomy/                 # Domain vocabularies & ontologies
│   └── skills.csv            # 100 canonical skills, categories & aliases
│
├── ANNOTATION_GUIDELINES.md  # Standard operating rules for compatibility labeling
├── DATASET_CARD.md           # Dataset documentation, metadata & ethics disclosure
├── DATA_SPLIT_STRATEGY.md    # Candidate-level partitioning rules preventing data leakage
├── DATABASE_SCHEMA.md        # Supabase PostgreSQL relational schema & ER diagram
├── PRIVACY_NOTES.md          # PII protection, access control & governance guidelines
├── README.md                 # This overview file
└── supabase_schema.sql       # PostgreSQL DDL table definitions & indexes
```

---

## 2. Separation of ML Dataset vs Application Database

* **ML Training Datasets (`data/raw/` and `data/processed/`):**
  * Stored locally and tracked as reproducible tabular files.
  * Used for training, cross-validation, and benchmarking NLP/ML models (TF-IDF, SVM, ANN).

* **Application Database (Supabase PostgreSQL):**
  * Live relational database backing the Flask REST API.
  * Holds registered users, active job requisitions, candidate resume uploads, and realtime match inference scores.
  * Schema defined in [`supabase_schema.sql`](file:///d:/advance%20ai/Resume_Job_Matcher/data/supabase_schema.sql).

---

## 3. Data Flow Overview

```
Raw Resumes (data/raw/resumes/) ──┐
                                  ├──> Preprocessing & Extraction (Phase 3 & 4) ──> Processed Data (data/processed/)
Raw Jobs (data/raw/jobs/) ────────┘                                                          │
                                                                                             ▼
Skill Taxonomy (data/taxonomy/) ───────────────────────────────────────────────────> Feature Engineering & ML Models
                                                                                             │
                                                                                             ▼
Raw Pairs (data/raw/raw_pairs.csv) ──> Candidate-Level Split ────────────────────────> Model Benchmarking & Evaluation
```
