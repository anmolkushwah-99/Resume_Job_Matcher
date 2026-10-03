# Supabase PostgreSQL Database Architecture & Schema Specification

## 1. Overview

The application database for the **AI-Based Resume Screening and Job Matching System** is built on **Supabase PostgreSQL**. It manages live production data (user accounts, uploaded resumes, job requisitions, extracted skill associations, and calculated match results) independently from the offline ML training datasets stored in `data/raw/`.

---

## 2. Entity Relationship (ER) Diagram

```mermaid
erDiagram
    USERS ||--o{ RESUMES : "uploads (candidate)"
    USERS ||--o{ JOBS : "posts (recruiter)"
    RESUMES ||--o{ RESUME_SKILLS : "contains"
    SKILLS ||--o{ RESUME_SKILLS : "classified in"
    JOBS ||--o{ JOB_SKILLS : "specifies"
    SKILLS ||--o{ JOB_SKILLS : "categorized in"
    RESUMES ||--o{ MATCHES : "evaluated in"
    JOBS ||--o{ MATCHES : "matched against"

    USERS {
        uuid id PK
        varchar name
        varchar email UK
        varchar role "candidate | recruiter | admin"
        timestamp with_time_zone created_at
    }

    RESUMES {
        uuid id PK
        uuid user_id FK
        varchar filename
        varchar file_path "Supabase storage path"
        text extracted_text
        varchar education
        varchar degree
        varchar field_of_study
        numeric experience_years
        timestamp with_time_zone uploaded_at
    }

    JOBS {
        uuid id PK
        uuid recruiter_id FK
        varchar job_title
        varchar company
        text job_description
        varchar job_category
        numeric minimum_experience
        varchar education_requirement
        timestamp with_time_zone created_at
    }

    SKILLS {
        uuid id PK
        varchar skill_name UK
        varchar category
        text aliases
    }

    RESUME_SKILLS {
        uuid resume_id PK, FK
        uuid skill_id PK, FK
    }

    JOB_SKILLS {
        uuid job_id PK, FK
        uuid skill_id PK, FK
        boolean is_required "true = required, false = preferred"
    }

    MATCHES {
        uuid id PK
        uuid resume_id FK
        uuid job_id FK
        numeric similarity_score
        numeric skill_score
        numeric overall_score
        varchar model_name
        timestamp with_time_zone created_at
    }
```

---

## 3. Detailed Table Specifications

### 3.1 `users`
Represents registered system actors.
* `id` (`UUID`, PK, default `gen_random_uuid()`): Unique identifier.
* `name` (`VARCHAR(255)`, NOT NULL): Full user name.
* `email` (`VARCHAR(255)`, UNIQUE, NOT NULL): Contact email address.
* `role` (`VARCHAR(50)`, NOT NULL, CHECK `role IN ('candidate', 'recruiter', 'admin')`): System access role.
* `created_at` (`TIMESTAMPTZ`, default `now()`): Creation timestamp.

### 3.2 `resumes`
Stores candidate resume metadata, parsed text, and storage pointers.
* `id` (`UUID`, PK, default `gen_random_uuid()`): Unique resume record ID.
* `user_id` (`UUID`, FK -> `users.id`, ON DELETE CASCADE): Uploading candidate.
* `filename` (`VARCHAR(255)`, NOT NULL): Original uploaded PDF filename.
* `file_path` (`TEXT`, NOT NULL): Object storage path inside Supabase Storage bucket (`resumes/`).
* `extracted_text` (`TEXT`, NULL): Raw parsed text from PDF.
* `education` (`VARCHAR(100)`): Level of education (e.g., B.Tech, M.S.).
* `degree` (`VARCHAR(100)`): Degree classification (e.g., Bachelor, Master).
* `field_of_study` (`VARCHAR(150)`): Academic major (e.g., Computer Science).
* `experience_years` (`NUMERIC(4,1)`, default `0.0`): Total years of professional experience.
* `uploaded_at` (`TIMESTAMPTZ`, default `now()`): Upload timestamp.

### 3.3 `jobs`
Stores job postings and requirement specifications.
* `id` (`UUID`, PK, default `gen_random_uuid()`): Unique job posting ID.
* `recruiter_id` (`UUID`, FK -> `users.id`, ON DELETE SET NULL): Posting recruiter/admin.
* `job_title` (`VARCHAR(255)`, NOT NULL): Position title.
* `company` (`VARCHAR(255)`, NOT NULL): Hiring company name.
* `job_description` (`TEXT`, NOT NULL): Full position overview and duties.
* `job_category` (`VARCHAR(100)`, NOT NULL): Technical domain.
* `minimum_experience` (`NUMERIC(4,1)`, default `0.0`): Minimum experience threshold in years.
* `education_requirement` (`VARCHAR(255)`): Stated education prerequisite.
* `created_at` (`TIMESTAMPTZ`, default `now()`): Posting creation timestamp.

### 3.4 `skills`
Master catalog of recognized technical and domain competencies.
* `id` (`UUID`, PK, default `gen_random_uuid()`): Unique skill identifier.
* `skill_name` (`VARCHAR(100)`, UNIQUE, NOT NULL): Canonical skill name.
* `category` (`VARCHAR(100)`, NOT NULL): Taxonomy classification.
* `aliases` (`TEXT`): Semicolon-delimited aliases and common variants.

### 3.5 `resume_skills` (Many-to-Many)
Links resumes to detected or claimed skills.
* `resume_id` (`UUID`, FK -> `resumes.id`, ON DELETE CASCADE)
* `skill_id` (`UUID`, FK -> `skills.id`, ON DELETE CASCADE)
* Composite Primary Key: `(resume_id, skill_id)`.

### 3.6 `job_skills` (Many-to-Many)
Links job descriptions to required and preferred skills.
* `job_id` (`UUID`, FK -> `jobs.id`, ON DELETE CASCADE)
* `skill_id` (`UUID`, FK -> `skills.id`, ON DELETE CASCADE)
* `is_required` (`BOOLEAN`, NOT NULL, default `true`): Flag differentiating mandatory vs preferred qualifications.
* Composite Primary Key: `(job_id, skill_id)`.

### 3.7 `matches`
Stores compatibility inference records generated by matching models.
* `id` (`UUID`, PK, default `gen_random_uuid()`)
* `resume_id` (`UUID`, FK -> `resumes.id`, ON DELETE CASCADE)
* `job_id` (`UUID`, FK -> `jobs.id`, ON DELETE CASCADE)
* `similarity_score` (`NUMERIC(5,4)`): Textual semantic similarity (0.0000 - 1.0000).
* `skill_score` (`NUMERIC(5,4)`): Jaccard/weighted skill overlap ratio (0.0000 - 1.0000).
* `overall_score` (`NUMERIC(5,4)`): Composite ranking score (0.0000 - 1.0000).
* `model_name` (`VARCHAR(100)`): Inference model tag (e.g., `tfidf_cosine_v1`, `svm_classifier_v1`, `ann_v1`).
* `created_at` (`TIMESTAMPTZ`, default `now()`)

---

## 4. Supabase Storage Bucket Design

* **Bucket Name:** `resumes`
* **Access Policy:** Private (authenticated access only).
* **Storage Path Pattern:** `resumes/{user_id}/{resume_id}.pdf`
* **Retention:** PDFs are stored in object storage; database stores only the `file_path` reference string.
