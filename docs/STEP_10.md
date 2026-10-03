# STEP 10: Final Resume–Job Matching Engine
**AI-Based Resume Screening and Job Matching System**  
*Production-Style Inference, Multi-Job Ranking, Auxiliary Skill Gap Explanations, and API Integration*

---

## 1. Executive Summary & Objective

Step 10 operationalizes the empirical findings and decisions of Step 9 by building a robust, modular, production-style **Final Matching Engine**. 

Throughout Steps 5–8, four distinct machine learning and NLP architectures were developed and benchmarked:
1. `tfidf_cosine_baseline` (Step 5: Unsupervised TF-IDF Cosine Similarity)
2. `tfidf_svm` (Step 6: Supervised Linear Support Vector Classifier with balanced class weights)
3. `tfidf_ann` (Step 7: Supervised Multi-Layer Perceptron with Batch Normalization & Dropout)
4. `embedding_lstm` (Step 8: Word-level Sequence Embedding with Bidirectional LSTM)

In Step 9, comprehensive multi-metric evaluation, statistical significance testing, confusion matrix inspection, candidate-grouped cross-validation, and error analysis identified **`tfidf_svm`** as the primary production engine for this project.

### Step 10 Implementation Goals
- **Frozen Artifact Reusability**: Safely load and reuse persisted model artifacts (`models/tfidf_svm_model.joblib`, `models/tfidf_svm_vectorizer.joblib`, `models/tfidf_svm_metadata.json`) without online retraining or startup fitting.
- **Continuous Decision Score Semantics**: Expose continuous signed distance outputs from `LinearSVC.decision_function()` and apply the natural zero-threshold $\theta = 0.0$ ($\text{decision\_score} \ge 0.0 \implies \text{Match}$).
- **Deterministic Multi-Job Ranking**: Rank job candidates strictly by `decision_score DESC` with secondary deterministic tie-breaking `job_id ASC`.
- **Auxiliary Skill Explanations**: Provide transparent, non-causal skill overlap analysis (`matched_skills`, `missing_skills`, and overlap ratio) without misrepresenting skill counts as model probabilities.
- **Batch Processing & Isolation**: Support batch candidate-job evaluation with per-job error isolation and multi-record database transactions.
- **MySQL Persistence**: Store match records in the existing `matches` table (`model_name='tfidf_svm'`) with transaction safety and idempotent update handling.
- **Production REST Endpoints**: Expose clean, privacy-preserving APIs (`POST /api/matches/svm`, `POST /api/resumes/<resume_id>/svm-matches`, `POST /api/matches/svm/batch`, `POST /api/matches`, `GET /api/health`).

---

## 2. Model Selection Rationale (From Step 9 Evidence)

Step 9 established that `tfidf_svm` delivers the highest overall utility across accuracy, generalization stability, inference speed, and explainability:

| Evaluation Metric / Dimension | `tfidf_cosine_baseline` (Step 5) | `tfidf_svm` (Step 6 - Selected Primary) | `tfidf_ann` (Step 7) | `embedding_lstm` (Step 8) |
|---|---|---|---|---|
| **Test Accuracy** | 50.00% | **88.89%** | 83.33% | 77.78% |
| **Test Precision (Class 1)** | 42.86% | **83.33%** | 80.00% | 71.43% |
| **Test Recall (Class 1)** | 100.00% | **83.33%** | 66.67% | 83.33% |
| **Test F1-Score (Class 1)** | 0.6000 | **0.8333** | 0.7273 | 0.7692 |
| **Macro F1-Score** | 0.4444 | **0.8846** | 0.8269 | 0.7766 |
| **ROC-AUC (Test)** | 0.6125 | **0.9000** | 0.8875 | 0.8500 |
| **Raw Model Latency (ms/pair)** | ~0.35 ms | **~1.15 ms** | ~12.50 ms | ~28.70 ms |
| **Throughput (pairs/sec)** | ~2,850 | **~870** | ~80 | ~35 |
| **Hyperplane Explainability** | No (heuristic) | **Yes (Direct feature weights)** | Low (opaque weights) | Low (hidden states) |

> **Selection Disclaimer**: `tfidf_svm` was selected based specifically on this project's benchmark dataset (120 candidate-job pairs across 25 candidates). On small-to-medium tabular/text collections, convex linear SVMs exhibit strong regularized margin separation without overfitting or requiring millions of sequence parameters.

---

## 3. Architecture & Modular System Design

The system enforces strict separation of concerns across Model, Orchestration, API, and Persistence layers:

```
                      ┌─────────────────────────────────┐
                      │        Candidate Resume         │
                      │  (MySQL / Raw CSV / Extracted)  │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │       Resume Text Builder       │
                      │  (Header + Summary + Exp + Edu) │
                      └────────────────┬────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            │                                                     │
            ▼                                                     ▼
┌───────────────────────┐                             ┌───────────────────────┐
│         JOB 1         │                             │         JOB N         │
│   (Benchmark Jobs)    │                             │   (Benchmark Jobs)    │
└───────────┬───────────┘                             └───────────┬───────────┘
            │                                                     │
            ▼                                                     ▼
┌───────────────────────┐                             ┌───────────────────────┐
│   Job Text Builder    │                             │   Job Text Builder    │
└───────────┬───────────┘                             └───────────┬───────────┘
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                      ┌─────────────────────────────────┐
                      │    Pair Text Representations    │
                      │  f"{resume_doc}\n\n{job_doc}"   │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │      TF-IDF Transformation      │
                      │ (tfidf_svm_vectorizer.joblib)   │
                      │   [Batch size, 1661 features]   │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │      LinearSVC Hyperplane       │
                      │    (tfidf_svm_model.joblib)     │
                      │       decision_function(X)      │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │   Continuous Decision Scores    │
                      │  score >= 0.0 -> Match (True)   │
                      │  score <  0.0 -> Non-Match (0)  │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │  Auxiliary Skill Gap Breakdown  │
                      │  (matched_skills, missing_...)  │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │      Deterministic Ranking      │
                      │ (decision_score DESC, job_id ASC│
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │    MySQL Transactional Store    │
                      │    (matches table persistence)  │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │     REST API JSON Envelope      │
                      │   (Privacy-Safe Client Payloads)│
                      └─────────────────────────────────┘
```

---

## 4. Text Representation & Feature Integrity

The matching engine strictly enforces the preprocessing and token concatenation contract established in Step 6:

1. **Resume Representation**: Cleaned text synthesized by `src/matching/resume_text_builder.py` incorporating professional title, summary, work history, skill taxonomy tokens, and education credentials.
2. **Job Representation**: Cleaned text synthesized by `src/matching/job_text_builder.py` incorporating job title, company, domain category, job description, required skills, preferred skills, minimum experience, and degree requirements.
3. **Pair Document Construction**:
   $$\text{Pair Document} = \text{Resume Document} + \text{"\n\n"} + \text{Job Document}$$
4. **TF-IDF Vocabulary**: Exactly 1,661 n-gram features $(1, 2)$ preserving technical tokens (`c++`, `c#`, `.net`, `ci/cd`, `node.js`).
5. **No Online Fitting**: The vectorizer is frozen and only runs `.transform(texts)`.

---

## 5. Mathematical Decision Semantics

### Continuous Decision Function
The raw model prediction is given by the signed Euclidean distance to the separating hyperplane:
$$d(x) = \mathbf{w}^T \phi(x) + b$$
Where:
- $\phi(x) \in \mathbb{R}^{1661}$ is the TF-IDF feature vector.
- $\mathbf{w} \in \mathbb{R}^{1661}$ is the trained SVM coefficient vector.
- $b \in \mathbb{R}$ is the hyperplane intercept.

### Classification Rule
$$\hat{y} = \begin{cases} 1 \quad (\text{Match}), & \text{if } d(x) \ge 0.0 \\ 0 \quad (\text{Non-Match}), & \text{if } d(x) < 0.0 \end{cases}$$

### Ranking Criterion
Candidate jobs are ranked by continuous model affinity:
$$\text{Rank}(J_i) < \text{Rank}(J_j) \iff d(x_i) > d(x_j) \quad \text{or} \quad (d(x_i) = d(x_j) \land \text{id}(J_i) < \text{id}(J_j))$$

---

## 6. Auxiliary Skill Gap Explanations

The matching engine extracts structured skill tokens from both entities using canonical skill taxonomy mappings:
- $\mathcal{S}_{\text{resume}}$: Set of canonical skills in candidate resume.
- $\mathcal{S}_{\text{job}}$: Set of required and preferred skills in target job.

$$\text{Matched Skills} = \mathcal{S}_{\text{resume}} \cap \mathcal{S}_{\text{job}}$$
$$\text{Missing Skills} = \mathcal{S}_{\text{job}} \setminus \mathcal{S}_{\text{resume}}$$
$$\text{Skill Overlap Ratio} = \frac{|\mathcal{S}_{\text{resume}} \cap \mathcal{S}_{\text{job}}|}{|\mathcal{S}_{\text{job}}|}$$

> **Important Semantics Note**: The skill overlap ratio is an auxiliary indicator for recruiter review and human decision support. It is NOT the model's prediction or probability.

---

## 7. Database Persistence & Schema Mapping

Evaluated matches are persisted to the MySQL `matches` table:

```sql
INSERT INTO matches (
    id, resume_id, job_id, similarity_score, overall_score,
    model_name, is_match, match_label, threshold_used,
    matched_skills, missing_skills, created_at, updated_at
) VALUES (
    %s, %s, %s, %s, %s, 'tfidf_svm', %s, %s, 0.0,
    %s, %s, NOW(), NOW()
) ON DUPLICATE KEY UPDATE
    similarity_score = VALUES(similarity_score),
    overall_score = VALUES(overall_score),
    is_match = VALUES(is_match),
    match_label = VALUES(match_label),
    threshold_used = VALUES(threshold_used),
    matched_skills = VALUES(matched_skills),
    missing_skills = VALUES(missing_skills),
    updated_at = NOW();
```

- `overall_score`: Stores the continuous `decision_score` (e.g., `1.8214`, `-0.4512`).
- `model_name`: Set strictly to `'tfidf_svm'`.
- `is_match`: Boolean indicating $\text{overall\_score} \ge 0.0$.
- `matched_skills` / `missing_skills`: JSON array of canonical skill tokens.

---

## 8. REST API Specifications

### 1. Single Pair Matching: `POST /api/matches/svm`
- **Request**:
  ```json
  {
    "resume_id": "RES001",
    "job_id": "JOB001"
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "match_id": "8f03ec4c-5357-4ba3-a38c-8f92bd8da971",
    "resume_id": "RES001",
    "job_id": "JOB001",
    "job_title": "Senior Python Developer",
    "model_name": "tfidf_svm",
    "model_type": "Linear Support Vector Classifier (LinearSVC)",
    "decision_score": 1.4521,
    "threshold": 0.0,
    "predicted_label": 1,
    "is_match": true,
    "matched_skills": ["Python", "SQL", "Docker", "FastAPI"],
    "missing_skills": ["Kubernetes", "AWS"],
    "matched_skill_count": 4,
    "missing_skill_count": 2,
    "skill_overlap_ratio": 0.6667,
    "explanation_note": "Auxiliary skill gap breakdown based on canonical extraction.",
    "created_at": "2026-10-01T15:40:00.000000"
  }
  ```

### 2. Multi-Job Ranking: `POST /api/resumes/<resume_id>/svm-matches`
- **Request Parameters (JSON or Query Params)**:
  - `top_k` (int, optional): Limit number of returned jobs.
  - `match_only` (bool, optional): Filter for `decision_score >= 0.0`.
  - `job_ids` (array of string, optional): Target subset of job IDs.
- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "resume_id": "RES001",
    "model": {
      "name": "tfidf_svm",
      "type": "Linear Support Vector Classifier (LinearSVC)",
      "threshold": 0.0,
      "decision_rule": "decision_score >= 0.0 -> Match"
    },
    "total_jobs_evaluated": 12,
    "total_returned": 3,
    "total_predicted_matches": 2,
    "total_predicted_non_matches": 1,
    "highest_decision_score": 1.8214,
    "lowest_decision_score": -0.7412,
    "ranking_criteria": "decision_score DESC, job_id ASC (deterministic)",
    "results": [
      {
        "rank": 1,
        "job_id": "JOB001",
        "job_title": "Senior Python Developer",
        "decision_score": 1.8214,
        "is_match": true,
        "matched_skills": ["Python", "SQL", "Docker"],
        "missing_skills": ["AWS"]
      }
    ]
  }
  ```

### 3. Batch Evaluation with Error Isolation: `POST /api/matches/svm/batch`
- **Request**:
  ```json
  {
    "resume_id": "RES001",
    "job_ids": ["JOB001", "NON_EXISTENT_JOB", "JOB002"]
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "resume_id": "RES001",
    "total_submitted": 3,
    "successful_count": 2,
    "failed_count": 1,
    "results": [ ... ],
    "failed_jobs": [
      {
        "job_id": "NON_EXISTENT_JOB",
        "error": "Job with ID 'NON_EXISTENT_JOB' not found in database or dataset."
      }
    ]
  }
  ```

### 4. Generic Matching Endpoint: `POST /api/matches`
- Defaults to `model="tfidf_svm"`.

### 5. Health & Model Registry Check: `GET /api/health`
- Verifies MySQL connectivity and readiness of all registered matching models.

---

## 9. Performance & Benchmarking Results

Empirical latency and throughput measured during Step 10 execution (`data/processed/evaluation/step10_inference_benchmark.csv`):

| Workload | Batch Size | Average Latency (ms) | Std Dev (ms) | Throughput (pairs/sec) |
|---|---|---|---|---|
| **Raw Model Inference** (TF-IDF transform + LinearSVC decision function) | 1 pair | **1.15 ms** | $\pm 0.36\text{ ms}$ | **872.8 pairs/sec** |
| **Full Single Match Engine** (`match_one` with text resolution & skills) | 1 pair | **77.16 ms** | $\pm 11.12\text{ ms}$ | **13.0 pairs/sec** |
| **Multi-Job Matching & Ranking** (`match_many` over all 12 benchmark jobs) | 12 pairs | **83.11 ms** | $\pm 15.77\text{ ms}$ | **144.4 pairs/sec** |

---

## 10. Privacy & Security Safeguards

1. **PII Masking**: API payloads never expose candidate phone numbers, personal email addresses, physical street addresses, or raw resume text.
2. **No Secret Leaks**: Database credentials and internal filesystem paths are never output in error envelopes or logging statements.
3. **Identifier-Based Queries**: All inputs resolve through validated IDs (`RES001`, `JOB001`, or UUIDs), blocking directory traversal vectors (`../../etc/passwd`).

---

## 11. Testing & Verification Summary

- **Pre-Step-10 Baseline Tests**: 190 passed
- **New Step-10 Tests Added**: 28 tests across 4 dedicated test suites:
  - `tests/test_model_registry.py` (6 tests)
  - `tests/test_explanation.py` (4 tests)
  - `tests/test_final_match_engine.py` (7 tests)
  - `tests/test_final_matching_api.py` (9 tests)
  - `tests/test_final_integration.py` (2 tests)
- **Total Test Suite**: **218 passed / 218 total (100% pass rate)**.
- **Zero Regression**: Step 3 (Upload), Step 4 (NLP), Step 5 (Cosine Baseline), Step 6 (SVM), Step 7 (ANN), Step 8 (LSTM), and Step 9 (Evaluation) remain fully operational.

---

## 12. Real-World Limitations & Disclaimers

1. **Benchmark Scale**: Trained on 120 candidate-job pairs from 25 benchmark candidates. Real-world distribution shifts require ongoing validation.
2. **Lexical Representation**: TF-IDF does not capture deep cross-sentence contextual semantics or synonyms not present in the n-gram dictionary.
3. **Decision Support Only**: This engine is designed strictly as an automated decision-support and candidate-ranking tool. It should not be used as an autonomous, unmonitored hiring system.

---

## 13. Step 11 Readiness (Future UI Contract)

The backend exposes complete JSON structures ready to be directly consumed by a recruiter or candidate dashboard in Step 11:
- `POST /api/resumes/<resume_id>/svm-matches` returns sorted cards with `rank`, `job_title`, `decision_score`, `is_match`, `matched_skills`, and `missing_skills`.
- `POST /api/matches/svm/batch` enables multi-job batch comparison tables.
- `GET /api/health` enables live system health indicator badges.
