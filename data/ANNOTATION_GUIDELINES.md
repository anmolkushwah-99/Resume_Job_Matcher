# Annotation Guidelines: Resume-Job Compatibility Classification

## 1. Overview and Purpose

This document outlines the standard operational guidelines for annotating resume and job description pairs (`raw_pairs.csv`) within the **AI-Based Resume Screening and Job Matching System**.

The primary objective is to establish an objective, reproducible labeling framework for binary compatibility classification:
* **`1` = Match (Qualified / Strong Alignment)**
* **`0` = Poor / No Match (Insufficient Alignment)**

---

## 2. Label Definitions

### 2.1 Positive Match (`1`)
A candidate-job pair is assigned a label of `1` when:
1. **Core Required Skills:** The candidate possesses a substantial majority (typically ≥ 70–80%) of the explicitly mandatory required skills.
2. **Domain/Role Alignment:** The candidate's background, past projects, and primary technical domain match the target job category.
3. **Experience Threshold:** The candidate meets or is within a reasonable buffer (e.g., within 1 year or demonstrated equivalent project depth) of the stated minimum experience.
4. **Educational Alignment:** The candidate's degree and field of study are reasonably relevant to the position (e.g., CS, IT, Data Science, Statistics for software/data roles).

### 2.2 Poor / No Match (`0`)
A candidate-job pair is assigned a label of `0` when:
1. **Critical Skill Gap:** One or more essential core required skills are missing (e.g., a candidate with only React applying for a pure Java Spring Boot backend role).
2. **Domain Disconnect:** The candidate's career trajectory is in an unrelated discipline (e.g., Graphic Design or HR applying for Machine Learning Engineering).
3. **Severe Experience Mismatch:** A candidate with 1 year of junior experience applying for a Principal/Senior Architect role requiring 5+ years of distributed systems leadership.
4. **Incompatible Education:** Role demands a quantitative/engineering foundation and candidate has an unrelated non-technical background without compensating technical experience.

---

## 3. Required vs. Preferred Skills Treatment

* **Required Skills (`required_skills`):** Mandatory qualifications. Missing required skills directly lowers compatibility and usually disqualifies the candidate (`0`), unless the candidate has direct equivalent expertise.
* **Preferred / Nice-to-Have Skills (`preferred_skills`):** Bonus qualifications. The presence of preferred skills bolsters a match, but their absence **must not** cause a negative label (`0`) if all required skills are fulfilled.

---

## 4. Evaluation Criteria Hierarchy

Annotators should evaluate candidate-job compatibility along four dimensions:

```
┌────────────────────────────────────────────────────────┐
│ 1. Core Technical Skills & Tooling (Weight: ~50%)     │
├────────────────────────────────────────────────────────┤
│ 2. Domain & Job Category Relevance (Weight: ~25%)      │
├────────────────────────────────────────────────────────┤
│ 3. Relevant Years of Experience (Weight: ~15%)        │
├────────────────────────────────────────────────────────┤
│ 4. Education & Certifications (Weight: ~10%)          │
└────────────────────────────────────────────────────────┘
```

---

## 5. Handling Missing or Incomplete Information

* **Missing Education:** If education is omitted but the candidate demonstrates strong verifiable skills and adequate experience, do not penalize heavily.
* **Missing Certifications:** Certifications are supplementary. Never assign `0` solely due to lack of certifications unless explicitly mandated by regulatory standards.
* **Ambiguous Experience Years:** Look at graduation dates and project timelines. If unclear, infer junior level (1–2 years).

---

## 6. Concrete Pair Examples

### 6.1 Positive Pair Examples (`1`)

* **Pair Example A:**
  * **Resume:** 3 years experience, Python, Flask, PostgreSQL, Docker, REST API, Git.
  * **Job:** Python Backend Developer (Req: Python, Flask, PostgreSQL, REST API; Min Exp: 2 yrs).
  * **Decision:** `1` (Full skill overlap, experience satisfied).

* **Pair Example B:**
  * **Resume:** Master's in Data Science, 5 years exp, Python, Scikit-learn, PyTorch, Pandas, NumPy, NLP.
  * **Job:** Machine Learning Engineer (Req: Python, ML, Scikit-learn, Pandas, NumPy; Min Exp: 2 yrs).
  * **Decision:** `1` (Exceeds requirements, domain match).

### 6.2 Negative Pair Examples (`0`)

* **Pair Example C:**
  * **Resume:** Frontend Developer with React, JavaScript, HTML, CSS, Tailwind CSS.
  * **Job:** Java Enterprise Backend Developer (Req: Java, Spring Boot, MySQL, REST API).
  * **Decision:** `0` (Zero required backend tech overlap).

* **Pair Example D:**
  * **Resume:** HR Specialist with Excel and Campus Recruitment background.
  * **Job:** Python Backend Developer.
  * **Decision:** `0` (Completely unrelated domain).

---

## 7. Sources of Ambiguity and Subjectivity

1. **Adjacent Technologies:** (e.g., FastAPI vs Flask, PyTorch vs TensorFlow). A candidate strong in FastAPI is often capable of Flask, but strict keyword matchers might miss this. Annotators should consider transferable skills when assigning ground truth.
2. **Seniority vs. Skill Overlap:** A junior candidate may have all keywords (e.g., Python, SQL, Spark) but only 1 year experience against a 4-year requirement. For conservative screening, this is labeled `0`.
3. **Subjectivity Acknowledgement:** Human recruiter evaluations naturally exhibit minor variance (inter-annotator variance). Clear guidelines and structured rubric minimize bias.
