# Data Privacy, Security, and Governance Guidelines

## 1. Core Principles

Candidate resumes inherently contain personal and professional background information. Protecting applicant privacy and maintaining data security is a foundational requirement of this project.

---

## 2. Privacy Guidelines & Best Practices

1. **Git Isolation:**
   * Never commit real candidate resumes (PDFs, docs, images) to the Git repository.
   * Verify that `.gitignore` enforces exclusion of `data/resumes/*` and `data/jobs/*`.

2. **Synthetic Data for Development:**
   * Use curated synthetic profiles during local development, testing, algorithm tuning, and continuous integration.
   * Never use real-world applicant data for open-source demonstrations.

3. **Data Minimization:**
   * Do not extract or store non-essential Personally Identifiable Information (PII) such as national identification numbers, passport numbers, home addresses, dates of birth, or sensitive demographic attributes.
   * Focus extraction exclusively on professional competencies: skills, education level, domain experience, and projects.

4. **Credential Security:**
   * Never hardcode database connection strings, Supabase API keys, or JWT secrets in source code.
   * Always load credentials from `.env` files via environment variables (see `.env.example`).

5. **Storage Security & Access Control:**
   * Resume files stored in Supabase Storage buckets (`resumes/`) must be configured with private access policies.
   * Implement Row Level Security (RLS) policies in PostgreSQL so candidates can only view their own uploaded resumes and recruiters only access assigned requisitions.

6. **Anti-Scraping Commitment:**
   * Do not scrape professional networks (LinkedIn, Indeed, Naukri) or private portfolios.
   * Only use appropriately licensed or authorized datasets.
