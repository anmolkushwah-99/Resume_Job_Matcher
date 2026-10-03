"""
Match Service Module.
Orchestrates resume-job matching using the TF-IDF Baseline, persists scoring results
to MySQL 'matches' table transactionally, and formats privacy-safe responses.
"""

import os
import uuid
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from src.matching.resume_text_builder import build_resume_document, build_resume_document_from_db
from src.matching.job_text_builder import build_job_document, build_job_document_from_db
from src.matching.tfidf_matcher import TFIDFMatcher
from src.database.mysql_client import get_db_cursor

logger = logging.getLogger(__name__)


class MatchServiceError(Exception):
    """Base exception for match service errors."""
    pass


class ResumeNotFoundError(MatchServiceError):
    """Raised when the specified resume is not found."""
    pass


class JobNotFoundError(MatchServiceError):
    """Raised when the specified job is not found."""
    pass


def get_resume_skills_from_db_or_nlp(resume_id: str) -> List[str]:
    """
    Retrieves the list of extracted skill names for a given resume_id.
    """
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT s.skill_name
                FROM resume_skills rs
                JOIN skills s ON rs.skill_id = s.id
                WHERE rs.resume_id = %s
                """,
                (resume_id.strip(),)
            )
            rows = cur.fetchall()
            if rows:
                return [r["skill_name"] for r in rows]
    except Exception as e:
        logger.debug("Could not fetch resume skills from DB for %s: %s", resume_id, str(e))

    # Check raw CSV fallback
    resumes_csv = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "resumes" / "resumes.csv"
    if resumes_csv.exists():
        try:
            df = pd.read_csv(resumes_csv)
            row = df[df["resume_id"] == resume_id]
            if not row.empty and pd.notna(row.iloc[0].get("skills")):
                return [s.strip() for s in str(row.iloc[0]["skills"]).split(";") if s.strip()]
        except Exception:
            pass

    return []


def get_job_skills_from_db_or_csv(job_id: str) -> List[str]:
    """
    Retrieves the list of required and preferred skills for a job.
    """
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT s.skill_name
                FROM job_skills js
                JOIN skills s ON js.skill_id = s.id
                WHERE js.job_id = %s
                """,
                (job_id.strip(),)
            )
            rows = cur.fetchall()
            if rows:
                return [r["skill_name"] for r in rows]
    except Exception as e:
        logger.debug("Could not fetch job skills from DB for %s: %s", job_id, str(e))

    # Check raw CSV fallback
    jobs_csv = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "jobs" / "jobs.csv"
    if jobs_csv.exists():
        try:
            df = pd.read_csv(jobs_csv)
            row = df[df["job_id"] == job_id]
            if not row.empty:
                req = str(row.iloc[0].get("required_skills", ""))
                pref = str(row.iloc[0].get("preferred_skills", ""))
                skills = []
                for chunk in [req, pref]:
                    if chunk and pd.notna(chunk):
                        skills.extend([s.strip() for s in chunk.split(";") if s.strip()])
                return list(dict.fromkeys(skills))
        except Exception:
            pass

    return []


def get_resume_text(resume_id: str) -> str:
    """
    Gets normalized resume text from MySQL or raw fallback.
    """
    doc = build_resume_document_from_db(resume_id)
    if doc:
        return doc

    # Fallback to resumes.csv
    resumes_csv = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "resumes" / "resumes.csv"
    if resumes_csv.exists():
        df = pd.read_csv(resumes_csv)
        row = df[df["resume_id"] == resume_id]
        if not row.empty:
            return build_resume_document(row.iloc[0].to_dict())

    raise ResumeNotFoundError(f"Resume with ID '{resume_id}' not found in database or dataset.")


def get_job_text(job_id: str) -> str:
    """
    Gets normalized job text from MySQL or raw fallback.
    """
    doc = build_job_document_from_db(job_id)
    if doc:
        return doc

    # Fallback to jobs.csv
    jobs_csv = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "jobs" / "jobs.csv"
    if jobs_csv.exists():
        df = pd.read_csv(jobs_csv)
        row = df[df["job_id"] == job_id]
        if not row.empty:
            return build_job_document(row.iloc[0].to_dict())

    raise JobNotFoundError(f"Job with ID '{job_id}' not found in database or dataset.")


def calculate_and_store_match(
    resume_id: str,
    job_id: str,
    model_name: str = "tfidf_cosine_baseline",
    vectorizer: Optional[TFIDFMatcher] = None,
    persist: bool = True
) -> Dict[str, Any]:
    """
    Calculates TF-IDF Cosine Similarity for a single (resume_id, job_id) pair,
    persists the result to MySQL 'matches' table idempotently, and returns a
    privacy-safe structured summary.

    :param resume_id: Identifier of the candidate resume.
    :param job_id: Identifier of the target benchmark job.
    :param model_name: Name of the matching baseline model.
    :param vectorizer: Optional pre-fitted TFIDFMatcher instance.
    :param persist: Whether to persist the match score to MySQL.
    :return: Structured match result dictionary.
    """
    if not resume_id or not str(resume_id).strip():
        raise ValueError("resume_id parameter is required.")
    if not job_id or not str(job_id).strip():
        raise ValueError("job_id parameter is required.")

    resume_id = str(resume_id).strip()
    job_id = str(job_id).strip()

    # 1. Fetch resume and job documents
    resume_doc = get_resume_text(resume_id)
    job_doc = get_job_text(job_id)

    # 2. Vectorize and compute similarity
    matcher = vectorizer or TFIDFMatcher()
    similarity = matcher.compute_similarity(resume_doc, job_doc)
    sim_score = float(round(similarity, 4))
    sim_percentage = float(round(similarity * 100, 2))

    # 3. Detect shared skills for explainable metadata
    res_skills = set(get_resume_skills_from_db_or_nlp(resume_id))
    jb_skills = set(get_job_skills_from_db_or_csv(job_id))
    shared_skills = sorted(list(res_skills.intersection(jb_skills)))

    # 4. Persist result to MySQL matches table
    if persist:
        try:
            with get_db_cursor(commit=True) as cur:
                # Check if match record already exists
                cur.execute(
                    """
                    SELECT id FROM matches
                    WHERE resume_id = %s AND job_id = %s AND model_name = %s
                    LIMIT 1
                    """,
                    (resume_id, job_id, model_name)
                )
                existing = cur.fetchone()

                if existing:
                    # Update existing record
                    cur.execute(
                        """
                        UPDATE matches
                        SET similarity_score = %s, created_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (sim_score, existing["id"])
                    )
                else:
                    # Insert new match record
                    match_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO matches (id, resume_id, job_id, similarity_score, model_name)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (match_id, resume_id, job_id, sim_score, model_name)
                    )
        except Exception as e:
            # If foreign key fails because resume/job only existed in CSV and not in DB,
            # log warning but do not crash match calculation
            logger.warning("Could not persist match to MySQL: %s", str(e))

    return {
        "success": True,
        "resume_id": resume_id,
        "job_id": job_id,
        "model": model_name,
        "similarity_score": sim_score,
        "similarity_percentage": sim_percentage,
        "shared_skills": shared_skills,
    }


def calculate_matches_for_resume(
    resume_id: str,
    model_name: str = "tfidf_cosine_baseline",
    limit: Optional[int] = None,
    persist: bool = True
) -> Dict[str, Any]:
    """
    Computes TF-IDF cosine similarity scores for a given resume against all available
    benchmark jobs, ordered from highest to lowest similarity.

    :param resume_id: Identifier of the resume.
    :param model_name: Baseline model identifier.
    :param limit: Optional maximum number of matches to return.
    :param persist: Whether to persist each match record.
    :return: Dictionary containing ranked job matches.
    """
    if not resume_id or not str(resume_id).strip():
        raise ValueError("resume_id parameter is required.")

    resume_id = str(resume_id).strip()
    resume_doc = get_resume_text(resume_id)

    # 1. Fetch all available jobs from DB or jobs.csv
    jobs_data: List[Dict[str, Any]] = []

    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT id, job_title, company, job_description, job_category,
                       minimum_experience, education_requirement
                FROM jobs
                ORDER BY id ASC
                """
            )
            rows = cur.fetchall()
            for r in rows:
                jobs_data.append({
                    "job_id": r["id"],
                    "job_title": r["job_title"],
                    "company": r["company"],
                    "doc": build_job_document(dict(r))
                })
    except Exception:
        pass

    if not jobs_data:
        jobs_csv = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "jobs" / "jobs.csv"
        if jobs_csv.exists():
            df = pd.read_csv(jobs_csv)
            for _, r in df.iterrows():
                jobs_data.append({
                    "job_id": r["job_id"],
                    "job_title": r["job_title"],
                    "company": r["company"],
                    "doc": build_job_document(r.to_dict())
                })

    if not jobs_data:
        raise JobNotFoundError("No jobs available to match against.")

    # 2. Build shared corpus of resume + all jobs
    corpus = [resume_doc] + [j["doc"] for j in jobs_data]
    matcher = TFIDFMatcher()
    matcher.fit(corpus)

    res_vec = matcher.transform([resume_doc])
    job_vecs = matcher.transform([j["doc"] for j in jobs_data])

    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(res_vec, job_vecs)[0]

    matches_list = []
    for idx, job_info in enumerate(jobs_data):
        sim = float(round(scores[idx], 4))
        matches_list.append({
            "job_id": job_info["job_id"],
            "job_title": job_info["job_title"],
            "company": job_info["company"],
            "similarity_score": sim,
            "similarity_percentage": float(round(sim * 100, 2)),
        })

    # Sort descending by similarity score
    matches_list.sort(key=lambda x: x["similarity_score"], reverse=True)

    if limit is not None and limit > 0:
        matches_list = matches_list[:limit]

    # Optionally persist
    if persist:
        for m in matches_list:
            try:
                calculate_and_store_match(
                    resume_id=resume_id,
                    job_id=m["job_id"],
                    model_name=model_name,
                    vectorizer=matcher,
                    persist=True
                )
            except Exception as e:
                logger.debug("Failed to persist match (%s, %s): %s", resume_id, m["job_id"], str(e))

    return {
        "success": True,
        "resume_id": resume_id,
        "model": model_name,
        "total_jobs_evaluated": len(jobs_data),
        "matches": matches_list,
    }
