"""
ANN Inference and Prediction Service Module.
Orchestrates single-pair and multi-job predictions using the trained Keras ANN classifier,
persists prediction results to the MySQL 'matches' table, and formats privacy-safe responses.
"""

import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from src.models.ann_classifier import ANNResumeJobClassifier
from src.matching.match_service import (
    get_resume_text,
    get_job_text,
    get_resume_skills_from_db_or_nlp,
    get_job_skills_from_db_or_csv,
    ResumeNotFoundError,
    JobNotFoundError,
    MatchServiceError,
)
from src.database.mysql_client import get_db_cursor
from src.matching.job_text_builder import build_job_document

logger = logging.getLogger(__name__)

# Cached ANN classifier instance for fast inference
_cached_ann_classifier: Optional[ANNResumeJobClassifier] = None


def get_default_ann_model_dir() -> Path:
    """Returns the default directory for trained ANN artifacts."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    dir_1 = base_dir / "models"
    dir_2 = base_dir / "data" / "processed" / "models"
    if (dir_1 / "tfidf_ann_model.keras").exists():
        return dir_1
    if (dir_2 / "tfidf_ann_model.keras").exists():
        return dir_2
    return dir_1


def get_ann_classifier(model_dir: Optional[Path] = None) -> ANNResumeJobClassifier:
    """
    Returns the loaded ANN classifier instance from disk (cached in-memory).
    """
    global _cached_ann_classifier
    if _cached_ann_classifier is not None and _cached_ann_classifier.is_trained:
        return _cached_ann_classifier

    if model_dir is None:
        model_dir = get_default_ann_model_dir()

    if not (model_path := Path(model_dir) / "tfidf_ann_model.keras").exists():
        logger.info("Trained ANN model not found on disk at %s. Training default benchmark model...", model_path)
        from src.evaluate_ann import run_ann_evaluation
        run_ann_evaluation(model_dir=model_dir)

    _cached_ann_classifier = ANNResumeJobClassifier.load(model_dir)
    return _cached_ann_classifier


def set_cached_ann_classifier(classifier: Optional[ANNResumeJobClassifier]) -> None:
    """Sets or overrides the in-memory cached ANN classifier."""
    global _cached_ann_classifier
    _cached_ann_classifier = classifier


def predict_and_store_ann_match(
    resume_id: str,
    job_id: str,
    model_name: str = "tfidf_ann",
    model_dir: Optional[Path] = None,
    persist: bool = True,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Performs supervised binary classification for a given (resume_id, job_id) pair
    using the trained TF-IDF Keras ANN classifier, persists the score to MySQL, and
    returns a privacy-safe result summary.

    :param resume_id: Candidate resume identifier.
    :param job_id: Benchmark job identifier.
    :param model_name: Identifier for the model record in MySQL.
    :param model_dir: Optional custom directory for model artifacts.
    :param persist: Whether to persist the match record into MySQL.
    :param threshold: Optional decision threshold override.
    :return: Structured prediction response.
    """
    if not resume_id or not str(resume_id).strip():
        raise ValueError("resume_id parameter is required.")
    if not job_id or not str(job_id).strip():
        raise ValueError("job_id parameter is required.")

    resume_id = str(resume_id).strip()
    job_id = str(job_id).strip()

    # 1. Fetch resume and job texts
    resume_doc = get_resume_text(resume_id)
    job_doc = get_job_text(job_id)

    # 2. Combine into pair representation
    pair_text = resume_doc + "\n\n" + job_doc

    # 3. Predict using trained ANN
    classifier = get_ann_classifier(model_dir=model_dir)
    t = threshold if threshold is not None else classifier.threshold
    scores = classifier.predict_score([pair_text])
    ann_score = float(round(scores[0], 4))
    pred_class = int(1 if ann_score >= t else 0)

    # 4. Extract shared skills for explainable metadata
    res_skills = set(get_resume_skills_from_db_or_nlp(resume_id))
    jb_skills = set(get_job_skills_from_db_or_csv(job_id))
    shared_skills = sorted(list(res_skills.intersection(jb_skills)))

    # 5. Persist to MySQL matches table
    if persist:
        try:
            with get_db_cursor(commit=True) as cur:
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
                    cur.execute(
                        """
                        UPDATE matches
                        SET overall_score = %s, created_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (ann_score, existing["id"])
                    )
                else:
                    match_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO matches (id, resume_id, job_id, overall_score, model_name)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (match_id, resume_id, job_id, ann_score, model_name)
                    )
        except Exception as e:
            logger.warning("Could not persist ANN match to MySQL: %s", str(e))

    return {
        "success": True,
        "resume_id": resume_id,
        "job_id": job_id,
        "model": model_name,
        "predicted_label": pred_class,
        "is_match": bool(pred_class == 1),
        "ann_score": ann_score,
        "threshold": round(t, 4),
        "shared_skills": shared_skills,
    }


def predict_ann_matches_for_resume(
    resume_id: str,
    model_name: str = "tfidf_ann",
    limit: Optional[int] = None,
    persist: bool = True,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates a candidate resume against all available benchmark jobs using the
    supervised ANN classifier, returning results ranked by ANN score descending.

    :param resume_id: Candidate resume identifier.
    :param model_name: Model identifier string.
    :param limit: Optional max matches to return.
    :param persist: Whether to persist each match to MySQL.
    :param threshold: Optional decision threshold override.
    :return: Dictionary containing ranked job matches.
    """
    if not resume_id or not str(resume_id).strip():
        raise ValueError("resume_id parameter is required.")

    resume_id = str(resume_id).strip()
    resume_doc = get_resume_text(resume_id)

    # 1. Fetch available jobs
    import pandas as pd
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
        raise JobNotFoundError("No benchmark jobs available to evaluate against.")

    # 2. Predict on all pairs
    classifier = get_ann_classifier()
    t = threshold if threshold is not None else classifier.threshold
    pair_texts = [resume_doc + "\n\n" + j["doc"] for j in jobs_data]

    scores = classifier.predict_score(pair_texts)

    matches_list = []
    for idx, job_info in enumerate(jobs_data):
        score = float(round(scores[idx], 4))
        pred = int(1 if score >= t else 0)
        matches_list.append({
            "job_id": job_info["job_id"],
            "job_title": job_info["job_title"],
            "company": job_info["company"],
            "predicted_label": pred,
            "is_match": bool(pred == 1),
            "ann_score": score,
            "threshold": round(t, 4),
        })

    # Sort descending by ANN score
    matches_list.sort(key=lambda x: x["ann_score"], reverse=True)

    if limit is not None and limit > 0:
        matches_list = matches_list[:limit]

    # Optionally persist
    if persist:
        for m in matches_list:
            try:
                predict_and_store_ann_match(
                    resume_id=resume_id,
                    job_id=m["job_id"],
                    model_name=model_name,
                    persist=True,
                    threshold=t,
                )
            except Exception as e:
                logger.debug("Failed to persist ANN match (%s, %s): %s", resume_id, m["job_id"], str(e))

    return {
        "success": True,
        "resume_id": resume_id,
        "model": model_name,
        "total_jobs_evaluated": len(jobs_data),
        "threshold": round(t, 4),
        "matches": matches_list,
    }
