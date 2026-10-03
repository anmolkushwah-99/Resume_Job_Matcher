"""
Final Resume-Job Matching Engine (Step 10).
Orchestrates production-style inference, multi-job ranking, skill explanations,
batch optimization, and MySQL persistence using the Step 9 validated primary model (Linear SVM).
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import pandas as pd
import numpy as np

from src.models.svm_classifier import SVMResumeJobClassifier
from src.matching.explanation import calculate_skill_explanation
from src.matching.model_registry import ModelRegistry
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

logger = logging.getLogger(__name__)


def _get_all_job_ids_from_db_or_csv() -> List[str]:
    """Fetches all unique benchmark job IDs from MySQL or jobs.csv fallback."""
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute("SELECT id FROM jobs ORDER BY id ASC")
            rows = cur.fetchall()
            if rows:
                return [str(r["id"]).strip() for r in rows]
    except Exception as e:
        logger.debug("Could not fetch job IDs from DB: %s", str(e))

    jobs_csv = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "jobs" / "jobs.csv"
    if jobs_csv.exists():
        try:
            df = pd.read_csv(jobs_csv)
            if "job_id" in df.columns:
                return [str(jid).strip() for jid in df["job_id"].tolist()]
        except Exception:
            pass
    return []


def _get_job_metadata(job_id: str) -> Dict[str, Any]:
    """Retrieves metadata fields for a specific job from MySQL or jobs.csv fallback."""
    job_id_clean = str(job_id).strip()
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT id, job_title, company, job_category, minimum_experience, education_requirement
                FROM jobs WHERE id = %s
                """,
                (job_id_clean,)
            )
            row = cur.fetchone()
            if row:
                return {
                    "job_id": row["id"],
                    "title": row.get("job_title", job_id_clean),
                    "company": row.get("company", ""),
                    "category": row.get("job_category", ""),
                    "minimum_experience": row.get("minimum_experience"),
                    "education_requirement": row.get("education_requirement", ""),
                }
    except Exception as e:
        logger.debug("Could not fetch job metadata from DB for %s: %s", job_id_clean, str(e))

    jobs_csv = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "jobs" / "jobs.csv"
    if jobs_csv.exists():
        try:
            df = pd.read_csv(jobs_csv)
            sub = df[df["job_id"] == job_id_clean]
            if not sub.empty:
                r = sub.iloc[0]
                return {
                    "job_id": job_id_clean,
                    "title": r.get("job_title", job_id_clean),
                    "company": r.get("company", ""),
                    "category": r.get("job_category", ""),
                    "minimum_experience": r.get("minimum_experience"),
                    "education_requirement": r.get("education_requirement", ""),
                }
        except Exception:
            pass
    return {"job_id": job_id_clean, "title": job_id_clean}



class FinalMatchEngine:
    """
    Production-grade matching engine utilizing the validated Step 6/9 Linear SVM model.
    Encapsulates single-pair evaluation, batch multi-job ranking, skill gap explanations,
    and MySQL persistence.
    """

    _instance: Optional["FinalMatchEngine"] = None

    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            model_dir = ModelRegistry.get_models_dir()

        self.model_dir = Path(model_dir)
        self.model_name = "tfidf_svm"
        self.model_type = "Linear Support Vector Classifier (LinearSVC)"
        self.threshold = 0.0

        if not ModelRegistry.verify_model_artifacts(self.model_name, self.model_dir):
            raise FileNotFoundError(
                f"Required model artifacts for '{self.model_name}' missing or incompatible in {self.model_dir}"
            )

        logger.info("Initializing FinalMatchEngine with primary model: %s from %s", self.model_name, self.model_dir)
        self.classifier = SVMResumeJobClassifier.load(self.model_dir)
        self.is_ready = True
        self._job_cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(cls, model_dir: Optional[Path] = None) -> "FinalMatchEngine":
        """Singleton factory accessor for FinalMatchEngine."""
        if cls._instance is None or not cls._instance.is_ready:
            cls._instance = cls(model_dir=model_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resets the singleton instance (used in test teardowns)."""
        cls._instance = None

    def _resolve_job(self, job_id: str) -> Dict[str, Any]:
        """Loads and caches job text, skills, and metadata."""
        job_id_clean = str(job_id).strip()
        if job_id_clean in self._job_cache:
            return self._job_cache[job_id_clean]

        j_doc = get_job_text(job_id_clean)
        j_skills = get_job_skills_from_db_or_csv(job_id_clean)
        j_meta = _get_job_metadata(job_id_clean)

        data = {
            "doc": j_doc,
            "skills": j_skills,
            "meta": j_meta
        }
        self._job_cache[job_id_clean] = data
        return data

    def clear_job_cache(self) -> None:
        """Clears the cached job data."""
        self._job_cache.clear()

    def match_one(
        self,
        resume_id: str,
        job_id: str,
        persist: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates a single resume against a single job description.

        :param resume_id: Candidate resume identifier.
        :param job_id: Benchmark job identifier.
        :param persist: Whether to persist result into MySQL matches table.
        :return: Structured match result dictionary.
        """
        if not resume_id or not str(resume_id).strip():
            raise ValueError("resume_id parameter is required.")
        if not job_id or not str(job_id).strip():
            raise ValueError("job_id parameter is required.")

        resume_id = str(resume_id).strip()
        job_id = str(job_id).strip()

        # 1. Fetch resume and job texts
        resume_doc = get_resume_text(resume_id)
        job_info = self._resolve_job(job_id)
        job_doc = job_info["doc"]

        # 2. Extract skills for explanation
        resume_skills = get_resume_skills_from_db_or_nlp(resume_id)
        job_skills = job_info["skills"]

        # 3. Build pair representation identical to Step 6 training
        pair_text = f"{resume_doc}\n\n{job_doc}"

        # 4. Compute continuous SVM decision score
        decision_score_arr = self.classifier.decision_function([pair_text])
        raw_score = float(decision_score_arr[0])
        decision_score = round(raw_score, 4)

        # 5. Apply natural decision threshold (0.0)
        is_match = bool(raw_score >= self.threshold)
        predicted_label = 1 if is_match else 0

        # 6. Generate auxiliary skill explanation
        explanation = calculate_skill_explanation(resume_skills, job_skills)

        # 7. Persist to MySQL
        match_id = str(uuid.uuid4())
        created_at_iso = datetime.utcnow().isoformat()

        if persist:
            self._persist_match_record(
                match_id=match_id,
                resume_id=resume_id,
                job_id=job_id,
                decision_score=decision_score,
                is_match=is_match,
                matched_skills=explanation["matched_skills"],
                missing_skills=explanation["missing_skills"],
                skill_score=explanation["skill_overlap_ratio"]
            )

        # 8. Retrieve job metadata for context
        job_meta = job_info["meta"]

        return {
            "success": True,
            "match_id": match_id,
            "resume_id": resume_id,
            "job_id": job_id,
            "job_title": job_meta.get("title", job_id),
            "model_name": self.model_name,
            "model_type": self.model_type,
            "decision_score": decision_score,
            "threshold": self.threshold,
            "predicted_label": predicted_label,
            "is_match": is_match,
            "matched_skills": explanation["matched_skills"],
            "missing_skills": explanation["missing_skills"],
            "matched_skill_count": explanation["matched_count"],
            "missing_skill_count": explanation["missing_count"],
            "skill_overlap_ratio": explanation["skill_overlap_ratio"],
            "explanation_note": explanation["explanation_note"],
            "created_at": created_at_iso
        }

    def match_many(
        self,
        resume_id: str,
        job_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        match_only: bool = False,
        persist: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates and ranks multiple benchmark jobs for a given resume using
        batch TF-IDF vectorization and deterministic score sorting.

        :param resume_id: Candidate resume identifier.
        :param job_ids: Optional list of job identifiers. If None, evaluates all available jobs.
        :param top_k: Optional integer to limit the number of returned ranked jobs.
        :param match_only: If True, returns only jobs classified as matches (decision_score >= 0.0).
        :param persist: Whether to persist all match records into MySQL.
        :return: Structured dictionary with ranked jobs and summary statistics.
        """
        if not resume_id or not str(resume_id).strip():
            raise ValueError("resume_id parameter is required.")

        resume_id = str(resume_id).strip()

        if top_k is not None:
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k must be a positive integer.")

        # 1. Fetch resume document and skills
        resume_doc = get_resume_text(resume_id)
        resume_skills = get_resume_skills_from_db_or_nlp(resume_id)

        # 2. Determine target job IDs
        if job_ids is None or len(job_ids) == 0:
            target_job_ids = _get_all_job_ids_from_db_or_csv()
        else:
            target_job_ids = [str(jid).strip() for jid in job_ids if str(jid).strip()]

        if not target_job_ids:
            raise ValueError("No eligible jobs found for matching.")

        # 3. Batch build pair texts
        valid_job_ids = []
        pair_texts = []
        job_skills_map = {}
        job_meta_map = {}

        for jid in target_job_ids:
            try:
                job_info = self._resolve_job(jid)
                valid_job_ids.append(jid)
                pair_texts.append(f"{resume_doc}\n\n{job_info['doc']}")
                job_skills_map[jid] = job_info["skills"]
                job_meta_map[jid] = job_info["meta"]
            except JobNotFoundError:
                logger.warning("Job %s not found during multi-job matching; skipping.", jid)
                continue

        if not valid_job_ids:
            raise JobNotFoundError("None of the specified job IDs were found in the database or benchmark dataset.")

        # 4. Batch vectorization & inference
        decision_scores_raw = self.classifier.decision_function(pair_texts)

        results = []
        matches_to_persist = []

        for idx, jid in enumerate(valid_job_ids):
            raw_score = float(decision_scores_raw[idx])
            decision_score = round(raw_score, 4)
            is_match = bool(raw_score >= self.threshold)
            predicted_label = 1 if is_match else 0

            j_skills = job_skills_map[jid]
            j_meta = job_meta_map[jid]
            explanation = calculate_skill_explanation(resume_skills, j_skills)

            match_id = str(uuid.uuid4())
            item = {
                "match_id": match_id,
                "job_id": jid,
                "job_title": j_meta.get("title", jid),
                "job_role": j_meta.get("role", ""),
                "department": j_meta.get("department", ""),
                "decision_score": decision_score,
                "threshold": self.threshold,
                "predicted_label": predicted_label,
                "is_match": is_match,
                "matched_skills": explanation["matched_skills"],
                "missing_skills": explanation["missing_skills"],
                "matched_skill_count": explanation["matched_count"],
                "missing_skill_count": explanation["missing_count"],
                "skill_overlap_ratio": explanation["skill_overlap_ratio"],
            }
            results.append(item)

            if persist:
                matches_to_persist.append({
                    "match_id": match_id,
                    "resume_id": resume_id,
                    "job_id": jid,
                    "decision_score": decision_score,
                    "is_match": is_match,
                    "matched_skills": explanation["matched_skills"],
                    "missing_skills": explanation["missing_skills"],
                    "skill_score": explanation["skill_overlap_ratio"]
                })

        # 5. Persist batch to MySQL
        if persist and matches_to_persist:
            self._persist_batch_records(matches_to_persist)

        # 6. Deterministic Sorting: Primary decision_score DESC, Secondary job_id ASC
        results.sort(key=lambda x: (-x["decision_score"], x["job_id"]))

        # 7. Apply match_only filter if requested
        if match_only:
            results = [r for r in results if r["is_match"]]

        # 8. Apply top_k slice if requested
        if top_k is not None:
            results = results[:top_k]

        # 9. Invert rank index
        for rank_idx, r in enumerate(results, start=1):
            r["rank"] = rank_idx

        # 10. Summary statistics
        total_eval = len(valid_job_ids)
        total_matches = sum(1 for r in results if r["is_match"])
        total_non_matches = sum(1 for r in results if not r["is_match"])
        scores = [r["decision_score"] for r in results]
        highest_score = max(scores) if scores else None
        lowest_score = min(scores) if scores else None

        return {
            "success": True,
            "resume_id": resume_id,
            "model": {
                "name": self.model_name,
                "type": self.model_type,
                "threshold": self.threshold,
                "decision_rule": "decision_score >= 0.0 -> Match"
            },
            "total_jobs_evaluated": total_eval,
            "total_returned": len(results),
            "total_predicted_matches": total_matches,
            "total_predicted_non_matches": total_non_matches,
            "highest_decision_score": highest_score,
            "lowest_decision_score": lowest_score,
            "ranking_criteria": "decision_score DESC, job_id ASC (deterministic)",
            "results": results
        }

    def match_batch(
        self,
        resume_id: str,
        job_ids: List[str],
        persist: bool = True
    ) -> Dict[str, Any]:
        """
        Executes batch matching with per-job failure isolation.
        """
        if not isinstance(job_ids, list) or len(job_ids) == 0:
            raise ValueError("job_ids must be a non-empty list of strings.")

        successful_results = []
        failed_jobs = []

        for jid in job_ids:
            jid_str = str(jid).strip()
            try:
                res = self.match_one(resume_id, jid_str, persist=persist)
                successful_results.append(res)
            except Exception as e:
                logger.warning("Batch matching failed for pair (%s, %s): %s", resume_id, jid_str, e)
                failed_jobs.append({
                    "job_id": jid_str,
                    "error": str(e)
                })

        return {
            "success": True,
            "resume_id": resume_id,
            "total_submitted": len(job_ids),
            "successful_count": len(successful_results),
            "failed_count": len(failed_jobs),
            "results": successful_results,
            "failed_jobs": failed_jobs
        }

    def _persist_match_record(
        self,
        match_id: str,
        resume_id: str,
        job_id: str,
        decision_score: float,
        is_match: bool,
        matched_skills: List[str],
        missing_skills: List[str],
        skill_score: float
    ) -> None:
        """Persists a single match record to the MySQL matches table."""
        try:
            with get_db_cursor(commit=True) as cur:
                cur.execute(
                    """
                    SELECT id FROM matches
                    WHERE resume_id = %s AND job_id = %s AND model_name = %s
                    LIMIT 1
                    """,
                    (resume_id, job_id, self.model_name)
                )
                existing = cur.fetchone()
                if existing:
                    cur.execute(
                        """
                        UPDATE matches
                        SET overall_score = %s, skill_score = %s, created_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (decision_score, skill_score, existing["id"])
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO matches (id, resume_id, job_id, overall_score, skill_score, model_name)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (match_id, resume_id, job_id, decision_score, skill_score, self.model_name)
                    )
        except Exception as e:
            logger.warning("Could not persist match record (%s, %s) to MySQL: %s", resume_id, job_id, str(e))

    def _persist_batch_records(self, records: List[Dict[str, Any]]) -> None:
        """Persists multiple match records to MySQL."""
        for r in records:
            self._persist_match_record(
                match_id=r["match_id"],
                resume_id=r["resume_id"],
                job_id=r["job_id"],
                decision_score=r["decision_score"],
                is_match=r["is_match"],
                matched_skills=r["matched_skills"],
                missing_skills=r["missing_skills"],
                skill_score=r.get("skill_score", 0.0)
            )


def get_final_match_engine(model_dir: Optional[Path] = None) -> FinalMatchEngine:
    """Convenience functional accessor for FinalMatchEngine."""
    return FinalMatchEngine.get_instance(model_dir=model_dir)

