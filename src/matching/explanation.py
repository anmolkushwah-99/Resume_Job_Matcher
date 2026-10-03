"""
Skill Gap and Explanation Generation Module (Step 10).
Provides transparent, deterministic calculation of matched and missing skills
between a candidate resume and a benchmark job description.

Note: Skill gap explanations serve as auxiliary descriptive decision-support
indicators; they do not represent the causal weights of the underlying machine learning model.
"""

from typing import List, Dict, Any, Set


def calculate_skill_explanation(
    resume_skills: List[str],
    job_skills: List[str]
) -> Dict[str, Any]:
    """
    Computes matched skills, missing skills, and overlap statistics.

    :param resume_skills: List of extracted canonical skill names from candidate resume.
    :param job_skills: List of canonical skill requirements from benchmark job.
    :return: Dictionary containing matched_skills, missing_skills, counts, and overlap ratio.
    """
    if not resume_skills:
        resume_skills = []
    if not job_skills:
        job_skills = []

    # Case-insensitive mapping for normalization
    resume_map = {s.strip().lower(): s.strip() for s in resume_skills if str(s).strip()}
    job_map = {s.strip().lower(): s.strip() for s in job_skills if str(s).strip()}

    resume_set_lower = set(resume_map.keys())
    job_set_lower = set(job_map.keys())

    matched_lower = resume_set_lower.intersection(job_set_lower)
    missing_lower = job_set_lower - resume_set_lower

    # Reconstruct original title-cased names deterministically sorted
    matched_skills = sorted([job_map[s] for s in matched_lower])
    missing_skills = sorted([job_map[s] for s in missing_lower])

    job_count = len(job_set_lower)
    resume_count = len(resume_set_lower)
    matched_count = len(matched_skills)
    missing_count = len(missing_skills)

    overlap_ratio = round(matched_count / max(1, job_count), 4) if job_count > 0 else 0.0

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_count": matched_count,
        "missing_count": missing_count,
        "resume_skill_count": resume_count,
        "job_skill_count": job_count,
        "skill_overlap_ratio": overlap_ratio,
        "explanation_note": "Auxiliary skill overlap based on taxonomy extraction; decision-support indicator."
    }
