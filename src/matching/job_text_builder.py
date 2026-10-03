"""
Job Text Builder Module.
Constructs clean, representative textual documents for job descriptions by combining
job title, company, category, description, required skills, preferred skills,
minimum experience, and education requirements.
"""

import logging
from typing import Dict, Any, List, Union, Optional
from src.preprocessing.text_cleaner import clean_resume_text
from src.database.mysql_client import get_db_cursor

logger = logging.getLogger(__name__)


def build_job_document(job_data: Union[Dict[str, Any], str]) -> str:
    """
    Builds a normalized textual representation of a job document.

    If given a string: cleans and returns the text.
    If given a dictionary (e.g. from MySQL jobs table or jobs.csv):
      Combines job title, category, description, required skills, preferred skills,
      and qualifications into a cohesive document for TF-IDF vectorization.

    :param job_data: Job description string or dictionary of job fields.
    :return: Cleaned job document string.
    """
    if not job_data:
        return ""

    if isinstance(job_data, str):
        return clean_resume_text(job_data)

    parts: List[str] = []

    # 1. Job Title & Company
    title = job_data.get("job_title", "").strip()
    company = job_data.get("company", "").strip()
    if title and company:
        parts.append(f"{title} at {company}")
    elif title:
        parts.append(title)

    # 2. Job Category / Domain
    category = job_data.get("job_category", "").strip()
    if category:
        parts.append(f"Category: {category}")

    # 3. Job Description Body
    desc = job_data.get("job_description", "").strip()
    if desc:
        parts.append(clean_resume_text(desc))

    # 4. Required Skills
    req_skills = job_data.get("required_skills")
    if req_skills:
        skill_list: List[str] = []
        if isinstance(req_skills, list):
            for s in req_skills:
                if isinstance(s, dict) and "skill_name" in s:
                    skill_list.append(s["skill_name"])
                elif isinstance(s, str) and s.strip():
                    skill_list.append(s.strip())
        elif isinstance(req_skills, str) and req_skills.strip():
            sep = ";" if ";" in req_skills else ","
            skill_list = [sk.strip() for sk in req_skills.split(sep) if sk.strip()]

        if skill_list:
            parts.append(f"Required Skills: {', '.join(skill_list)}")

    # 5. Preferred Skills
    pref_skills = job_data.get("preferred_skills")
    if pref_skills:
        pref_list: List[str] = []
        if isinstance(pref_skills, list):
            for s in pref_skills:
                if isinstance(s, dict) and "skill_name" in s:
                    pref_list.append(s["skill_name"])
                elif isinstance(s, str) and s.strip():
                    pref_list.append(s.strip())
        elif isinstance(pref_skills, str) and pref_skills.strip():
            sep = ";" if ";" in pref_skills else ","
            pref_list = [sk.strip() for sk in pref_skills.split(sep) if sk.strip()]

        if pref_list:
            parts.append(f"Preferred Skills: {', '.join(pref_list)}")

    # 6. Minimum Experience
    min_exp = job_data.get("minimum_experience")
    if min_exp is not None:
        try:
            exp_val = float(min_exp)
            if exp_val > 0:
                parts.append(f"Experience Requirement: {exp_val} years")
        except (ValueError, TypeError):
            pass

    # 7. Education Requirement
    edu_req = job_data.get("education_requirement", "").strip()
    if edu_req:
        parts.append(f"Education Requirement: {edu_req}")

    combined = "\n\n".join(parts)
    return clean_resume_text(combined)


def build_job_document_from_db(job_id: str) -> Optional[str]:
    """
    Loads job record from MySQL and builds the representative document.

    :param job_id: UUID or job ID in MySQL.
    :return: Normalized job document or None if not found.
    """
    if not job_id or not job_id.strip():
        return None

    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT id, job_title, company, job_description, job_category,
                   minimum_experience, education_requirement
            FROM jobs
            WHERE id = %s
            """,
            (job_id.strip(),)
        )
        job_row = cur.fetchone()

        if not job_row:
            return None

        # Fetch associated skills from job_skills if any
        cur.execute(
            """
            SELECT s.skill_name, js.is_required
            FROM job_skills js
            JOIN skills s ON js.skill_id = s.id
            WHERE js.job_id = %s
            ORDER BY js.is_required DESC, s.skill_name ASC
            """,
            (job_id.strip(),)
        )
        skill_rows = cur.fetchall()

    job_data = dict(job_row)
    if skill_rows:
        job_data["required_skills"] = [r["skill_name"] for r in skill_rows if r["is_required"]]
        job_data["preferred_skills"] = [r["skill_name"] for r in skill_rows if not r["is_required"]]

    return build_job_document(job_data)
