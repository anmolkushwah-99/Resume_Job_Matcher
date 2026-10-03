"""
Resume Text Builder Module.
Constructs clean, representative textual documents for resumes by combining
extracted raw text and structured NLP features (skills, education, experience, sections)
without artificial keyword inflation or redundancy.
"""

import logging
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import json

from src.preprocessing.text_cleaner import clean_resume_text
from src.database.mysql_client import get_db_cursor

logger = logging.getLogger(__name__)


def build_resume_document(resume_data: Union[Dict[str, Any], str]) -> str:
    """
    Builds a normalized textual representation of a resume document.

    If given a string: cleans and returns the text.
    If given a dictionary (e.g. from MySQL or processed NLP JSON):
      Combines cleaned resume text with structured entities (skills, education,
      experience, sections) in a balanced, natural format.

    :param resume_data: Extracted text string or dictionary of resume attributes.
    :return: Cleaned and structured resume document string.
    """
    if not resume_data:
        return ""

    if isinstance(resume_data, str):
        return clean_resume_text(resume_data)

    parts: List[str] = []

    # 1. Base text (extracted_text or resume_text or summary)
    raw_text = (
        resume_data.get("extracted_text")
        or resume_data.get("resume_text")
        or ""
    )
    if raw_text and raw_text.strip():
        parts.append(clean_resume_text(raw_text))

    # 2. Sections (if explicitly stored separately and base text is minimal)
    sections = resume_data.get("sections")
    if isinstance(sections, dict) and not raw_text:
        for sec_name, sec_content in sections.items():
            if sec_content and isinstance(sec_content, str) and sec_content.strip():
                parts.append(f"{sec_name.title()}: {clean_resume_text(sec_content)}")

    # 3. Canonical Skills (if available and not purely duplicating text)
    skills = resume_data.get("skills") or resume_data.get("skill_names")
    if skills:
        skill_names: List[str] = []
        if isinstance(skills, list):
            for s in skills:
                if isinstance(s, dict) and "skill_name" in s:
                    skill_names.append(s["skill_name"])
                elif isinstance(s, str) and s.strip():
                    skill_names.append(s.strip())
        elif isinstance(skills, str) and skills.strip():
            # Semicolon or comma separated string
            sep = ";" if ";" in skills else ","
            skill_names = [sk.strip() for sk in skills.split(sep) if sk.strip()]

        if skill_names:
            # Include skills summary line for strong domain keyword representation
            parts.append(f"Skills: {', '.join(skill_names)}")

    # 4. Education & Field of Study
    edu_parts = []
    degree = resume_data.get("degree")
    field = resume_data.get("field_of_study")
    edu_str = resume_data.get("education")

    if isinstance(edu_str, dict):
        degree = degree or edu_str.get("degree")
        field = field or edu_str.get("field_of_study")
    elif isinstance(edu_str, str) and edu_str.strip():
        edu_parts.append(edu_str.strip())

    if degree:
        edu_parts.append(str(degree).strip())
    if field:
        edu_parts.append(str(field).strip())

    if edu_parts and not any(p in raw_text for p in edu_parts):
        parts.append(f"Education: {' '.join(edu_parts)}")

    # 5. Experience years
    exp_years = resume_data.get("experience_years")
    if exp_years is not None:
        try:
            exp_val = float(exp_years)
            if exp_val > 0 and f"{exp_val} years" not in raw_text.lower():
                parts.append(f"Experience: {exp_val} years")
        except (ValueError, TypeError):
            pass

    # 6. Projects and Certifications if available
    certifications = resume_data.get("certifications") or resume_data.get("certifications_section")
    if certifications and isinstance(certifications, str) and certifications.strip():
        if certifications.strip() not in raw_text:
            parts.append(f"Certifications: {certifications.strip()}")

    projects = resume_data.get("projects") or resume_data.get("projects_section")
    if projects and isinstance(projects, str) and projects.strip():
        if projects.strip() not in raw_text:
            parts.append(f"Projects: {projects.strip()}")

    combined = "\n\n".join(parts)
    return clean_resume_text(combined)


def build_resume_document_from_db(resume_id: str) -> Optional[str]:
    """
    Loads resume information from MySQL and builds the representative document.

    :param resume_id: UUID of the resume in MySQL.
    :return: Normalized resume document or None if not found.
    """
    if not resume_id or not resume_id.strip():
        return None

    # 1. Fetch resume record
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT id, filename, extracted_text, education, degree, field_of_study, experience_years
            FROM resumes
            WHERE id = %s
            """,
            (resume_id.strip(),)
        )
        resume_row = cur.fetchone()

        if not resume_row:
            return None

        # 2. Fetch associated skills
        cur.execute(
            """
            SELECT s.skill_name, s.category
            FROM resume_skills rs
            JOIN skills s ON rs.skill_id = s.id
            WHERE rs.resume_id = %s
            ORDER BY s.skill_name ASC
            """,
            (resume_id.strip(),)
        )
        skill_rows = cur.fetchall()

    resume_data = dict(resume_row)
    resume_data["skills"] = [row["skill_name"] for row in skill_rows]

    # Also check if processed NLP JSON artifact exists for richer context
    json_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "resume_nlp" / f"{resume_id.strip()}.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                nlp_json = json.load(f)
                if not resume_data.get("extracted_text"):
                    resume_data["extracted_text"] = nlp_json.get("sections", {}).get("summary", "")
                if not resume_data.get("skills"):
                    resume_data["skills"] = nlp_json.get("skill_names", [])
                if nlp_json.get("certifications_section"):
                    resume_data["certifications"] = nlp_json["certifications_section"]
                if nlp_json.get("projects_section"):
                    resume_data["projects"] = nlp_json["projects_section"]
        except Exception as e:
            logger.debug("Failed to read processed NLP artifact for %s: %s", resume_id, str(e))

    return build_resume_document(resume_data)
