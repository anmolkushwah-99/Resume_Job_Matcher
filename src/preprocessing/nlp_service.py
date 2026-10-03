"""
Resume NLP Processing and Database Persistence Service.
Manages transactional update of extracted NLP features and skill relations in MySQL.
"""

import logging
from typing import Dict, Any, Optional

from src.database.mysql_client import get_db_cursor
from src.preprocessing.nlp_pipeline import process_resume_text

logger = logging.getLogger(__name__)


class ResumeNotFoundError(Exception):
    """Raised when the requested resume ID does not exist in MySQL."""
    pass


class ResumeProcessingError(Exception):
    """Raised when NLP processing or database update fails."""
    pass


def process_resume_by_id(resume_id: str) -> Dict[str, Any]:
    """
    Fetches resume by ID from MySQL, executes the NLP pipeline,
    and transactionally updates resume columns and skill mappings in MySQL.

    :param resume_id: UUID of the resume in MySQL.
    :return: Privacy-safe summary dictionary.
    """
    if not resume_id or not resume_id.strip():
        raise ValueError("Invalid or empty resume_id provided.")

    clean_resume_id = resume_id.strip()

    # 1. Fetch resume from MySQL
    resume_row = None
    with get_db_cursor(commit=False) as cur:
        cur.execute("SELECT id, filename, extracted_text FROM resumes WHERE id = %s;", (clean_resume_id,))
        resume_row = cur.fetchone()

    if not resume_row:
        raise ResumeNotFoundError(f"Resume with ID '{clean_resume_id}' not found in database.")

    extracted_text = resume_row.get("extracted_text")
    if not extracted_text or not extracted_text.strip():
        raise ResumeProcessingError(f"Resume '{clean_resume_id}' has no extracted text to process.")

    # 2. Run Central NLP Pipeline
    try:
        nlp_result = process_resume_text(extracted_text, resume_id=clean_resume_id)
    except Exception as nlp_err:
        logger.error("NLP extraction failed for resume '%s': %s", clean_resume_id, str(nlp_err))
        raise ResumeProcessingError(f"NLP pipeline error: {str(nlp_err)}") from nlp_err

    # 3. Transactional Database Update
    try:
        with get_db_cursor(commit=True) as cur:
            # Update resumes table with degree, education, field_of_study, experience_years
            update_query = """
                UPDATE resumes
                SET education = %s,
                    degree = %s,
                    field_of_study = %s,
                    experience_years = %s
                WHERE id = %s;
            """
            cur.execute(update_query, (
                nlp_result.get("degree"),
                nlp_result.get("degree"),
                nlp_result.get("field_of_study"),
                nlp_result.get("experience_years", 0.0),
                clean_resume_id
            ))

            # Delete previous resume_skills to maintain idempotency
            cur.execute("DELETE FROM resume_skills WHERE resume_id = %s;", (clean_resume_id,))

            # Insert extracted skills
            skill_inserts = []
            for skill in nlp_result.get("skills", []):
                skill_id = skill.get("skill_id")
                if skill_id:
                    skill_inserts.append((clean_resume_id, skill_id))

            if skill_inserts:
                insert_skills_query = """
                    INSERT INTO resume_skills (resume_id, skill_id)
                    VALUES (%s, %s);
                """
                cur.executemany(insert_skills_query, skill_inserts)

        logger.info("Successfully persisted NLP features and %d skills for resume %s", len(skill_inserts), clean_resume_id)

    except Exception as db_err:
        logger.error("Database transaction failed while persisting NLP results for resume '%s': %s", clean_resume_id, str(db_err))
        raise ResumeProcessingError(f"Database persistence failed: {str(db_err)}") from db_err

    # 4. Return Privacy-Safe Summary (DO NOT echo full text)
    detected_section_names = [k for k, v in nlp_result.get("sections", {}).items() if v]

    return {
        "success": True,
        "message": "Resume NLP processing completed successfully",
        "resume_id": clean_resume_id,
        "filename": resume_row.get("filename"),
        "skills_detected": nlp_result.get("skills_count", 0),
        "skills": nlp_result.get("skill_names", []),
        "education_detected": bool(nlp_result.get("degree") or nlp_result.get("field_of_study")),
        "degree": nlp_result.get("degree"),
        "field_of_study": nlp_result.get("field_of_study"),
        "experience_years": nlp_result.get("experience_years", 0.0),
        "sections_detected": detected_section_names
    }
