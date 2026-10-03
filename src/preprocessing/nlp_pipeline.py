"""
Central Resume NLP Preprocessing Pipeline.
Coordinates tokenization, stopword removal, lemmatization, section parsing,
skill extraction, education extraction, and experience calculation into structured features.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from src.preprocessing.text_cleaner import clean_resume_text
from src.preprocessing.tokenizer import (
    tokenize,
    normalize_tokens,
    remove_stopwords,
    lemmatize_tokens,
)
from src.preprocessing.section_parser import parse_resume_sections
from src.preprocessing.skill_extractor import extract_skills, extract_skill_names
from src.preprocessing.education_extractor import extract_education
from src.preprocessing.experience_extractor import extract_experience

logger = logging.getLogger(__name__)


def get_processed_nlp_dir() -> Path:
    """
    Returns the absolute path to the directory where structured NLP JSON artifacts are stored.
    Default: data/processed/resume_nlp
    """
    base = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "resume_nlp"
    base.mkdir(parents=True, exist_ok=True)
    return base


def process_resume_text(text: str, resume_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Processes raw or cleaned resume text through the end-to-end NLP feature extraction pipeline.

    :param text: Resume text string.
    :param resume_id: Optional identifier for JSON artifact persistence.
    :return: Serialized dictionary of structured candidate features.
    """
    if not text or not isinstance(text, str) or not text.strip():
        raise ValueError("Resume text is empty or invalid. Cannot run NLP processing.")

    # 1. Ensure basic text cleaning
    clean_text = clean_resume_text(text)
    if not clean_text:
        raise ValueError("Cleaned resume text yielded 0 characters.")

    # 2. Section Parsing
    sections = parse_resume_sections(clean_text)

    # 3. Tokenization & NLP Normalization
    raw_tokens = tokenize(clean_text)
    norm_tokens = normalize_tokens(raw_tokens)
    filtered_tokens = remove_stopwords(raw_tokens)
    lemmas = lemmatize_tokens(filtered_tokens)

    # 4. Skill Extraction (Using Master Taxonomy)
    skills = extract_skills(clean_text)
    skill_names = [s["skill_name"] for s in skills]

    # 5. Education Extraction
    education_info = extract_education(clean_text, sections.get("education"))

    # 6. Experience Extraction
    experience_info = extract_experience(clean_text, sections.get("experience"))

    # 7. Compile Structured Output
    structured_result: Dict[str, Any] = {
        "resume_id": resume_id,
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "tokens_count": len(raw_tokens),
        "tokens": raw_tokens,
        "normalized_tokens": norm_tokens,
        "filtered_tokens": filtered_tokens,
        "lemmas": lemmas,
        "sections": sections,
        "skills_count": len(skills),
        "skills": skills,
        "skill_names": skill_names,
        "education": education_info,
        "degree": education_info.get("degree") if education_info else None,
        "field_of_study": education_info.get("field_of_study") if education_info else None,
        "experience": experience_info,
        "experience_years": experience_info.get("experience_years", 0.0),
        "projects_section": sections.get("projects", ""),
        "certifications_section": sections.get("certifications", "")
    }

    # 8. Persist Processed JSON Artifact if resume_id is provided
    if resume_id:
        try:
            output_dir = get_processed_nlp_dir()
            safe_id = "".join(c for c in resume_id if c.isalnum() or c in ("-", "_"))
            out_file = output_dir / f"{safe_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(structured_result, f, indent=2, ensure_ascii=False)
            logger.info("Saved structured NLP artifact: %s", out_file)
        except Exception as e:
            logger.warning("Failed to save NLP JSON artifact for resume '%s': %s", resume_id, str(e))

    return structured_result
