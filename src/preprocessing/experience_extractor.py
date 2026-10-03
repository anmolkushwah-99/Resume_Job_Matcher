"""
Professional Experience Extraction Module.
Extracts numeric years of experience, job roles, internships, and organization mentions.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Explicit experience duration regex patterns
EXPERIENCE_YEARS_PATTERNS = [
    # "X.X years of experience", "X+ years experience", "X yrs of software development experience"
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+[a-zA-Z\s]{0,30}?)?\s*(?:experience|expertise|in\b)",
    # "with X years of expertise"
    r"with\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+[a-zA-Z\s]{0,30}?)?\s*(?:experience|expertise|in\b)",
    # "over X years", "more than X years"
    r"(?:over|more\s+than)\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)(?:\s+of)?(?:\s+[a-zA-Z\s]{0,30}?)?\s*(?:experience|expertise|in\b)?",
    # "X years as a developer"
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+as\s+(?:a|an)\b",
]

# Months pattern ("X months of experience")
MONTHS_PATTERNS = [
    r"(\d+)\s*\+?\s*(?:months?|mos?)(?:\s+of)?\s*(?:experience|internship|work)",
]

# Common job title patterns
JOB_TITLE_KEYWORDS = [
    r"software\s+(?:engineer|developer)",
    r"(?:frontend|backend|full\s*stack)\s+(?:developer|engineer)",
    r"machine\s+learning\s+engineer",
    r"data\s+(?:scientist|analyst|engineer)",
    r"devops\s+engineer",
    r"cloud\s+engineer",
    r"mobile\s+(?:app\s+)?developer",
    r"python\s+developer",
    r"java\s+developer",
    r"web\s+developer",
    r"intern|internship|trainee",
    r"associate\s+engineer",
    r"system\s+administrator"
]


def extract_experience(text: str, experience_section: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts total years of experience, detected job roles, and organizations.

    :param text: Cleaned full resume text.
    :param experience_section: Specific experience section text if available.
    :return: Dictionary containing experience_years (float), roles (list), and is_fresher (bool).
    """
    if not text:
        return {
            "experience_years": 0.0,
            "roles": [],
            "is_fresher": True
        }

    search_corpus = experience_section.strip() if experience_section and experience_section.strip() else text

    detected_years: List[float] = []

    # 1. Match explicit years patterns
    for pat in EXPERIENCE_YEARS_PATTERNS:
        matches = re.finditer(pat, search_corpus, re.IGNORECASE)
        for m in matches:
            try:
                val = float(m.group(1))
                if 0.0 <= val <= 40.0:  # Sanity range
                    detected_years.append(val)
            except (ValueError, IndexError):
                pass

    # 2. Match months patterns (convert months to years)
    for pat in MONTHS_PATTERNS:
        matches = re.finditer(pat, search_corpus, re.IGNORECASE)
        for m in matches:
            try:
                val = float(m.group(1)) / 12.0
                if 0.0 <= val <= 40.0:
                    detected_years.append(round(val, 1))
            except (ValueError, IndexError):
                pass

    # 3. Detect "Fresher" mentions
    is_fresher = False
    if re.search(r"\b(fresher|entry[\s-]level|recent\s+graduate|no\s+experience)\b", text, re.IGNORECASE):
        is_fresher = True

    # Compute final experience_years (take the maximum explicitly mentioned years)
    final_years = max(detected_years) if detected_years else 0.0
    if is_fresher and not detected_years:
        final_years = 0.0

    # 4. Extract Roles & Organization titles
    roles: List[Dict[str, str]] = []
    lines = search_corpus.split("\n")
    for line in lines:
        line_clean = line.strip()
        if not line_clean or len(line_clean) > 120:
            continue

        for title_kw in JOB_TITLE_KEYWORDS:
            if re.search(rf"\b{title_kw}\b", line_clean, re.IGNORECASE):
                # Found a role line
                roles.append({
                    "title": line_clean,
                    "matched_keyword": title_kw
                })
                break

    return {
        "experience_years": float(final_years),
        "roles": roles,
        "is_fresher": is_fresher or (final_years == 0.0 and not roles)
    }
