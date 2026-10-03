"""
Education Information Extraction Module.
Detects degree, field of study, and institution from resume text and sections.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Degree normalization mappings (regex pattern to canonical name)
DEGREE_PATTERNS = [
    (r"\b(b\.?\s*tech(?:nology)?|bachelor\s+of\s+technology)\b", "B.Tech"),
    (r"\b(b\.?\s*e\.?|bachelor\s+of\s+engineering)\b", "B.E."),
    (r"\b(b\.?\s*sc\.?|b\.?\s*s\.?|bachelor\s+of\s+science)\b", "B.Sc"),
    (r"\b(bca|bachelor\s+of\s+computer\s+applications?)\b", "BCA"),
    (r"\b(bba|bachelor\s+of\s+business\s+administration)\b", "BBA"),
    (r"\b(b\.?\s*a\.?|bachelor\s+of\s+arts)\b", "B.A."),
    (r"\b(bachelor(?:'s)?(?:\s+degree)?)\b", "Bachelor"),
    (r"\b(m\.?\s*tech(?:nology)?|master\s+of\s+technology)\b", "M.Tech"),
    (r"\b(m\.?\s*e\.?|master\s+of\s+engineering)\b", "M.E."),
    (r"\b(m\.?\s*sc\.?|m\.?\s*s\.?|master\s+of\s+science)\b", "M.S."),
    (r"\b(mca|master\s+of\s+computer\s+applications?)\b", "MCA"),
    (r"\b(mba|master\s+of\s+business\s+administration)\b", "MBA"),
    (r"\b(master(?:'s)?(?:\s+degree)?)\b", "Master"),
    (r"\b(ph\.?d\.?|doctor\s+of\s+philosophy|doctorate)\b", "PhD"),
    (r"\b(diploma(?:\s+in\s+[a-zA-Z\s]+)?)\b", "Diploma"),
]

# Common Fields of Study patterns
FIELD_OF_STUDY_PATTERNS = [
    (r"\b(computer\s+science(?:\s+and\s+engineering)?|cs(?:e)?)\b", "Computer Science"),
    (r"\b(information\s+technology|it)\b", "Information Technology"),
    (r"\b(data\s+science(?:\s+and\s+analytics)?)\b", "Data Science"),
    (r"\b(artificial\s+intelligence(?:\s+and\s+machine\s+learning)?|ai(?:\s*(&|and)\s*ml)?)\b", "Artificial Intelligence"),
    (r"\b(software\s+engineering)\b", "Software Engineering"),
    (r"\b(computer\s+engineering)\b", "Computer Engineering"),
    (r"\b(electronics(?:\s+and\s+communication(?:\s+engineering)?)?|ece)\b", "Electronics and Communication"),
    (r"\b(electrical\s+engineering|eee)\b", "Electrical Engineering"),
    (r"\b(mechanical\s+engineering)\b", "Mechanical Engineering"),
    (r"\b(civil\s+engineering)\b", "Civil Engineering"),
    (r"\b(statistics|applied\s+statistics)\b", "Statistics"),
    (r"\b(mathematics|applied\s+mathematics)\b", "Mathematics"),
    (r"\b(business\s+administration|management)\b", "Business Administration"),
]

# Institution keywords
INSTITUTION_KEYWORDS = [
    r"university", r"institute(?:\s+of\s+technology)?", r"college", r"school\s+of",
    r"iit", r"nit", r"iiit", r"bits", r"polytechnic", r"academy"
]


def extract_education(text: str, education_section: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Extracts candidate education details (degree, field of study, institution).

    :param text: Full cleaned resume text.
    :param education_section: Specific text from the education section if available.
    :return: Dictionary of extracted education features, or None if not found.
    """
    if not text:
        return None

    # Priority to search: education section first, then full text
    search_corpus = education_section.strip() if education_section and education_section.strip() else text

    detected_degree = None
    detected_field = None
    detected_institution = None
    matched_snippet = None

    # 1. Search for Degree
    for pattern, canonical_degree in DEGREE_PATTERNS:
        match = re.search(pattern, search_corpus, re.IGNORECASE)
        if match:
            detected_degree = canonical_degree
            # Capture context line/snippet around match
            start_pos = max(0, match.start() - 20)
            end_pos = min(len(search_corpus), match.end() + 60)
            matched_snippet = search_corpus[start_pos:end_pos].strip()
            break

    # 2. Search for Field of Study
    for pattern, canonical_field in FIELD_OF_STUDY_PATTERNS:
        match = re.search(pattern, search_corpus, re.IGNORECASE)
        if match:
            detected_field = canonical_field
            break

    # 3. Search for Institution (Universities/Colleges)
    for line in search_corpus.split("\n"):
        line_clean = line.strip()
        for inst_kw in INSTITUTION_KEYWORDS:
            if re.search(rf"\b{inst_kw}\b", line_clean, re.IGNORECASE):
                # Avoid capturing entire long paragraphs as institution name
                if len(line_clean) < 100 and not line_clean.lower().startswith("skills"):
                    detected_institution = line_clean
                    break
        if detected_institution:
            break

    # If neither degree nor field was found, return None
    if not detected_degree and not detected_field:
        return None

    return {
        "degree": detected_degree,
        "field_of_study": detected_field,
        "institution": detected_institution,
        "raw_text": matched_snippet
    }
