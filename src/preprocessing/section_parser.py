"""
Resume Section Detection and Parsing Module.
Detects common resume sections (Summary, Skills, Education, Experience, Projects, Certifications)
using robust heading pattern detection and line-boundary analysis.
"""

import re
import logging
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Canonical Section Names and their regex heading patterns
SECTION_PATTERNS: Dict[str, List[str]] = {
    "summary": [
        r"professional\s+summary",
        r"executive\s+summary",
        r"career\s+objective",
        r"career\s+summary",
        r"summary\s+of\s+qualifications",
        r"about\s+me",
        r"profile",
        r"summary",
        r"objective"
    ],
    "skills": [
        r"technical\s+skills",
        r"core\s+competencies",
        r"areas\s+of\s+expertise",
        r"technical\s+proficiencies",
        r"technical\s+expertise",
        r"key\s+skills",
        r"skill\s+set",
        r"tech\s+stack",
        r"skills\s+and\s+tools",
        r"skills\s+&\s+technologies",
        r"skills",
        r"technologies"
    ],
    "education": [
        r"academic\s+qualifications",
        r"educational\s+background",
        r"academic\s+background",
        r"education\s+&\s+certifications",
        r"education\s+and\s+training",
        r"education",
        r"academics",
        r"qualifications"
    ],
    "experience": [
        r"professional\s+experience",
        r"work\s+experience",
        r"employment\s+history",
        r"work\s+history",
        r"relevant\s+experience",
        r"internship\s+experience",
        r"internships",
        r"internship",
        r"experience"
    ],
    "projects": [
        r"personal\s+projects",
        r"academic\s+projects",
        r"key\s+projects",
        r"notable\s+projects",
        r"technical\s+projects",
        r"projects\s+undertaken",
        r"projects",
        r"project\s+work"
    ],
    "certifications": [
        r"licenses\s+(&|and)\s+certifications",
        r"professional\s+certifications",
        r"certificates\s+(&|and)\s+licenses",
        r"courses\s+(&|and)\s+certifications",
        r"certifications",
        r"certificates",
        r"licenses",
        r"courses",
        r"training"
    ],
    "achievements": [
        r"honors\s+(&|and)\s+awards",
        r"awards\s+(&|and)\s+achievements",
        r"accomplishments",
        r"achievements",
        r"awards",
        r"honors"
    ],
    "languages": [
        r"language\s+proficiency",
        r"languages\s+known",
        r"languages"
    ],
    "interests": [
        r"extracurricular\s+activities",
        r"extra-curricular\s+activities",
        r"hobbies\s+(&|and)\s+interests",
        r"interests",
        r"hobbies"
    ]
}


def _compile_heading_regex() -> List[Tuple[str, re.Pattern]]:
    """
    Compiles heading patterns in order of specificity (longest patterns first).
    """
    compiled = []
    for canonical_name, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            # Matches isolated heading line (allowing optional colons or dashes)
            pattern_str = rf"^(?:[-*•\s]*)({pat})(?:\s*[:\-]*)?$"
            regex = re.compile(pattern_str, re.IGNORECASE)
            compiled.append((canonical_name, regex))
    return compiled


COMPILED_SECTION_PATTERNS = _compile_heading_regex()


def parse_resume_sections(text: str) -> Dict[str, str]:
    """
    Parses full resume text into classified logical sections.

    :param text: Cleaned full resume text.
    :return: Dictionary mapping canonical section names to extracted text blocks.
    """
    sections: Dict[str, List[str]] = {
        "summary": [],
        "skills": [],
        "education": [],
        "experience": [],
        "projects": [],
        "certifications": [],
        "achievements": [],
        "languages": [],
        "interests": [],
        "general": []
    }

    if not text or not isinstance(text, str):
        return {k: "" for k in sections if k != "general"}

    lines = [line.strip() for line in text.split("\n")]
    current_section = "general"

    for line in lines:
        if not line:
            continue

        matched_section = None
        # Check if this line is a section heading
        for canonical_name, regex in COMPILED_SECTION_PATTERNS:
            if regex.match(line):
                matched_section = canonical_name
                break

        if matched_section is not None:
            current_section = matched_section
        else:
            sections[current_section].append(line)

    # Join lines for each section
    result: Dict[str, str] = {}
    for sec_name, sec_lines in sections.items():
        if sec_name == "general":
            continue
        joined = "\n".join(sec_lines).strip()
        result[sec_name] = joined

    # If summary is empty but general has introductory text, assign general to summary
    if not result["summary"] and sections["general"]:
        result["summary"] = "\n".join(sections["general"]).strip()

    return result
