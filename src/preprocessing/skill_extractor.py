"""
Skill Extraction and Taxonomy Normalization Module.
Extracts candidate skills from resume text using the 100-skill Master Taxonomy.
Performs alias mapping, phrase-level boundary matching, and deduplication.
"""

import re
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_TAXONOMY_CACHE: Optional[List[Dict[str, Any]]] = None
_COMPILED_SKILL_PATTERNS: Optional[List[Tuple[Dict[str, Any], re.Pattern, str]]] = None


def load_skill_taxonomy(taxonomy_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Loads and caches the 100-skill master taxonomy from CSV.
    """
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is not None:
        return _TAXONOMY_CACHE

    if taxonomy_file is None:
        taxonomy_file = Path(__file__).resolve().parent.parent.parent / "data" / "taxonomy" / "skills.csv"

    if not taxonomy_file.exists():
        logger.error("Skill taxonomy file not found at: %s", taxonomy_file)
        return []

    taxonomy = []
    with open(taxonomy_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skill_id = row["skill_id"].strip()
            name = row["skill_name"].strip()
            category = row["category"].strip()
            raw_aliases = row.get("aliases", "")

            aliases_list = [a.strip() for a in raw_aliases.split(";") if a.strip()]
            # Include canonical name in aliases list if not already present
            if name.lower() not in [a.lower() for a in aliases_list]:
                aliases_list.append(name)

            taxonomy.append({
                "skill_id": skill_id,
                "skill_name": name,
                "category": category,
                "aliases": aliases_list
            })

    _TAXONOMY_CACHE = taxonomy
    logger.info("Loaded %d skills into taxonomy cache.", len(_TAXONOMY_CACHE))
    return _TAXONOMY_CACHE


def _build_regex_for_term(term: str) -> re.Pattern:
    """
    Constructs a precise regex pattern for a skill term respecting technical punctuation
    and word boundaries to prevent false-positive substring matches.
    """
    escaped = re.escape(term)

    # Special handling for single/short terms or terms with punctuation
    if term.lower() in ("c", "r"):
        # Match single letters C or R only when surrounded by non-alphanumeric chars,
        # but avoid matching if followed by ++, #, etc.
        pattern = rf"(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_+#])"
        return re.compile(pattern)  # Case-sensitive for 1-letter skills
    elif term.lower() in ("go", "ai", "it", "ml", "dl", "cv", "js", "ts", "py"):
        pattern = rf"(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_])"
        return re.compile(pattern, re.IGNORECASE)
    elif term.startswith("."):
        # E.g. .NET -> match when not preceded by word char or dot
        pattern = rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])"
        return re.compile(pattern, re.IGNORECASE)
    elif "+" in term or "#" in term or "/" in term:
        # E.g. C++, C#, CI/CD, TCP/IP
        pattern = rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])"
        return re.compile(pattern, re.IGNORECASE)
    else:
        # Standard word boundary
        pattern = rf"\b{escaped}\b"
        return re.compile(pattern, re.IGNORECASE)


def get_compiled_skill_patterns() -> List[Tuple[Dict[str, Any], re.Pattern, str]]:
    """
    Compiles regex patterns for all taxonomy skills and their aliases.
    Sorted by alias length descending so longer multi-word phrases match first.
    """
    global _COMPILED_SKILL_PATTERNS
    if _COMPILED_SKILL_PATTERNS is not None:
        return _COMPILED_SKILL_PATTERNS

    taxonomy = load_skill_taxonomy()
    patterns = []

    for skill in taxonomy:
        # Sort aliases by length descending
        sorted_aliases = sorted(skill["aliases"], key=len, reverse=True)
        for alias in sorted_aliases:
            if not alias:
                continue
            regex = _build_regex_for_term(alias)
            patterns.append((skill, regex, alias))

    # Sort all patterns so longer alias terms are evaluated first
    patterns.sort(key=lambda x: len(x[2]), reverse=True)
    _COMPILED_SKILL_PATTERNS = patterns
    return _COMPILED_SKILL_PATTERNS


def extract_skills(text: str) -> List[Dict[str, Any]]:
    """
    Extracts canonical skills from resume text using taxonomy alias matching.

    :param text: Cleaned resume text string.
    :return: List of unique detected skill dictionaries.
    """
    if not text or not isinstance(text, str):
        return []

    compiled_patterns = get_compiled_skill_patterns()
    detected_skills: Dict[str, Dict[str, Any]] = {}

    for skill_info, regex, alias in compiled_patterns:
        skill_id = skill_info["skill_id"]

        # If this canonical skill is already matched, skip further alias checks
        if skill_id in detected_skills:
            continue

        match = regex.search(text)
        if match:
            detected_skills[skill_id] = {
                "skill_id": skill_id,
                "skill_name": skill_info["skill_name"],
                "category": skill_info["category"],
                "matched_alias": match.group(0)
            }

    # Return list sorted alphabetically by canonical skill name
    return sorted(detected_skills.values(), key=lambda s: s["skill_name"])


def extract_skill_names(text: str) -> List[str]:
    """
    Convenience function returning only the list of canonical skill name strings.
    """
    skills = extract_skills(text)
    return [s["skill_name"] for s in skills]
