"""
Resume Text Cleaning Module.
Provides basic, conservative text normalization while strictly preserving
technical terms, symbols, numbers, punctuation, and structural layout.
"""

import re
import unicodedata


def clean_resume_text(text: str) -> str:
    """
    Cleans and normalizes extracted resume text conservatively.

    Operations performed:
    1. Replaces null bytes, form feeds, and weird control characters.
    2. Normalizes carriage returns (\r\n -> \n, \r -> \n).
    3. Normalizes unicode characters (NFKC normalization) to standard forms.
    4. Replaces non-breaking spaces and irregular horizontal whitespace with standard spaces.
    5. Strips trailing whitespace on individual lines.
    6. Normalizes multiple consecutive inline spaces into single spaces.
    7. Collapses excessive consecutive blank lines (3+ newlines -> 2 newlines).
    8. Trims leading and trailing whitespace from the full text block.

    What is PRESERVED:
    - Technical terms (e.g., C++, C#, .NET, Node.js, React.js, Scikit-learn, REST API).
    - Casing, numbers, bullets, colons, hyphens, and essential punctuation.
    - Section headers and paragraph separation.
    - Full words (no stemming, lemmatization, or stopword removal).

    :param text: Raw extracted text string from PDF.
    :return: Cleaned and normalized text string.
    """
    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Unicode normalization (NFKC ensures ligatures and special chars are normalized cleanly)
    cleaned = unicodedata.normalize("NFKC", text)

    # 2. Normalize carriage returns and line endings
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Remove non-printable control artifacts (except tab \t and newline \n)
    # Filter out null bytes (\x00), form feeds (\x0c), etc.
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", cleaned)

    # 4. Replace non-breaking spaces and other special horizontal spaces with standard space
    cleaned = re.sub(r"[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]", " ", cleaned)

    # 5. Process line by line to clean inline whitespace while preserving line structure
    lines = cleaned.split("\n")
    processed_lines = []
    for line in lines:
        # Normalize multiple horizontal spaces/tabs within the line to a single space
        line_clean = re.sub(r"[ \t]+", " ", line).strip()
        processed_lines.append(line_clean)

    # 6. Rejoin lines with newline
    cleaned = "\n".join(processed_lines)

    # 7. Collapse excessive consecutive blank lines (3 or more consecutive newlines -> 2 newlines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 8. Strip overall leading and trailing whitespace
    return cleaned.strip()
