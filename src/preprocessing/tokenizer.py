"""
NLP Tokenization, Stopword Handling, and Lemmatization Module.
Provides modular tokenization with special handling for technical terms
(e.g., C++, C#, .NET, Node.js, Scikit-learn, CI/CD).
"""

import logging
from typing import List, Optional, Set
import spacy
from spacy.symbols import ORTH
from spacy.language import Language

logger = logging.getLogger(__name__)

# Cached spaCy NLP model instance
_nlp_model: Optional[Language] = None

# Technical terms to register as single indivisible tokens in spaCy tokenizer
SPECIAL_TECHNICAL_TOKENS = [
    "C++", "c++", "C#", "c#", ".NET", ".net", "ASP.NET", "asp.net",
    "Node.js", "node.js", "React.js", "react.js", "Next.js", "next.js",
    "Vue.js", "vue.js", "Express.js", "express.js", "Scikit-learn", "scikit-learn",
    "CI/CD", "ci/cd", "REST API", "rest api", "TCP/IP", "tcp/ip",
    "PL/SQL", "pl/sql", "T-SQL", "t-sql", "Bash/Shell", "bash/shell",
    "Angular.js", "angular.js"
]

# Protected technical tokens that should NEVER be filtered out by stopword removal
PROTECTED_TECHNICAL_TOKENS: Set[str] = {
    "c++", "c#", "c", "r", "go", "ai", "ml", "dl", "nlp", "cv", "it",
    ".net", "node.js", "react.js", "next.js", "vue.js", "express.js",
    "scikit-learn", "ci/cd", "sql", "aws", "gcp", "git", "db", "etl"
}


def get_nlp_model() -> Language:
    """
    Returns the cached spaCy en_core_web_sm model instance.
    Configures special tokenization rules for technical terms.
    """
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model

    try:
        _nlp_model = spacy.load("en_core_web_sm")
    except Exception:
        # Fallback to creating blank English model if package is missing
        logger.warning("en_core_web_sm model not loaded directly; attempting spacy blank en.")
        _nlp_model = spacy.blank("en")

    # Add custom tokenizer special cases for technical terms
    for term in SPECIAL_TECHNICAL_TOKENS:
        _nlp_model.tokenizer.add_special_case(term, [{ORTH: term}])

    return _nlp_model


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text using spaCy with technical term preservation.
    Returns the list of raw token strings (excluding pure whitespace tokens).
    """
    if not text or not isinstance(text, str):
        return []

    nlp = get_nlp_model()
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_space]
    return tokens


def normalize_tokens(tokens: List[str]) -> List[str]:
    """
    Normalizes a list of token strings (strips whitespace, preserves case-sensitive
    distinctions where needed or returns lowercased representation).
    """
    if not tokens:
        return []
    return [t.strip().lower() for t in tokens if t and t.strip()]


def remove_stopwords(tokens: List[str], custom_stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Filters out common grammatical stopwords from a list of tokens
    while strictly preserving technical terms (such as 'c', 'r', 'ai', 'it', 'go').

    :param tokens: List of input token strings.
    :param custom_stopwords: Optional additional stopwords set.
    :return: Filtered token list.
    """
    if not tokens:
        return []

    nlp = get_nlp_model()
    spacy_stopwords = nlp.Defaults.stop_words

    stop_words = set(spacy_stopwords)
    if custom_stopwords:
        stop_words.update(custom_stopwords)

    filtered = []
    for token in tokens:
        t_clean = token.strip()
        if not t_clean:
            continue

        t_lower = t_clean.lower()

        # Always keep protected technical terms even if they are single letters or in stopwords
        if t_lower in PROTECTED_TECHNICAL_TOKENS:
            filtered.append(t_clean)
            continue

        if t_lower in stop_words:
            continue

        # Keep non-stopword tokens
        filtered.append(t_clean)

    return filtered


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lemmatizes a list of tokens using spaCy, while preserving proper nouns
    and technical acronyms.
    """
    if not tokens:
        return []

    text = " ".join(tokens)
    nlp = get_nlp_model()
    doc = nlp(text)

    lemmatized = []
    for token in doc:
        if token.is_space:
            continue

        t_lower = token.text.lower()
        if t_lower in PROTECTED_TECHNICAL_TOKENS:
            lemmatized.append(token.text)
        elif token.pos_ in ("PROPN", "NOUN") and token.text.isupper() and len(token.text) <= 5:
            # Preserve acronyms like AWS, SQL, REST, API
            lemmatized.append(token.text)
        else:
            lemmatized.append(token.lemma_)

    return lemmatized
