"""
Unit Tests for NLP Tokenizer, Stopword Handler, and Lemmatizer.
Tests technical term preservation, stopwords filtering, and lemmatization.
"""

from src.preprocessing.tokenizer import (
    tokenize,
    normalize_tokens,
    remove_stopwords,
    lemmatize_tokens,
)


class TestTokenizer:
    """Test suite for tokenization and normalization."""

    def test_tokenize_standard_text(self):
        """Verify standard English text tokenization."""
        text = "Software developer with experience in machine learning."
        tokens = tokenize(text)
        assert "Software" in tokens
        assert "developer" in tokens
        assert "machine" in tokens
        assert "learning" in tokens

    def test_tokenize_technical_terms_preserved(self):
        """Verify technical terms are not split erroneously."""
        text = "Skills: C++, C#, .NET, Node.js, React.js, Scikit-learn, CI/CD, ASP.NET."
        tokens = tokenize(text)

        assert "C++" in tokens
        assert "C#" in tokens
        assert ".NET" in tokens
        assert "Node.js" in tokens
        assert "React.js" in tokens
        assert "Scikit-learn" in tokens
        assert "CI/CD" in tokens
        assert "ASP.NET" in tokens

    def test_normalize_tokens(self):
        """Verify token normalization downcases and cleans tokens."""
        tokens = ["Python", "Flask", "SQL", "Scikit-learn"]
        normalized = normalize_tokens(tokens)
        assert normalized == ["python", "flask", "sql", "scikit-learn"]

    def test_remove_stopwords_protects_technical_tokens(self):
        """Verify standard stopwords are removed while technical tokens (e.g. C, R, AI, Go) are preserved."""
        raw_tokens = [
            "Experienced", "software", "developer", "with", "strong", "knowledge",
            "of", "Python", "and", "C++", "and", "AI", "and", "Go"
        ]
        filtered = remove_stopwords(raw_tokens)

        # Stopwords removed
        assert "with" not in filtered
        assert "of" not in filtered
        assert "and" not in filtered

        # Content and technical tokens preserved
        assert "Experienced" in filtered
        assert "software" in filtered
        assert "developer" in filtered
        assert "Python" in filtered
        assert "C++" in filtered
        assert "AI" in filtered
        assert "Go" in filtered

    def test_lemmatization(self):
        """Verify verbs and plural nouns are lemmatized while technical terms are protected."""
        tokens = ["developers", "developing", "managed", "applications", "Python", "AWS"]
        lemmas = lemmatize_tokens(tokens)

        assert "developer" in lemmas
        assert "develop" in lemmas
        assert "manage" in lemmas
        assert "application" in lemmas
        assert "Python" in lemmas
        assert "AWS" in lemmas

    def test_empty_inputs(self):
        """Verify graceful handling of empty or None inputs."""
        assert tokenize("") == []
        assert tokenize(None) == []
        assert normalize_tokens([]) == []
        assert remove_stopwords([]) == []
        assert lemmatize_tokens([]) == []
