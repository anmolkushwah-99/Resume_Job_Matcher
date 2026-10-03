"""
Unit Tests for TF-IDF Vectorizer and Cosine Similarity Matcher.
Verifies vectorizer fitting, shared vocabulary, similarity properties,
technical token preservation, and zero-vector handling.
"""

import pytest
import numpy as np
from src.matching.tfidf_matcher import TFIDFMatcher


class TestTFIDFMatcher:

    def test_matcher_initialization_and_config(self):
        matcher = TFIDFMatcher(ngram_range=(1, 2), min_df=1)
        config = matcher.get_config()
        assert config["model_name"] == "tfidf_cosine_baseline"
        assert config["ngram_range"] == [1, 2]
        assert config["min_df"] == 1
        assert config["lowercase"] is True

    def test_identical_documents_high_similarity(self):
        matcher = TFIDFMatcher()
        doc = "Python developer with machine learning, NLP, and TensorFlow expertise."
        score = matcher.compute_similarity(doc, doc)
        assert pytest.approx(score, rel=1e-3) == 1.0

    def test_unrelated_documents_low_similarity(self):
        matcher = TFIDFMatcher()
        doc_a = "Expert in Python, Flask, Django, PostgreSQL, Docker, and REST APIs."
        doc_b = "Fashion designer specializing in textile patterns, apparel styling, and Photoshop."
        score = matcher.compute_similarity(doc_a, doc_b)
        assert score < 0.15

    def test_related_documents_moderate_to_high_similarity(self):
        matcher = TFIDFMatcher()
        doc_resume = "Python backend developer with 3 years of experience in Flask, PostgreSQL, and REST APIs."
        doc_job = "Looking for a Python Backend Developer with Flask, PostgreSQL, and REST API experience."
        score = matcher.compute_similarity(doc_resume, doc_job)
        assert score > 0.30

    def test_technical_tokens_in_vocabulary(self):
        corpus = [
            "Experience with C++, C#, and .NET core.",
            "Frontend with React.js, Next.js, and Node.js.",
            "Machine learning using Scikit-learn and CI/CD pipelines.",
        ]
        matcher = TFIDFMatcher()
        matcher.fit(corpus)
        vocab = matcher.vectorizer.vocabulary_
        # Check that technical terms exist as distinct tokens
        assert any("c++" in term for term in vocab)
        assert any("c#" in term for term in vocab)
        assert any(".net" in term for term in vocab)
        assert any("node.js" in term for term in vocab)
        assert any("react.js" in term for term in vocab)

    def test_empty_and_zero_vector_handling(self):
        matcher = TFIDFMatcher()
        assert matcher.compute_similarity("", "Python Developer") == 0.0
        assert matcher.compute_similarity("Software Engineer", "") == 0.0
        assert matcher.compute_similarity("", "") == 0.0
        assert matcher.compute_similarity(None, None) == 0.0

    def test_compute_similarity_matrix(self):
        resumes = [
            "Python machine learning engineer with PyTorch and Pandas.",
            "React frontend developer with TypeScript and HTML CSS.",
        ]
        jobs = [
            "Looking for ML Engineer with Python and PyTorch.",
            "Hiring React Developer with TypeScript.",
        ]
        matcher = TFIDFMatcher()
        sim_matrix = matcher.compute_similarity_matrix(resumes, jobs)
        assert sim_matrix.shape == (2, 2)
        # Diagonal (matching domains) should be higher than off-diagonal
        assert sim_matrix[0, 0] > sim_matrix[0, 1]
        assert sim_matrix[1, 1] > sim_matrix[1, 0]

    def test_get_shared_terms(self):
        doc_a = "Python Flask backend developer with SQL and Docker."
        doc_b = "Hiring Python Flask engineer with Docker and Redis."
        matcher = TFIDFMatcher()
        matcher.fit([doc_a, doc_b])
        shared = matcher.get_shared_terms(doc_a, doc_b, top_k=10)
        terms = [s["term"] for s in shared]
        assert any("python" in t for t in terms)
        assert any("flask" in t for t in terms)
        assert any("docker" in t for t in terms)
