"""
TF-IDF Vectorization and Cosine Similarity Matching Engine.
Implements reproducible, shared-vocabulary TF-IDF feature representation
and cosine similarity computation for resume-job matching baselines.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaner import clean_resume_text
from src.preprocessing.tokenizer import tokenize

logger = logging.getLogger(__name__)


class TFIDFMatcher:
    """
    TF-IDF Vectorizer and Cosine Similarity Matcher for Resume-Job Matching.

    Features:
    - Shared vocabulary fitting across all benchmark documents.
    - Safe handling of technical tokens (C++, C#, .NET, Node.js, etc.).
    - Robust zero-vector / empty text protection.
    - Explainable shared feature extraction.
    """

    DEFAULT_CONFIG = {
        "model_name": "tfidf_cosine_baseline",
        "ngram_range": (1, 2),
        "min_df": 1,
        "lowercase": True,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": False,
    }

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        lowercase: bool = True,
        norm: str = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        max_features: Optional[int] = None,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.lowercase = lowercase
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.max_features = max_features

        # Custom token pattern that preserves technical tokens (C++, C#, .NET, CI/CD, Node.js)
        # while stripping trailing sentence punctuation from normal words
        self.token_pattern = r"(?u)(?:[cC]\+\+|[cC]\#|\.NET|\.net|\b[a-zA-Z0-9_\#\/\-\.]*[a-zA-Z0-9_\+\#\/\-]\b)"

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            lowercase=self.lowercase,
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
            max_features=self.max_features,
            token_pattern=self.token_pattern,
        )
        self.is_fitted = False

    def get_config(self) -> Dict[str, Any]:
        """Returns the vectorizer configuration dictionary."""
        return {
            "model_name": self.DEFAULT_CONFIG["model_name"],
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "lowercase": self.lowercase,
            "norm": self.norm,
            "use_idf": self.use_idf,
            "smooth_idf": self.smooth_idf,
            "sublinear_tf": self.sublinear_tf,
            "max_features": self.max_features,
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.is_fitted else 0,
        }

    def fit(self, corpus: List[str]) -> "TFIDFMatcher":
        """
        Fits the TF-IDF vectorizer on a corpus of text documents.

        :param corpus: List of text document strings.
        :return: self
        """
        valid_docs = [clean_resume_text(doc) for doc in corpus if doc and str(doc).strip()]
        if not valid_docs:
            valid_docs = ["empty document fallback"]

        self.vectorizer.fit(valid_docs)
        self.is_fitted = True
        return self

    def fit_transform(self, corpus: List[str]):
        """
        Fits vectorizer and returns transformed TF-IDF sparse matrix.
        """
        valid_docs = [clean_resume_text(doc) for doc in corpus]
        res = self.vectorizer.fit_transform(valid_docs)
        self.is_fitted = True
        return res

    def get_feature_names_out(self):
        """Returns feature names array from underlying scikit-learn vectorizer."""
        if hasattr(self.vectorizer, "get_feature_names_out"):
            return self.vectorizer.get_feature_names_out()
        return np.array([])

    def transform(self, documents: List[str]):
        """
        Transforms a list of documents into TF-IDF sparse matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("TFIDFMatcher must be fitted on a corpus before transform().")

        cleaned = [clean_resume_text(doc) if doc else "" for doc in documents]
        return self.vectorizer.transform(cleaned)


    def compute_similarity(self, doc_a: str, doc_b: str) -> float:
        """
        Computes cosine similarity between two text documents.
        If the vectorizer is not yet fitted, fits on the pair [doc_a, doc_b].

        :param doc_a: First document text (e.g. Resume).
        :param doc_b: Second document text (e.g. Job Description).
        :return: Cosine similarity score in range [0.0, 1.0].
        """
        clean_a = clean_resume_text(doc_a) if doc_a else ""
        clean_b = clean_resume_text(doc_b) if doc_b else ""

        # Safe empty handling
        if not clean_a.strip() or not clean_b.strip():
            return 0.0

        if not self.is_fitted:
            # Fit on the pair dynamically if no global corpus was set
            vec_matrix = self.vectorizer.fit_transform([clean_a, clean_b])
            vec_a = vec_matrix[0]
            vec_b = vec_matrix[1]
        else:
            vec_matrix = self.vectorizer.transform([clean_a, clean_b])
            vec_a = vec_matrix[0]
            vec_b = vec_matrix[1]

        # Check for zero norms
        if vec_a.nnz == 0 or vec_b.nnz == 0:
            return 0.0

        sim_matrix = cosine_similarity(vec_a, vec_b)
        score = float(sim_matrix[0][0])

        # Clamp floating point precision within [0.0, 1.0]
        score = max(0.0, min(1.0, score))
        return score

    def compute_similarity_matrix(self, resume_docs: List[str], job_docs: List[str]) -> np.ndarray:
        """
        Computes the pairwise cosine similarity matrix between a list of resume
        documents and a list of job documents.

        :param resume_docs: List of N resume documents.
        :param job_docs: List of M job documents.
        :return: (N x M) numpy array of cosine similarity scores in [0.0, 1.0].
        """
        if not self.is_fitted:
            all_docs = resume_docs + job_docs
            self.fit(all_docs)

        res_vecs = self.transform(resume_docs)
        job_vecs = self.transform(job_docs)

        sim_matrix = cosine_similarity(res_vecs, job_vecs)
        # Ensure values stay strictly bounded in [0.0, 1.0]
        return np.clip(sim_matrix, 0.0, 1.0)

    def get_shared_terms(self, doc_a: str, doc_b: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Extracts the highest-weighted overlapping n-grams / terms between two documents.
        Useful for explainable baseline reporting.

        :param doc_a: First document text.
        :param doc_b: Second document text.
        :param top_k: Number of top overlapping terms to return.
        :return: List of dicts with term, weight_a, weight_b, product.
        """
        if not doc_a or not doc_b or not self.is_fitted:
            return []

        vec_a = self.transform([doc_a]).toarray()[0]
        vec_b = self.transform([doc_b]).toarray()[0]

        feature_names = self.vectorizer.get_feature_names_out()
        product = vec_a * vec_b

        top_indices = np.argsort(product)[::-1]
        shared = []
        for idx in top_indices:
            if product[idx] <= 0:
                break
            shared.append({
                "term": str(feature_names[idx]),
                "weight_a": float(round(vec_a[idx], 4)),
                "weight_b": float(round(vec_b[idx], 4)),
                "overlap_score": float(round(product[idx], 4)),
            })
            if len(shared) >= top_k:
                break

        return shared
