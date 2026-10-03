"""
SVM Classifier Module for Resume-Job Matching.
Implements supervised linear Support Vector Machine classification on TF-IDF feature
vectors, with support for decision scores, feature coefficient interpretation,
and artifact persistence.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import joblib
from sklearn.svm import LinearSVC

from src.matching.tfidf_matcher import TFIDFMatcher

logger = logging.getLogger(__name__)


class SVMResumeJobClassifier:
    """
    Supervised Linear Support Vector Machine for Binary Resume-Job Matching.

    Features:
    - High-dimensional sparse TF-IDF text representation.
    - Linear maximum-margin hyper-plane separation.
    - Class-weight balancing for imbalanced datasets.
    - Transparent decision scores and top feature coefficient extraction.
    - Full artifact persistence and reloading via joblib.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        max_iter: int = 2000,
        ngram_range: tuple = (1, 2),
        min_df: int = 1,
    ):
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.ngram_range = ngram_range
        self.min_df = min_df

        self.model = LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.vectorizer: Optional[TFIDFMatcher] = None
        self.is_trained: bool = False
        self.feature_names: Optional[np.ndarray] = None

    def get_config(self) -> Dict[str, Any]:
        """Returns the classifier and vectorizer hyperparameter configuration."""
        return {
            "model_type": "LinearSVC",
            "C": self.C,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names) if self.feature_names is not None else 0,
        }

    def train(
        self,
        train_texts: List[str],
        y_train: Union[List[int], np.ndarray],
        vectorizer: Optional[TFIDFMatcher] = None
    ) -> "SVMResumeJobClassifier":
        """
        Fits the TF-IDF vectorizer strictly on training texts and trains the LinearSVC.

        :param train_texts: List of combined resume-job text strings from the training set.
        :param y_train: Binary match labels (0 or 1) for the training set.
        :param vectorizer: Optional pre-configured TFIDFMatcher.
        :return: self
        """
        if not train_texts or len(train_texts) == 0:
            raise ValueError("train_texts cannot be empty.")

        y_arr = np.asarray(y_train, dtype=int)
        unique_labels = np.unique(y_arr)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(f"Target labels must be binary [0, 1]. Found: {unique_labels}")

        # 1. Fit TF-IDF strictly on training texts
        if vectorizer is not None and vectorizer.is_fitted:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = TFIDFMatcher(ngram_range=self.ngram_range, min_df=self.min_df)
            self.vectorizer.fit(train_texts)

        # 2. Transform training texts
        X_train = self.vectorizer.transform(train_texts)
        self.feature_names = self.vectorizer.vectorizer.get_feature_names_out()

        # 3. Train Linear SVM
        self.model.fit(X_train, y_arr)
        self.is_trained = True

        logger.info(
            "Trained SVM Classifier (C=%s, class_weight=%s) on %d samples with %d features.",
            self.C, self.class_weight, len(train_texts), len(self.feature_names)
        )
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predicts binary match classes (0 = Non-Match, 1 = Match).

        :param texts: List of pair text documents.
        :return: 1D numpy array of binary predictions.
        """
        if not self.is_trained or self.vectorizer is None:
            raise RuntimeError("SVMResumeJobClassifier must be trained before predict().")

        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def decision_function(self, texts: List[str]) -> np.ndarray:
        """
        Computes raw linear decision scores (signed distance to the separating hyperplane).
        Positive values indicate match (Class 1), negative values indicate non-match (Class 0).

        :param texts: List of pair text documents.
        :return: 1D numpy array of continuous decision scores.
        """
        if not self.is_trained or self.vectorizer is None:
            raise RuntimeError("SVMResumeJobClassifier must be trained before decision_function().")

        X = self.vectorizer.transform(texts)
        return self.model.decision_function(X)

    def get_top_features(self, top_k: int = 15) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extracts the most influential positive and negative terms based on SVM hyperplane weights.

        :param top_k: Number of features to return per class.
        :return: Dictionary with 'positive_features' and 'negative_features'.
        """
        if not self.is_trained or self.feature_names is None:
            raise RuntimeError("Classifier must be trained before extracting top features.")

        coef = self.model.coef_[0]
        sorted_indices = np.argsort(coef)

        # Negative features (strongly predict Non-Match)
        top_neg_indices = sorted_indices[:top_k]
        neg_features = [
            {"term": str(self.feature_names[i]), "weight": float(round(coef[i], 4))}
            for i in top_neg_indices
        ]

        # Positive features (strongly predict Match)
        top_pos_indices = sorted_indices[::-1][:top_k]
        pos_features = [
            {"term": str(self.feature_names[i]), "weight": float(round(coef[i], 4))}
            for i in top_pos_indices
        ]

        return {
            "positive_features": pos_features,
            "negative_features": neg_features,
        }

    def save(self, model_dir: Union[str, Path]) -> Path:
        """
        Persists the trained SVM model, TF-IDF vectorizer, and metadata to disk.

        :param model_dir: Directory where artifacts should be stored.
        :return: Path to the directory.
        """
        if not self.is_trained or self.vectorizer is None:
            raise RuntimeError("Cannot save an untrained classifier.")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path / "tfidf_svm_model.joblib")
        joblib.dump(self.vectorizer, model_path / "tfidf_svm_vectorizer.joblib")

        metadata = self.get_config()
        with open(model_path / "tfidf_svm_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved SVM model artifacts to: %s", model_path)
        return model_path

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "SVMResumeJobClassifier":
        """
        Loads a persisted SVM classifier from disk.

        :param model_dir: Directory containing model artifacts.
        :return: Loaded SVMResumeJobClassifier instance.
        """
        model_path = Path(model_dir)
        clf_file = model_path / "tfidf_svm_model.joblib"
        vec_file = model_path / "tfidf_svm_vectorizer.joblib"
        meta_file = model_path / "tfidf_svm_metadata.json"

        if not clf_file.exists() or not vec_file.exists():
            raise FileNotFoundError(f"Missing model artifacts in directory: {model_path}")

        meta = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        instance = cls(
            C=meta.get("C", 1.0),
            class_weight=meta.get("class_weight", "balanced"),
            random_state=meta.get("random_state", 42),
            max_iter=meta.get("max_iter", 2000),
            ngram_range=tuple(meta.get("ngram_range", [1, 2])),
            min_df=meta.get("min_df", 1),
        )

        instance.model = joblib.load(clf_file)
        instance.vectorizer = joblib.load(vec_file)
        instance.is_trained = True
        instance.feature_names = instance.vectorizer.vectorizer.get_feature_names_out()

        logger.info("Loaded SVM classifier from %s with %d features.", model_path, len(instance.feature_names))
        return instance
