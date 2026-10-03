"""
Unit Tests for SVM Resume-Job Classifier Module.
Verifies training, prediction, decision function, feature importance,
model serialization, and persistence reloading.
"""

import pytest
import numpy as np
from pathlib import Path
from src.models.svm_classifier import SVMResumeJobClassifier


class TestSVMClassifier:

    @pytest.fixture
    def sample_training_data(self):
        train_texts = [
            "Python backend engineer Flask PostgreSQL REST API Docker",
            "Machine learning engineer NLP PyTorch Scikit-learn Pandas",
            "Frontend developer React TypeScript Next.js HTML CSS",
            "Java Spring Boot developer MySQL microservices Kafka",
            "Senior Python architect scalable microservices Redis FastAPI",
            "Textile apparel stylist graphic designer Photoshop fashion",
            "Civil construction site supervisor architecture masonry concrete",
            "Accounting bookkeeping finance audit tax compliance Excel ledger",
        ]
        y_train = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        return train_texts, y_train

    def test_initialization_and_config(self):
        clf = SVMResumeJobClassifier(C=1.0, class_weight="balanced")
        config = clf.get_config()
        assert config["model_type"] == "LinearSVC"
        assert config["C"] == 1.0
        assert config["class_weight"] == "balanced"
        assert config["is_trained"] is False

    def test_train_and_predict(self, sample_training_data):
        train_texts, y_train = sample_training_data
        clf = SVMResumeJobClassifier(C=1.0, random_state=42)
        clf.train(train_texts, y_train)

        assert clf.is_trained is True
        assert len(clf.feature_names) > 0

        # Predict matching sample
        test_match = ["Python developer with Flask and PostgreSQL REST APIs"]
        pred_match = clf.predict(test_match)
        assert pred_match[0] == 1

        # Predict non-matching sample
        test_non_match = ["Apparel stylist textile designing Photoshop"]
        pred_non_match = clf.predict(test_non_match)
        assert pred_non_match[0] == 0

    def test_decision_function(self, sample_training_data):
        train_texts, y_train = sample_training_data
        clf = SVMResumeJobClassifier(C=1.0, random_state=42)
        clf.train(train_texts, y_train)

        scores = clf.decision_function(["Python developer", "Textile designer"])
        assert len(scores) == 2
        assert isinstance(scores[0], (float, np.floating))
        # Python should have higher decision score than textile for matching
        assert scores[0] > scores[1]

    def test_get_top_features(self, sample_training_data):
        train_texts, y_train = sample_training_data
        clf = SVMResumeJobClassifier(C=1.0, random_state=42)
        clf.train(train_texts, y_train)

        top_feats = clf.get_top_features(top_k=5)
        assert "positive_features" in top_feats
        assert "negative_features" in top_feats
        assert len(top_feats["positive_features"]) <= 5
        assert len(top_feats["negative_features"]) <= 5

    def test_model_save_and_reload(self, sample_training_data, tmp_path):
        train_texts, y_train = sample_training_data
        clf = SVMResumeJobClassifier(C=0.5, class_weight="balanced", random_state=42)
        clf.train(train_texts, y_train)

        # Save to temporary path
        saved_dir = clf.save(tmp_path)
        assert (saved_dir / "tfidf_svm_model.joblib").exists()
        assert (saved_dir / "tfidf_svm_vectorizer.joblib").exists()
        assert (saved_dir / "tfidf_svm_metadata.json").exists()

        # Reload model
        reloaded_clf = SVMResumeJobClassifier.load(saved_dir)
        assert reloaded_clf.is_trained is True
        assert reloaded_clf.C == 0.5

        # Verify predictions remain identical
        test_sample = ["Python NLP engineer PyTorch Scikit-learn"]
        orig_pred = clf.predict(test_sample)
        reload_pred = reloaded_clf.predict(test_sample)
        assert orig_pred[0] == reload_pred[0]

    def test_untrained_model_raises_runtime_error(self):
        clf = SVMResumeJobClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(["Some text"])
        with pytest.raises(RuntimeError):
            clf.decision_function(["Some text"])

    def test_invalid_target_labels_raises_error(self):
        clf = SVMResumeJobClassifier()
        with pytest.raises(ValueError):
            clf.train(["text a", "text b"], [0, 2])
