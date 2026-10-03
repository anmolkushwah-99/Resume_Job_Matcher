"""
Unit tests for Step 10 Model Registry (src/matching/model_registry.py).
"""

import pytest
from pathlib import Path
from src.matching.model_registry import ModelRegistry


def test_model_registry_primary_model():
    """Verify primary model is correctly designated as tfidf_svm."""
    primary = ModelRegistry.get_primary_model_name()
    assert primary == "tfidf_svm"
    info = ModelRegistry.get_model_info("tfidf_svm")
    assert info["is_primary"] is True
    assert info["model_type"] == "Linear Support Vector Classifier (LinearSVC)"
    assert info["threshold"] == 0.0
    assert info["expected_feature_count"] == 1661


def test_model_registry_all_registered_models():
    """Verify all 4 models from Steps 5-8 are registered."""
    for m in ["tfidf_svm", "tfidf_ann", "tfidf_cosine_baseline", "embedding_lstm"]:
        info = ModelRegistry.get_model_info(m)
        assert info["model_name"] == m
        assert "decision_rule" in info
        assert "score_semantics" in info


def test_model_registry_unknown_model():
    """Verify querying an unknown model raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model name"):
        ModelRegistry.get_model_info("non_existent_model")


def test_model_registry_verify_artifacts():
    """Verify actual model artifacts on disk pass validation."""
    models_dir = ModelRegistry.get_models_dir()
    assert ModelRegistry.verify_model_artifacts("tfidf_svm", models_dir) is True


def test_model_registry_missing_artifact(tmp_path):
    """Verify artifact verification fails gracefully when model file is missing."""
    assert ModelRegistry.verify_model_artifacts("tfidf_svm", tmp_path) is False


def test_model_registry_check_health():
    """Verify health check returns comprehensive status dictionary."""
    health = ModelRegistry.check_health()
    assert "primary_model" in health
    assert health["primary_model"] == "tfidf_svm"
    assert health["primary_model_ready"] is True
    assert "models" in health
    assert "tfidf_svm" in health["models"]
    assert health["models"]["tfidf_svm"]["ready"] is True
