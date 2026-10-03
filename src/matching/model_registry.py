"""
Model Registry Module (Step 10).
Manages registered matching models, artifact paths, compatibility checks,
and operational readiness status.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import joblib

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry of validated matching models."""

    _REGISTRY: Dict[str, Dict[str, Any]] = {
        "tfidf_svm": {
            "model_name": "tfidf_svm",
            "model_type": "Linear Support Vector Classifier (LinearSVC)",
            "is_primary": True,
            "decision_rule": "decision_score >= 0.0 -> Match",
            "threshold": 0.0,
            "score_semantics": "LinearSVC decision_function continuous score",
            "model_file": "tfidf_svm_model.joblib",
            "vectorizer_file": "tfidf_svm_vectorizer.joblib",
            "metadata_file": "tfidf_svm_metadata.json",
            "expected_feature_count": 1661
        },
        "tfidf_ann": {
            "model_name": "tfidf_ann",
            "model_type": "Dense Multi-Layer Perceptron (MLP)",
            "is_primary": False,
            "decision_rule": "ann_score >= 0.30 -> Match",
            "threshold": 0.30,
            "score_semantics": "ANN Sigmoid model score in [0.0, 1.0]",
            "model_file": "tfidf_ann_model.keras",
            "vectorizer_file": "tfidf_ann_vectorizer.joblib",
            "metadata_file": "tfidf_ann_metadata.json",
            "expected_feature_count": 1661
        },
        "tfidf_cosine_baseline": {
            "model_name": "tfidf_cosine_baseline",
            "model_type": "Unsupervised TF-IDF Cosine Similarity",
            "is_primary": False,
            "decision_rule": "similarity_score >= 0.05 -> Match",
            "threshold": 0.05,
            "score_semantics": "Cosine similarity in [0.0, 1.0]",
            "model_file": None,
            "vectorizer_file": None,
            "metadata_file": None,
            "expected_feature_count": 1661
        },
        "embedding_lstm": {
            "model_name": "embedding_lstm",
            "model_type": "Recurrent Sequence Model (Embedding + LSTM)",
            "is_primary": False,
            "decision_rule": "lstm_score >= 0.30 -> Match",
            "threshold": 0.30,
            "score_semantics": "LSTM Sigmoid model score in [0.0, 1.0]",
            "model_file": "embedding_lstm_model.keras",
            "vectorizer_file": "embedding_lstm_tokenizer.joblib",
            "metadata_file": "embedding_lstm_metadata.json",
            "expected_feature_count": None
        }
    }

    @classmethod
    def get_models_dir(cls) -> Path:
        """Returns the project models directory."""
        return Path(__file__).resolve().parent.parent.parent / "models"

    @classmethod
    def get_primary_model_name(cls) -> str:
        """Returns the Step 9 selected primary model identifier."""
        return "tfidf_svm"

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Retrieves registry information for a specific model."""
        if model_name not in cls._REGISTRY:
            raise ValueError(f"Unknown model name '{model_name}'. Available models: {list(cls._REGISTRY.keys())}")
        return dict(cls._REGISTRY[model_name])

    @classmethod
    def verify_model_artifacts(cls, model_name: str = "tfidf_svm", model_dir: Optional[Path] = None) -> bool:
        """
        Verifies that model artifacts exist on disk and are mutually compatible.
        """
        if model_dir is None:
            model_dir = cls.get_models_dir()

        info = cls.get_model_info(model_name)
        model_file = info.get("model_file")
        vectorizer_file = info.get("vectorizer_file")

        if model_file is None:
            return True  # E.g., baseline cosine matcher

        m_path = model_dir / model_file
        v_path = model_dir / vectorizer_file if vectorizer_file else None

        if not m_path.exists():
            logger.error("Model artifact missing: %s", m_path)
            return False

        if v_path and not v_path.exists():
            logger.error("Vectorizer artifact missing: %s", v_path)
            return False

        # Verify dimension compatibility for SVM
        if model_name == "tfidf_svm":
            try:
                svm_model = joblib.load(m_path)
                vectorizer = joblib.load(v_path)
                if hasattr(vectorizer, "get_feature_names_out"):
                    feat_names = vectorizer.get_feature_names_out()
                    n_features_vec = len(feat_names) if hasattr(feat_names, "__len__") else 0
                elif hasattr(vectorizer, "vectorizer") and hasattr(vectorizer.vectorizer, "get_feature_names_out"):
                    feat_names = vectorizer.vectorizer.get_feature_names_out()
                    n_features_vec = len(feat_names) if hasattr(feat_names, "__len__") else 0
                elif hasattr(vectorizer, "vocabulary_"):
                    n_features_vec = len(vectorizer.vocabulary_)
                else:
                    n_features_vec = 0

                n_features_svm = svm_model.coef_.shape[1]
                if n_features_vec > 0 and n_features_vec != n_features_svm:
                    logger.error(
                        "Dimension mismatch: Vectorizer features (%d) != SVM coefficients (%d)",
                        n_features_vec, n_features_svm
                    )
                    return False
            except Exception as e:
                logger.error("Error verifying SVM artifact compatibility: %s", e)
                return False

        return True


    @classmethod
    def check_health(cls, model_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Returns readiness status for all registered matching models."""
        if model_dir is None:
            model_dir = cls.get_models_dir()

        status = {}
        for m_name in cls._REGISTRY:
            status[m_name] = {
                "ready": cls.verify_model_artifacts(m_name, model_dir),
                "is_primary": cls._REGISTRY[m_name]["is_primary"],
                "model_type": cls._REGISTRY[m_name]["model_type"]
            }
        return {
            "primary_model": cls.get_primary_model_name(),
            "primary_model_ready": status[cls.get_primary_model_name()]["ready"],
            "models": status
        }
