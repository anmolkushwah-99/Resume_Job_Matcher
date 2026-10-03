"""
Unit Tests for ANN / MLP Classifier Module (Step 7).
Tests model initialization, training, dynamic input dimensions, sigmoid score ranges,
threshold prediction conversion, model persistence, and reloading.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.models.ann_classifier import ANNResumeJobClassifier, build_ann_architecture


class TestANNClassifier:
    """Tests the ANNResumeJobClassifier implementation."""

    @pytest.fixture
    def sample_texts_and_labels(self):
        texts = [
            "python machine learning deep learning neural network tensorflow",
            "pytorch data science artificial intelligence natural language processing",
            "deep learning engineer computer vision mlops tensorflow keras",
            "accounting tax financial reporting general ledger audit",
            "cook chef culinary food preparation kitchen management",
            "python developer backend flask django postgresql rest api",
            "receptionist office administrator phone scheduling customer service",
            "machine learning researcher nlp transformer llm bert python",
        ]
        labels = np.array([1, 1, 1, 0, 0, 1, 0, 1])
        return texts, labels

    def test_initialization_and_config(self):
        clf = ANNResumeJobClassifier(
            hidden_units=(128, 64),
            dropout_rates=(0.30, 0.20),
            learning_rate=0.001,
            batch_size=8,
            epochs=20,
            threshold=0.45,
            class_weight_mode="balanced",
            random_state=42,
        )
        cfg = clf.get_config()
        assert cfg["model_type"] == "ANN_MLP"
        assert cfg["hidden_units"] == [128, 64]
        assert cfg["dropout_rates"] == [0.30, 0.20]
        assert cfg["learning_rate"] == 0.001
        assert cfg["threshold"] == 0.45
        assert cfg["is_trained"] is False

    def test_build_ann_architecture_shapes(self):
        input_dim = 250
        model = build_ann_architecture(input_dim=input_dim, hidden_units=(64, 32), dropout_rates=(0.2, 0.1))
        assert model.input_shape == (None, 250)
        assert model.output_shape == (None, 1)

    def test_train_and_predict_score_range(self, sample_texts_and_labels):
        texts, labels = sample_texts_and_labels
        clf = ANNResumeJobClassifier(
            hidden_units=(32, 16),
            dropout_rates=(0.1, 0.1),
            learning_rate=0.01,
            epochs=15,
            batch_size=4,
            threshold=0.50,
            random_state=42,
        )
        clf.train(train_texts=texts, y_train=labels)

        assert clf.is_trained is True
        assert clf.feature_count > 0

        scores = clf.predict_score(texts)
        assert len(scores) == len(texts)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

        preds = clf.predict(texts)
        assert len(preds) == len(texts)
        assert set(preds).issubset({0, 1})

    def test_threshold_modification(self, sample_texts_and_labels):
        texts, labels = sample_texts_and_labels
        clf = ANNResumeJobClassifier(
            hidden_units=(32, 16),
            epochs=10,
            batch_size=4,
            threshold=0.50,
            random_state=42,
        )
        clf.train(texts, labels)

        scores = clf.predict_score(texts)
        clf.set_threshold(0.0)
        assert np.all(clf.predict(texts) == 1)

        clf.set_threshold(1.0)
        assert np.all(clf.predict(texts) == 0)

        with pytest.raises(ValueError):
            clf.set_threshold(1.5)

    def test_model_save_and_reload(self, sample_texts_and_labels):
        texts, labels = sample_texts_and_labels
        clf = ANNResumeJobClassifier(
            hidden_units=(32, 16),
            dropout_rates=(0.1, 0.1),
            epochs=10,
            batch_size=4,
            threshold=0.40,
            random_state=42,
        )
        clf.train(texts, labels)
        scores_orig = clf.predict_score(texts)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            clf.save(save_path)

            assert (save_path / "tfidf_ann_model.keras").exists()
            assert (save_path / "tfidf_ann_vectorizer.joblib").exists()
            assert (save_path / "tfidf_ann_metadata.json").exists()

            loaded_clf = ANNResumeJobClassifier.load(save_path)
            assert loaded_clf.is_trained is True
            assert loaded_clf.threshold == 0.40
            assert loaded_clf.feature_count == clf.feature_count

            scores_loaded = loaded_clf.predict_score(texts)
            np.testing.assert_allclose(scores_orig, scores_loaded, rtol=1e-4, atol=1e-4)

    def test_untrained_model_raises_runtime_error(self):
        clf = ANNResumeJobClassifier()
        with pytest.raises(RuntimeError):
            clf.predict_score(["python data engineer"])
        with pytest.raises(RuntimeError):
            clf.save("tmp/models")

    def test_invalid_target_labels_raises_error(self):
        clf = ANNResumeJobClassifier()
        with pytest.raises(ValueError, match="binary"):
            clf.train(["text a", "text b"], [0, 2])

    def test_empty_train_texts_raises_error(self):
        clf = ANNResumeJobClassifier()
        with pytest.raises(ValueError):
            clf.train([], [])
