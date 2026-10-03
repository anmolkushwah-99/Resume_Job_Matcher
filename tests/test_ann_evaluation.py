"""
Integration Tests for ANN Evaluation Pipeline (Step 7).
Tests end-to-end evaluation execution, zero candidate leakage, and artifact generation.
"""

from pathlib import Path
import json
import pytest
from src.evaluate_ann import run_ann_evaluation


class TestANNEvaluation:
    """Tests full ANN evaluation pipeline and artifact generation."""

    def test_run_ann_evaluation_artifacts(self, tmp_path):
        data_dir = Path(__file__).resolve().parent.parent / "data"
        out_dir = tmp_path / "evaluation"
        mod_dir = tmp_path / "models"

        summary = run_ann_evaluation(
            data_dir=data_dir,
            output_dir=out_dir,
            model_dir=mod_dir,
            random_state=42
        )

        assert "dataset" in summary
        assert "training" in summary
        assert "validation" in summary
        assert "test" in summary
        assert "comparison" in summary

        # 1. Zero candidate overlap verification
        spl = summary["dataset"]
        assert spl["candidate_overlap"]["train_validation_overlap"] == 0
        assert spl["candidate_overlap"]["train_test_overlap"] == 0
        assert spl["candidate_overlap"]["validation_test_overlap"] == 0

        # 2. Check generated file artifacts
        assert (out_dir / "ann_dataset_split.json").exists()
        assert (out_dir / "ann_hyperparameter_results.csv").exists()
        assert (out_dir / "ann_threshold_results.csv").exists()
        assert (out_dir / "ann_training_history.csv").exists()
        assert (out_dir / "ann_test_predictions.csv").exists()
        assert (out_dir / "ann_metrics.json").exists()
        assert (out_dir / "ann_classification_report.txt").exists()
        assert (out_dir / "ann_model_comparison.csv").exists()
        assert (out_dir / "ann_confusion_matrix.png").exists()
        assert (out_dir / "ann_training_curves.png").exists()

        # 3. Check model persistence files
        assert (mod_dir / "tfidf_ann_model.keras").exists()
        assert (mod_dir / "tfidf_ann_vectorizer.joblib").exists()
        assert (mod_dir / "tfidf_ann_metadata.json").exists()
