"""
Integration Tests for LSTM Evaluation Pipeline (Step 8).
Tests sequence statistics generation, hyperparameter search, zero candidate leakage,
artifact generation, and 4-model comparison integrity.
"""

from pathlib import Path
import json
import pytest
from src.evaluate_lstm import run_lstm_evaluation


class TestLSTMEvaluation:
    """Tests full LSTM evaluation pipeline and artifact generation."""

    def test_run_lstm_evaluation_artifacts(self, tmp_path):
        data_dir = Path(__file__).resolve().parent.parent / "data"
        out_dir = tmp_path / "evaluation"
        mod_dir = tmp_path / "models"

        summary = run_lstm_evaluation(
            data_dir=data_dir,
            output_dir=out_dir,
            model_dir=mod_dir,
            random_state=42
        )

        assert "dataset" in summary
        assert "tokenization" in summary
        assert "sequence_configuration" in summary
        assert "training" in summary
        assert "validation" in summary
        assert "test" in summary
        assert "comparison" in summary

        # 1. Zero candidate overlap verification
        spl = summary["dataset"]
        assert spl["candidate_overlap"]["train_validation_overlap"] == 0
        assert spl["candidate_overlap"]["train_test_overlap"] == 0
        assert spl["candidate_overlap"]["validation_test_overlap"] == 0

        # 2. Sequence statistics verification
        seq_cfg = summary["sequence_configuration"]
        assert seq_cfg["selected_max_length"] > 0
        assert "training_distribution" in seq_cfg
        assert seq_cfg["training_distribution"]["percentile_95"] <= seq_cfg["selected_max_length"]

        # 3. Check generated file artifacts
        assert (out_dir / "lstm_dataset_split.json").exists()
        assert (out_dir / "lstm_sequence_statistics.json").exists()
        assert (out_dir / "lstm_hyperparameter_results.csv").exists()
        assert (out_dir / "lstm_threshold_results.csv").exists()
        assert (out_dir / "lstm_training_history.csv").exists()
        assert (out_dir / "lstm_test_predictions.csv").exists()
        assert (out_dir / "lstm_metrics.json").exists()
        assert (out_dir / "lstm_classification_report.txt").exists()
        assert (out_dir / "lstm_model_comparison.csv").exists()
        assert (out_dir / "lstm_confusion_matrix.png").exists()
        assert (out_dir / "lstm_training_curves.png").exists()

        # 4. Check model persistence files
        assert (mod_dir / "embedding_lstm_model.keras").exists()
        assert (mod_dir / "embedding_lstm_tokenizer.joblib").exists()
        assert (mod_dir / "embedding_lstm_metadata.json").exists()
