"""
Integration Tests for Supervised SVM Evaluation Script.
Verifies end-to-end execution of evaluation workflow, generated artifacts,
cross-validation metrics, and Step 5 baseline comparison.
"""

import pytest
import json
from pathlib import Path
import pandas as pd
from src.evaluate_svm import run_svm_evaluation


class TestSVMEvaluation:

    def test_run_svm_evaluation_artifacts(self, tmp_path):
        out_dir = tmp_path / "eval"
        mod_dir = tmp_path / "models"

        summary = run_svm_evaluation(
            output_dir=out_dir,
            model_dir=mod_dir,
            random_state=42
        )

        assert "dataset_split" in summary
        assert "hyperparameter_selection" in summary
        assert "grouped_cross_validation_5fold" in summary
        assert "final_test_evaluation" in summary
        assert "step5_test_comparison" in summary

        # Check candidate overlap == 0
        overlap = summary["dataset_split"]["candidate_overlap"]
        assert overlap["train_validation_overlap"] == 0
        assert overlap["train_test_overlap"] == 0
        assert overlap["validation_test_overlap"] == 0

        # Check generated files in out_dir
        split_file = out_dir / "svm_dataset_split.json"
        hp_file = out_dir / "svm_hyperparameter_results.csv"
        cv_file = out_dir / "svm_cross_validation_results.csv"
        test_preds_file = out_dir / "svm_test_predictions.csv"
        metrics_file = out_dir / "svm_metrics.json"
        report_file = out_dir / "svm_classification_report.txt"
        feats_file = out_dir / "svm_feature_importance.csv"
        comp_file = out_dir / "svm_baseline_comparison.csv"
        cm_plot = out_dir / "svm_confusion_matrix.png"
        dist_plot = out_dir / "svm_split_distribution.png"

        assert split_file.exists()
        assert hp_file.exists()
        assert cv_file.exists()
        assert test_preds_file.exists()
        assert metrics_file.exists()
        assert report_file.exists()
        assert feats_file.exists()
        assert comp_file.exists()
        assert cm_plot.exists()
        assert dist_plot.exists()

        # Check model artifacts in mod_dir
        assert (mod_dir / "tfidf_svm_model.joblib").exists()
        assert (mod_dir / "tfidf_svm_vectorizer.joblib").exists()
        assert (mod_dir / "tfidf_svm_metadata.json").exists()
