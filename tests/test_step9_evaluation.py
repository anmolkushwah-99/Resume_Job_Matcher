"""
Unit and Integration Tests for Step 9: Comprehensive Model Evaluation,
Error Analysis, Trade-Off Analysis, and Model Selection.

Tests data integrity, zero candidate leakage, common test set construction,
multi-model metric calculation, confusion matrix properties, error decomposition,
Step 8 behavior diagnostic, model complexity benchmarking, and selection framework.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluate_step9 import (
    load_dataset_split,
    construct_master_test_predictions,
    compute_comprehensive_metrics,
    analyze_validation_vs_test,
    analyze_threshold_sensitivity,
    perform_error_and_agreement_analysis,
    perform_skill_and_candidate_error_analysis,
    benchmark_model_complexities,
    benchmark_inference_latencies,
    generate_selection_matrix,
    run_comprehensive_step9_evaluation,
)


class TestStep9Evaluation:
    """Test suite for Step 9 evaluation, diagnostics, and selection framework."""

    @pytest.fixture
    def project_dirs(self):
        root_dir = Path(__file__).resolve().parent.parent
        data_dir = root_dir / "data"
        eval_dir = root_dir / "data" / "processed" / "evaluation"
        model_dir = root_dir / "models"
        return {"root": root_dir, "data": data_dir, "eval": eval_dir, "model": model_dir}

    def test_candidate_aware_split_integrity_and_zero_leakage(self, project_dirs):
        split_data = load_dataset_split(project_dirs["eval"])
        
        train_cands = set(split_data["train"]["candidates"])
        val_cands = set(split_data["validation"]["candidates"])
        test_cands = set(split_data["test"]["candidates"])

        # 1. Total candidates count
        assert len(train_cands) == 17
        assert len(val_cands) == 4
        assert len(test_cands) == 4
        assert len(train_cands | val_cands | test_cands) == 25

        # 2. Strict zero candidate overlap assertions
        assert len(train_cands & val_cands) == 0, "Train and Validation overlap detected!"
        assert len(train_cands & test_cands) == 0, "Train and Test overlap detected!"
        assert len(val_cands & test_cands) == 0, "Validation and Test overlap detected!"

        # 3. Expected held-out test candidates
        assert test_cands == {"RES001", "RES009", "RES017", "RES024"}

    def test_construct_master_test_predictions(self, project_dirs):
        df = construct_master_test_predictions(project_dirs["data"], project_dirs["eval"])
        
        assert len(df) == 18, "Test set must contain exactly 18 resume-job pairs."
        assert set(df["resume_id"].unique()) == {"RES001", "RES009", "RES017", "RES024"}
        
        # Label balance
        pos_count = (df["true_label"] == 1).sum()
        neg_count = (df["true_label"] == 0).sum()
        assert pos_count == 5
        assert neg_count == 13

        # Model columns present
        expected_cols = [
            "pair_id", "resume_id", "job_id", "true_label",
            "cosine_score", "cosine_pred",
            "svm_score", "svm_pred",
            "ann_score", "ann_pred",
            "lstm_score", "lstm_pred"
        ]
        for col in expected_cols:
            assert col in df.columns
            assert df[col].notnull().all(), f"Column {col} has null values."

    def test_compute_comprehensive_metrics_math(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_pred = np.array([1, 0, 0, 0, 1])
        y_score = np.array([0.9, 0.4, 0.1, 0.2, 0.8])

        metrics = compute_comprehensive_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            model_name="test_model",
            representation="test_rep",
            architecture="test_arch",
            threshold=0.5
        )

        assert metrics["total_test_samples"] == 5
        assert metrics["true_positive"] == 1
        assert metrics["false_negative"] == 1
        assert metrics["true_negative"] == 2
        assert metrics["false_positive"] == 1
        assert metrics["accuracy"] == 0.60
        assert metrics["precision"] == 0.50
        assert metrics["recall"] == 0.50
        assert metrics["specificity"] == 2.0 / 3.0 or round(metrics["specificity"], 4) == 0.6667
        assert metrics["roc_auc"] is not None
        assert metrics["pr_auc"] is not None

    def test_step8_lstm_all_positive_diagnostic(self, project_dirs):
        df = construct_master_test_predictions(project_dirs["data"], project_dirs["eval"])
        
        # Verify LSTM output characteristics
        lstm_scores = df["lstm_score"].values
        lstm_preds = df["lstm_pred"].values

        assert np.all(lstm_scores >= 0.0) and np.all(lstm_scores <= 1.0)
        assert np.all(lstm_scores >= 0.30), "All LSTM test scores must be >= 0.30, triggering positive predictions."
        assert np.all(lstm_preds == 1), "LSTM should predict 1 for all 18 pairs under theta=0.30."
        
        # Verify Specificity is 0.0 and Recall is 1.0
        cm = compute_comprehensive_metrics(df["true_label"].values, lstm_preds, lstm_scores)
        assert cm["recall"] == 1.0
        assert cm["specificity"] == 0.0
        assert cm["true_negative"] == 0
        assert cm["false_positive"] == 13

    def test_step6_svm_rejection_capability(self, project_dirs):
        df = construct_master_test_predictions(project_dirs["data"], project_dirs["eval"])
        svm_preds = df["svm_pred"].values
        cm = compute_comprehensive_metrics(df["true_label"].values, svm_preds)

        # SVM has the highest true negative rejection count (TN=8)
        assert cm["true_negative"] == 8
        assert cm["false_positive"] == 5
        assert cm["specificity"] == round(8 / 13, 4)
        assert cm["accuracy"] == 0.6111

    def test_model_agreement_and_error_decomposition(self, project_dirs):
        df = construct_master_test_predictions(project_dirs["data"], project_dirs["eval"])
        annotated_df, pairwise_df, stats = perform_error_and_agreement_analysis(df)

        assert len(annotated_df) == 18
        assert "agreement_category" in annotated_df.columns
        assert stats["total_test_pairs"] == 18
        assert stats["unanimous_correct_count"] >= 0
        assert stats["unanimous_wrong_count"] >= 0
        assert pairwise_df.shape == (4, 4)
        assert np.all(np.diag(pairwise_df.values) == 1.0)

    def test_candidate_level_metrics(self, project_dirs):
        df = construct_master_test_predictions(project_dirs["data"], project_dirs["eval"])
        _, cand_df = perform_skill_and_candidate_error_analysis(df, project_dirs["data"])

        assert len(cand_df) == 4
        assert set(cand_df["candidate_id"]) == {"RES001", "RES009", "RES017", "RES024"}
        assert cand_df["pairs_count"].sum() == 18

    def test_model_complexity_benchmarks(self, project_dirs):
        comp_df = benchmark_model_complexities(project_dirs["model"])
        assert len(comp_df) == 4
        
        # Verify parameter counts
        cosine_row = comp_df[comp_df["model_name"] == "tfidf_cosine_baseline"].iloc[0]
        svm_row = comp_df[comp_df["model_name"] == "tfidf_svm"].iloc[0]
        ann_row = comp_df[comp_df["model_name"] == "tfidf_ann"].iloc[0]
        lstm_row = comp_df[comp_df["model_name"] == "embedding_lstm"].iloc[0]

        assert cosine_row["trainable_parameters"] == 0
        assert svm_row["trainable_parameters"] == 1661
        assert ann_row["trainable_parameters"] > 200000
        assert lstm_row["trainable_parameters"] > 40000

    def test_selection_matrix_structure(self, project_dirs):
        metrics_df = pd.read_csv(project_dirs["eval"] / "step9_metric_comparison.csv")
        sel_df = generate_selection_matrix(metrics_df)

        assert len(sel_df) == 4
        assert "model" in sel_df.columns
        assert "step10_suitability" in sel_df.columns
        assert "selection_notes" in sel_df.columns

        # Verify SVM is selected as recommended primary model
        svm_rec = sel_df[sel_df["model"] == "tfidf_svm"].iloc[0]
        assert "RECOMMENDED PRIMARY MODEL" in svm_rec["step10_suitability"]

    def test_full_step9_evaluation_execution(self, tmp_path):
        root_dir = Path(__file__).resolve().parent.parent
        data_dir = root_dir / "data"
        model_dir = root_dir / "models"
        eval_dir = tmp_path / "evaluation"

        # Copy split file to tmp_path evaluation dir for isolated test
        eval_dir.mkdir(parents=True, exist_ok=True)
        split_src = root_dir / "data" / "processed" / "evaluation" / "svm_dataset_split.json"
        with open(split_src, "r", encoding="utf-8") as f:
            split_json = json.load(f)
        with open(eval_dir / "svm_dataset_split.json", "w", encoding="utf-8") as f:
            json.dump(split_json, f)

        # Copy previous predictions
        for f_name in [
            "tfidf_pair_scores.csv",
            "svm_test_predictions.csv",
            "ann_test_predictions.csv",
            "lstm_test_predictions.csv",
            "tfidf_baseline_summary.json",
            "svm_metrics.json",
            "ann_metrics.json",
            "lstm_metrics.json"
        ]:
            src_f = root_dir / "data" / "processed" / "evaluation" / f_name
            if src_f.exists():
                with open(src_f, "r", encoding="utf-8") as sf:
                    content = sf.read()
                with open(eval_dir / f_name, "w", encoding="utf-8") as df:
                    df.write(content)

        summary = run_comprehensive_step9_evaluation(
            data_dir=data_dir,
            eval_dir=eval_dir,
            model_dir=model_dir
        )

        assert "models" in summary
        assert "selection" in summary
        assert summary["selection"]["recommended_primary_model"] == "tfidf_svm"
        assert (eval_dir / "step9_master_test_predictions.csv").exists()
        assert (eval_dir / "step9_metric_comparison.csv").exists()
        assert (eval_dir / "step9_metrics.json").exists()
        assert (eval_dir / "step9_confusion_matrices.png").exists()
        assert (eval_dir / "step9_roc_curves.png").exists()
