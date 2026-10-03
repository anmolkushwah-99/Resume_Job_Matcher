"""
Unit & Integration Tests for TF-IDF Baseline Evaluation Script.
Verifies evaluation execution, generated artifact formats, and metric computations.
"""

import pytest
import json
from pathlib import Path
import pandas as pd
from src.evaluate_tfidf_baseline import run_tfidf_evaluation


class TestTFIDFEvaluation:

    def test_run_tfidf_evaluation_artifacts(self, tmp_path):
        summary = run_tfidf_evaluation(output_dir=tmp_path)

        assert "dataset" in summary
        assert summary["dataset"]["total_pairs"] == 120
        assert summary["dataset"]["positive_pairs"] == 38
        assert summary["dataset"]["negative_pairs"] == 82

        assert "score_statistics" in summary
        pos_stats = summary["score_statistics"]["positive_pairs"]
        neg_stats = summary["score_statistics"]["negative_pairs"]

        # Positive pairs should have noticeably higher mean and median than negative pairs
        assert pos_stats["mean"] > neg_stats["mean"]
        assert pos_stats["median"] > neg_stats["median"]

        # Verify generated files in output_dir
        pair_scores_file = tmp_path / "tfidf_pair_scores.csv"
        thresh_file = tmp_path / "tfidf_threshold_results.csv"
        summary_file = tmp_path / "tfidf_baseline_summary.json"
        dist_plot = tmp_path / "similarity_distribution.png"
        metrics_plot = tmp_path / "threshold_metrics.png"
        cm_plot = tmp_path / "confusion_matrix.png"

        assert pair_scores_file.exists()
        assert thresh_file.exists()
        assert summary_file.exists()
        assert dist_plot.exists()
        assert metrics_plot.exists()
        assert cm_plot.exists()

        # Check DataFrame content
        df_scores = pd.read_csv(pair_scores_file)
        assert len(df_scores) == 120
        assert "similarity_score" in df_scores.columns

        df_thresh = pd.read_csv(thresh_file)
        assert len(df_thresh) >= 10
        assert "f1_score" in df_thresh.columns
        assert "accuracy" in df_thresh.columns
