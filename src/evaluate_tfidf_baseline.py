"""
TF-IDF Baseline Evaluation and Threshold Analysis Script.
Phase 5: Evaluates the TF-IDF Cosine Similarity baseline on the 120 labeled
resume-job benchmark pairs without label leakage.
Generates evaluation tables, summary JSON, and metric plots in data/processed/evaluation/.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from src.matching.batch_matcher import run_batch_matching
from src.matching.tfidf_matcher import TFIDFMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_tfidf_evaluation(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes full evaluation of the TF-IDF Cosine Similarity Baseline.

    :param data_dir: Root data directory path.
    :param output_dir: Destination directory for evaluation artifacts.
    :return: Summary metrics dictionary.
    """
    if data_dir is None:
        data_dir = root_dir / "data"

    if output_dir is None:
        output_dir = data_dir / "processed" / "evaluation"

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = data_dir / "raw" / "raw_pairs.csv"
    resumes_path = data_dir / "raw" / "resumes" / "resumes.csv"
    jobs_path = data_dir / "raw" / "jobs" / "jobs.csv"

    for p, name in [(pairs_path, "raw_pairs.csv"), (resumes_path, "resumes.csv"), (jobs_path, "jobs.csv")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required dataset: {name} at {p}")

    df_pairs = pd.read_csv(pairs_path)
    df_resumes = pd.read_csv(resumes_path)
    df_jobs = pd.read_csv(jobs_path)

    logger.info("Loaded %d pairs, %d resumes, %d jobs.", len(df_pairs), len(df_resumes), len(df_jobs))

    # 1. Execute single-pass batch matching across the benchmark
    scored_df = run_batch_matching(
        pairs_df=df_pairs,
        resumes_df=df_resumes,
        jobs_df=df_jobs,
        model_name="tfidf_cosine_baseline",
        ngram_range=(1, 2),
        min_df=1,
    )

    # Save pair-level scores CSV
    pair_scores_file = output_dir / "tfidf_pair_scores.csv"
    scored_df.to_csv(pair_scores_file, index=False)
    logger.info("Saved pair scores to: %s", pair_scores_file)

    # 2. Compute Distribution Statistics
    pos_scores = scored_df[scored_df["match_label"] == 1]["similarity_score"].to_numpy()
    neg_scores = scored_df[scored_df["match_label"] == 0]["similarity_score"].to_numpy()
    all_scores = scored_df["similarity_score"].to_numpy()

    stats_summary = {
        "dataset": {
            "total_pairs": int(len(scored_df)),
            "positive_pairs": int(len(pos_scores)),
            "negative_pairs": int(len(neg_scores)),
            "positive_ratio": float(round(len(pos_scores) / len(scored_df), 4)),
            "negative_ratio": float(round(len(neg_scores) / len(scored_df), 4)),
        },
        "score_statistics": {
            "positive_pairs": {
                "mean": float(round(np.mean(pos_scores), 4)),
                "median": float(round(np.median(pos_scores), 4)),
                "std": float(round(np.std(pos_scores), 4)),
                "min": float(round(np.min(pos_scores), 4)),
                "max": float(round(np.max(pos_scores), 4)),
            },
            "negative_pairs": {
                "mean": float(round(np.mean(neg_scores), 4)),
                "median": float(round(np.median(neg_scores), 4)),
                "std": float(round(np.std(neg_scores), 4)),
                "min": float(round(np.min(neg_scores), 4)),
                "max": float(round(np.max(neg_scores), 4)),
            },
            "all_pairs": {
                "mean": float(round(np.mean(all_scores), 4)),
                "median": float(round(np.median(all_scores), 4)),
                "std": float(round(np.std(all_scores), 4)),
                "min": float(round(np.min(all_scores), 4)),
                "max": float(round(np.max(all_scores), 4)),
            }
        }
    }

    # 3. Threshold Analysis
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    threshold_results = []

    best_f1 = -1.0
    best_thresh = 0.50

    for thresh in thresholds:
        pred = (scored_df["similarity_score"] >= thresh).astype(int)
        actual = scored_df["match_label"].astype(int)

        tp = int(np.sum((pred == 1) & (actual == 1)))
        fp = int(np.sum((pred == 1) & (actual == 0)))
        tn = int(np.sum((pred == 0) & (actual == 0)))
        fn = int(np.sum((pred == 0) & (actual == 1)))

        total = len(actual)
        acc = float(round((tp + tn) / total, 4))
        prec = float(round(tp / (tp + fp), 4)) if (tp + fp) > 0 else 0.0
        rec = float(round(tp / (tp + fn), 4)) if (tp + fn) > 0 else 0.0
        f1 = float(round(2 * prec * rec / (prec + rec), 4)) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

        threshold_results.append({
            "threshold": thresh,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        })

    df_thresh = pd.DataFrame(threshold_results)
    thresh_csv_file = output_dir / "tfidf_threshold_results.csv"
    df_thresh.to_csv(thresh_csv_file, index=False)
    logger.info("Saved threshold results to: %s", thresh_csv_file)

    # 4. Save JSON Summary
    stats_summary["threshold_analysis"] = {
        "best_experimental_threshold": best_thresh,
        "best_experimental_f1": best_f1,
        "thresholds_evaluated": threshold_results
    }
    stats_summary["model_config"] = TFIDFMatcher().get_config()

    summary_json_file = output_dir / "tfidf_baseline_summary.json"
    with open(summary_json_file, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)
    logger.info("Saved summary JSON to: %s", summary_json_file)

    # 5. Generate Visualizations
    _generate_visualizations(scored_df, df_thresh, best_thresh, output_dir)

    # 6. Print Console Report
    _print_console_report(stats_summary, df_thresh)

    return stats_summary


def _generate_visualizations(
    scored_df: pd.DataFrame,
    df_thresh: pd.DataFrame,
    best_thresh: float,
    output_dir: Path
):
    """
    Generates high-quality evaluation plots and saves them as PNG files.
    """
    pos_scores = scored_df[scored_df["match_label"] == 1]["similarity_score"]
    neg_scores = scored_df[scored_df["match_label"] == 0]["similarity_score"]

    # 1. Similarity Score Distribution Plot
    plt.figure(figsize=(9, 5))
    plt.hist(neg_scores, bins=15, alpha=0.6, label=f"Negative (Non-Match, N={len(neg_scores)})", color="#e74c3c", edgecolor="black")
    plt.hist(pos_scores, bins=15, alpha=0.6, label=f"Positive (Match, N={len(pos_scores)})", color="#2ecc71", edgecolor="black")
    plt.axvline(best_thresh, color="#2c3e50", linestyle="--", linewidth=2, label=f"Best Threshold ({best_thresh:.2f})")
    plt.title("TF-IDF Cosine Similarity Score Distribution (120 Benchmark Pairs)", fontsize=13, fontweight="bold")
    plt.xlabel("Cosine Similarity Score", fontsize=11)
    plt.ylabel("Frequency (Pairs)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    dist_file = output_dir / "similarity_distribution.png"
    plt.savefig(dist_file, dpi=200)
    plt.close()

    # 2. Threshold vs Metric Curves
    plt.figure(figsize=(9, 5))
    plt.plot(df_thresh["threshold"], df_thresh["accuracy"], marker="o", label="Accuracy", color="#3498db", linewidth=2)
    plt.plot(df_thresh["threshold"], df_thresh["precision"], marker="s", label="Precision", color="#e67e22", linewidth=2)
    plt.plot(df_thresh["threshold"], df_thresh["recall"], marker="^", label="Recall", color="#9b59b6", linewidth=2)
    plt.plot(df_thresh["threshold"], df_thresh["f1_score"], marker="d", label="F1-Score", color="#27ae60", linewidth=2.5)
    plt.axvline(best_thresh, color="#7f8c8d", linestyle=":", label=f"Peak F1 ({best_thresh:.2f})")
    plt.title("TF-IDF Baseline Performance vs. Decision Threshold", fontsize=13, fontweight="bold")
    plt.xlabel("Similarity Threshold", fontsize=11)
    plt.ylabel("Score Metric", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    metrics_file = output_dir / "threshold_metrics.png"
    plt.savefig(metrics_file, dpi=200)
    plt.close()

    # 3. Confusion Matrix Heatmap at Best Threshold
    best_row = df_thresh[df_thresh["threshold"] == best_thresh].iloc[0]
    tp = int(best_row["true_positive"])
    fp = int(best_row["false_positive"])
    fn = int(best_row["false_negative"])
    tn = int(best_row["true_negative"])

    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix @ Threshold = {best_thresh:.2f}\n(Acc: {best_row['accuracy']:.2f}, F1: {best_row['f1_score']:.2f})", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Negative (0)", "Positive (1)"], fontsize=10)
    plt.yticks(tick_marks, ["Negative (0)", "Positive (1)"], fontsize=10)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)

    thresh_val = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > thresh_val else "black"
            lbl = f"{val}"
            plt.text(j, i, lbl, horizontalalignment="center", verticalalignment="center", color=color, fontsize=14, fontweight="bold")

    plt.tight_layout()
    cm_file = output_dir / "confusion_matrix.png"
    plt.savefig(cm_file, dpi=200)
    plt.close()

    logger.info("Saved visual evaluation charts to: %s", output_dir)


def _print_console_report(summary: Dict[str, Any], df_thresh: pd.DataFrame):
    """
    Prints a clean console evaluation report.
    """
    ds = summary["dataset"]
    pos_s = summary["score_statistics"]["positive_pairs"]
    neg_s = summary["score_statistics"]["negative_pairs"]
    all_s = summary["score_statistics"]["all_pairs"]
    best_t = summary["threshold_analysis"]["best_experimental_threshold"]
    best_f = summary["threshold_analysis"]["best_experimental_f1"]

    print("\n" + "=" * 72)
    print("AI-Based Resume Screening: TF-IDF & Cosine Similarity Baseline Evaluation")
    print("=" * 72)
    print(f"Total Benchmark Pairs:   {ds['total_pairs']}")
    print(f"Positive Matches (1):    {ds['positive_pairs']} ({ds['positive_ratio']*100:.1f}%)")
    print(f"Negative Matches (0):    {ds['negative_pairs']} ({ds['negative_ratio']*100:.1f}%)")
    print("-" * 72)
    print(f"{'Class':<16} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print(f"{'Positive (1)':<16} {pos_s['mean']:<8.4f} {pos_s['median']:<8.4f} {pos_s['std']:<8.4f} {pos_s['min']:<8.4f} {pos_s['max']:<8.4f}")
    print(f"{'Negative (0)':<16} {neg_s['mean']:<8.4f} {neg_s['median']:<8.4f} {neg_s['std']:<8.4f} {neg_s['min']:<8.4f} {neg_s['max']:<8.4f}")
    print(f"{'All Pairs':<16} {all_s['mean']:<8.4f} {all_s['median']:<8.4f} {all_s['std']:<8.4f} {all_s['min']:<8.4f} {all_s['max']:<8.4f}")
    print("-" * 72)
    print("Threshold Analysis Summary:")
    print(f"{'Threshold':<11} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<10} {'TP':<5} {'FP':<5} {'TN':<5} {'FN':<5}")
    for _, row in df_thresh.iterrows():
        star = " *" if row["threshold"] == best_t else ""
        print(f"{row['threshold']:<11.2f} {row['accuracy']:<10.4f} {row['precision']:<11.4f} {row['recall']:<9.4f} {row['f1_score']:<10.4f} {int(row['true_positive']):<5} {int(row['false_positive']):<5} {int(row['true_negative']):<5} {int(row['false_negative']):<5}{star}")
    print("-" * 72)
    print(f"Best Experimental Threshold: {best_t:.2f} (F1 = {best_f:.4f})")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run_tfidf_evaluation()
