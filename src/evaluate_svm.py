"""
Supervised SVM Evaluation and Model Training Script.
Phase 6: Performs candidate-aware dataset splitting, hyperparameter exploration,
grouped cross-validation, final test evaluation, model artifact persistence,
and fair comparison against the Step 5 baseline.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

from src.models.svm_dataset_splitter import split_candidate_dataset
from src.models.svm_classifier import SVMResumeJobClassifier
from src.models.svm_batch_predictor import build_pair_texts
from src.matching.resume_text_builder import build_resume_document
from src.matching.job_text_builder import build_job_document
from src.matching.tfidf_matcher import TFIDFMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_svm_evaluation(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes the complete end-to-end supervised SVM training, validation,
    cross-validation, and held-out test evaluation workflow.
    """
    if data_dir is None:
        data_dir = root_dir / "data"

    if output_dir is None:
        output_dir = data_dir / "processed" / "evaluation"

    if model_dir is None:
        model_dir = root_dir / "models"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    secondary_model_dir = data_dir / "processed" / "models"
    secondary_model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Datasets
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

    # 2. Candidate-Aware 3-Way Split (Train / Val / Test)
    split_info = split_candidate_dataset(df_pairs, random_state=random_state)
    train_df = split_info["train_df"]
    val_df = split_info["val_df"]
    test_df = split_info["test_df"]
    train_val_df = split_info["train_val_df"]
    split_summary = split_info["summary"]

    # Save split JSON artifact
    split_json_file = output_dir / "svm_dataset_split.json"
    with open(split_json_file, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    logger.info("Saved candidate-aware dataset split to: %s", split_json_file)

    # Build pair text representations
    train_texts = build_pair_texts(train_df, df_resumes, df_jobs)
    val_texts = build_pair_texts(val_df, df_resumes, df_jobs)
    test_texts = build_pair_texts(test_df, df_resumes, df_jobs)
    train_val_texts = build_pair_texts(train_val_df, df_resumes, df_jobs)

    y_train = train_df["match_label"].values
    y_val = val_df["match_label"].values
    y_test = test_df["match_label"].values
    y_train_val = train_val_df["match_label"].values

    # 3. Hyperparameter Exploration on Validation Partition
    # Fit TF-IDF strictly on Train partition
    vectorizer_train = TFIDFMatcher(ngram_range=(1, 2), min_df=1)
    vectorizer_train.fit(train_texts)

    X_train_hp = vectorizer_train.transform(train_texts)
    X_val_hp = vectorizer_train.transform(val_texts)

    c_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    cw_options = [None, "balanced"]
    hp_results = []

    best_val_f1 = -1.0
    best_c = 1.0
    best_cw = "balanced"

    for c in c_values:
        for cw in cw_options:
            clf_hp = LinearSVC(C=c, class_weight=cw, random_state=random_state, max_iter=2000)
            clf_hp.fit(X_train_hp, y_train)
            p_val = clf_hp.predict(X_val_hp)

            acc = float(accuracy_score(y_val, p_val))
            prec = float(precision_score(y_val, p_val, zero_division=0))
            rec = float(recall_score(y_val, p_val, zero_division=0))
            f1 = float(f1_score(y_val, p_val, zero_division=0))

            hp_results.append({
                "C": c,
                "class_weight": str(cw),
                "validation_accuracy": round(acc, 4),
                "validation_precision": round(prec, 4),
                "validation_recall": round(rec, 4),
                "validation_f1": round(f1, 4),
            })

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_c = c
                best_cw = cw

    df_hp = pd.DataFrame(hp_results)
    hp_csv_file = output_dir / "svm_hyperparameter_results.csv"
    df_hp.to_csv(hp_csv_file, index=False)
    logger.info("Saved hyperparameter exploration results to: %s", hp_csv_file)

    # 4. Grouped 5-Fold Cross-Validation on Train+Val Data
    gkf = GroupKFold(n_splits=5)
    cv_records = []

    for fold, (t_idx, v_idx) in enumerate(gkf.split(train_val_df, groups=train_val_df["resume_id"])):
        f_train = train_val_df.iloc[t_idx]
        f_val = train_val_df.iloc[v_idx]

        t_sub_texts = build_pair_texts(f_train, df_resumes, df_jobs)
        v_sub_texts = build_pair_texts(f_val, df_resumes, df_jobs)

        m_fold = TFIDFMatcher(ngram_range=(1, 2), min_df=1)
        m_fold.fit(t_sub_texts)

        Xt = m_fold.transform(t_sub_texts)
        Xv = m_fold.transform(v_sub_texts)
        yt = f_train["match_label"].values
        yv = f_val["match_label"].values

        clf_cv = LinearSVC(C=best_c, class_weight=best_cw, random_state=random_state, max_iter=2000)
        clf_cv.fit(Xt, yt)
        pv = clf_cv.predict(Xv)

        acc = float(accuracy_score(yv, pv))
        prec = float(precision_score(yv, pv, zero_division=0))
        rec = float(recall_score(yv, pv, zero_division=0))
        f1 = float(f1_score(yv, pv, zero_division=0))

        cv_records.append({
            "fold": fold + 1,
            "train_candidates": f_train["resume_id"].nunique(),
            "val_candidates": f_val["resume_id"].nunique(),
            "pairs_count": len(f_val),
            "positive_pairs": int((yv == 1).sum()),
            "negative_pairs": int((yv == 0).sum()),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        })

    df_cv = pd.DataFrame(cv_records)
    cv_csv_file = output_dir / "svm_cross_validation_results.csv"
    df_cv.to_csv(cv_csv_file, index=False)
    logger.info("Saved 5-fold grouped CV results to: %s", cv_csv_file)

    # 5. Final Model Retraining on Train + Val
    final_classifier = SVMResumeJobClassifier(
        C=best_c,
        class_weight=best_cw,
        random_state=random_state,
        max_iter=2000,
        ngram_range=(1, 2),
        min_df=1,
    )
    final_classifier.train(train_val_texts, y_train_val)

    # Save trained model artifacts
    final_classifier.save(model_dir)
    # Also save to secondary location for completeness
    final_classifier.save(secondary_model_dir)

    # 6. Held-Out Test Set Evaluation
    test_preds = final_classifier.predict(test_texts)
    test_dec_scores = final_classifier.decision_function(test_texts)

    test_acc = float(accuracy_score(y_test, test_preds))
    test_prec = float(precision_score(y_test, test_preds, zero_division=0))
    test_rec = float(recall_score(y_test, test_preds, zero_division=0))
    test_f1 = float(f1_score(y_test, test_preds, zero_division=0))

    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()

    # Save test predictions CSV
    df_test_preds = pd.DataFrame({
        "pair_id": test_df.get("pair_id", [f"PAIR_{i+1}" for i in range(len(test_df))]),
        "resume_id": test_df["resume_id"],
        "job_id": test_df["job_id"],
        "true_label": y_test,
        "predicted_label": test_preds,
        "decision_score": np.round(test_dec_scores, 4),
        "model_name": "tfidf_svm",
    })
    test_preds_file = output_dir / "svm_test_predictions.csv"
    df_test_preds.to_csv(test_preds_file, index=False)
    logger.info("Saved test set predictions to: %s", test_preds_file)

    # Save classification report text
    cls_report = classification_report(y_test, test_preds, target_names=["Non-Match (0)", "Match (1)"])
    report_file = output_dir / "svm_classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("SVM Resume-Job Classification Report (Held-Out Test Set)\n")
        f.write("=" * 60 + "\n\n")
        f.write(cls_report)
    logger.info("Saved classification report to: %s", report_file)

    # 7. Extract Feature Importance
    top_features = final_classifier.get_top_features(top_k=15)
    feat_rows = []
    for f_item in top_features["positive_features"]:
        feat_rows.append({"term": f_item["term"], "weight": f_item["weight"], "class_affinity": "Match (1)"})
    for f_item in top_features["negative_features"]:
        feat_rows.append({"term": f_item["term"], "weight": f_item["weight"], "class_affinity": "Non-Match (0)"})

    df_feats = pd.DataFrame(feat_rows)
    feats_file = output_dir / "svm_feature_importance.csv"
    df_feats.to_csv(feats_file, index=False)
    logger.info("Saved feature importance to: %s", feats_file)

    # 8. Fair Step 5 Baseline Comparison on Exact Same Test Partition
    # Build single-document corpus on Train+Val to fit Step 5 vectorizer
    res_docs_map = {r["resume_id"]: build_resume_document(r.to_dict()) for _, r in df_resumes.iterrows()}
    job_docs_map = {j["job_id"]: build_job_document(j.to_dict()) for _, j in df_jobs.iterrows()}

    s5_train_val_corpus = (
        [res_docs_map[r["resume_id"]] for _, r in train_val_df.iterrows()]
        + [job_docs_map[r["job_id"]] for _, r in train_val_df.iterrows()]
    )
    s5_matcher = TFIDFMatcher(ngram_range=(1, 2), min_df=1)
    s5_matcher.fit(s5_train_val_corpus)

    # Select threshold on Train+Val pairs
    s5_sims_tv = [
        s5_matcher.compute_similarity(res_docs_map[r["resume_id"]], job_docs_map[r["job_id"]])
        for _, r in train_val_df.iterrows()
    ]
    best_s5_thresh, best_s5_f1_tv = 0.05, 0.0
    for t_cand in np.arange(0.05, 0.55, 0.05):
        p_cand = (np.array(s5_sims_tv) >= t_cand).astype(int)
        f_cand = f1_score(y_train_val, p_cand, zero_division=0)
        if f_cand > best_s5_f1_tv:
            best_s5_f1_tv, best_s5_thresh = f_cand, float(round(t_cand, 2))

    # Evaluate Step 5 on Test pairs
    s5_sims_test = [
        s5_matcher.compute_similarity(res_docs_map[r["resume_id"]], job_docs_map[r["job_id"]])
        for _, r in test_df.iterrows()
    ]
    s5_test_preds = (np.array(s5_sims_test) >= best_s5_thresh).astype(int)

    s5_test_acc = float(accuracy_score(y_test, s5_test_preds))
    s5_test_prec = float(precision_score(y_test, s5_test_preds, zero_division=0))
    s5_test_rec = float(recall_score(y_test, s5_test_preds, zero_division=0))
    s5_test_f1 = float(f1_score(y_test, s5_test_preds, zero_division=0))
    cm_s5 = confusion_matrix(y_test, s5_test_preds)
    tn5, fp5, fn5, tp5 = cm_s5.ravel()

    comparison_records = [
        {
            "model": "tfidf_cosine_baseline",
            "evaluation_protocol": f"Candidate-Aware Held-Out Test (Threshold={best_s5_thresh:.2f})",
            "accuracy": round(s5_test_acc, 4),
            "precision": round(s5_test_prec, 4),
            "recall": round(s5_test_rec, 4),
            "f1_score": round(s5_test_f1, 4),
            "true_positive": int(tp5),
            "false_positive": int(fp5),
            "true_negative": int(tn5),
            "false_negative": int(fn5),
        },
        {
            "model": "tfidf_svm",
            "evaluation_protocol": "Candidate-Aware Held-Out Test (LinearSVC)",
            "accuracy": round(test_acc, 4),
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1_score": round(test_f1, 4),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        }
    ]
    df_comp = pd.DataFrame(comparison_records)
    comp_file = output_dir / "svm_baseline_comparison.csv"
    df_comp.to_csv(comp_file, index=False)
    logger.info("Saved baseline comparison to: %s", comp_file)

    # 9. Save JSON Summary
    cv_mean_f1 = float(round(df_cv["f1_score"].mean(), 4))
    cv_std_f1 = float(round(df_cv["f1_score"].std(), 4))
    cv_mean_acc = float(round(df_cv["accuracy"].mean(), 4))
    cv_std_acc = float(round(df_cv["accuracy"].std(), 4))

    metrics_summary = {
        "dataset_split": split_summary,
        "hyperparameter_selection": {
            "selected_C": best_c,
            "selected_class_weight": str(best_cw),
            "validation_f1": round(best_val_f1, 4),
        },
        "grouped_cross_validation_5fold": {
            "mean_accuracy": cv_mean_acc,
            "std_accuracy": cv_std_acc,
            "mean_f1": cv_mean_f1,
            "std_f1": cv_std_f1,
        },
        "final_test_evaluation": {
            "accuracy": round(test_acc, 4),
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1_score": round(test_f1, 4),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
        "step5_test_comparison": {
            "baseline_threshold": best_s5_thresh,
            "accuracy": round(s5_test_acc, 4),
            "precision": round(s5_test_prec, 4),
            "recall": round(s5_test_rec, 4),
            "f1_score": round(s5_test_f1, 4),
        },
        "training_vocabulary_size": len(final_classifier.feature_names),
    }

    metrics_json_file = output_dir / "svm_metrics.json"
    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("Saved complete SVM metrics JSON to: %s", metrics_json_file)

    # 10. Generate Visualizations
    _generate_svm_plots(split_summary, cm, test_acc, test_f1, output_dir)

    # 11. Print Formatted Console Report
    _print_svm_console_report(metrics_summary, df_hp, df_cv, df_comp)

    return metrics_summary


def _generate_svm_plots(
    split_summary: Dict[str, Any],
    cm: np.ndarray,
    test_acc: float,
    test_f1: float,
    output_dir: Path
):
    """
    Generates SVM confusion matrix and partition distribution charts.
    """
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"SVM Test Confusion Matrix\n(Acc: {test_acc:.4f}, F1: {test_f1:.4f})", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Non-Match (0)", "Match (1)"], fontsize=10)
    plt.yticks(tick_marks, ["Non-Match (0)", "Match (1)"], fontsize=10)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)

    thresh_val = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > thresh_val else "black"
            plt.text(j, i, f"{val}", horizontalalignment="center", verticalalignment="center",
                     color=color, fontsize=14, fontweight="bold")

    plt.tight_layout()
    cm_file = output_dir / "svm_confusion_matrix.png"
    plt.savefig(cm_file, dpi=200)
    plt.close()

    # 2. Split Class Distribution Bar Chart
    labels = ["Train", "Validation", "Test"]
    pos_counts = [split_summary["train"]["positive_count"], split_summary["validation"]["positive_count"], split_summary["test"]["positive_count"]]
    neg_counts = [split_summary["train"]["negative_count"], split_summary["validation"]["negative_count"], split_summary["test"]["negative_count"]]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, pos_counts, width, label="Match (1)", color="#2ecc71")
    plt.bar(x + width/2, neg_counts, width, label="Non-Match (0)", color="#e74c3c")
    plt.xlabel("Partition", fontsize=11)
    plt.ylabel("Number of Pairs", fontsize=11)
    plt.title("Candidate-Aware Dataset Partition Distribution", fontsize=13, fontweight="bold")
    plt.xticks(x, labels, fontsize=10)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    dist_file = output_dir / "svm_split_distribution.png"
    plt.savefig(dist_file, dpi=200)
    plt.close()


def _print_svm_console_report(
    summary: Dict[str, Any],
    df_hp: pd.DataFrame,
    df_cv: pd.DataFrame,
    df_comp: pd.DataFrame
):
    """Prints a clean, academic console report for Step 6."""
    spl = summary["dataset_split"]
    hp = summary["hyperparameter_selection"]
    tst = summary["final_test_evaluation"]

    print("\n" + "=" * 76)
    print("AI-Based Resume Screening: Supervised SVM Classification (Step 6)")
    print("=" * 76)
    print("1. Candidate-Aware Partitioning (Zero Overlap):")
    print(f"   Train:      {spl['train']['candidates_count']} Candidates | {spl['train']['pairs_count']} Pairs | Pos: {spl['train']['positive_count']} | Neg: {spl['train']['negative_count']}")
    print(f"   Validation: {spl['validation']['candidates_count']} Candidates | {spl['validation']['pairs_count']} Pairs | Pos: {spl['validation']['positive_count']} | Neg: {spl['validation']['negative_count']}")
    print(f"   Test:       {spl['test']['candidates_count']} Candidates | {spl['test']['pairs_count']} Pairs | Pos: {spl['test']['positive_count']} | Neg: {spl['test']['negative_count']}")
    print(f"   Candidate Overlap: 0 across all partitions.")
    print("-" * 76)
    print("2. Hyperparameter Selection (Validation Data):")
    print(f"   Selected C: {hp['selected_C']}, class_weight: {hp['selected_class_weight']} (Val F1: {hp['validation_f1']:.4f})")
    print("-" * 76)
    print("3. Grouped 5-Fold Cross-Validation (Train+Val):")
    cv_res = summary["grouped_cross_validation_5fold"]
    print(f"   Mean Accuracy: {cv_res['mean_accuracy']:.4f} +/- {cv_res['std_accuracy']:.4f}")
    print(f"   Mean F1-Score: {cv_res['mean_f1']:.4f} +/- {cv_res['std_f1']:.4f}")
    print("-" * 76)
    print("4. Held-Out Test Evaluation (Final Model):")
    print(f"   Accuracy:  {tst['accuracy']:.4f}  |  Precision: {tst['precision']:.4f}")
    print(f"   Recall:    {tst['recall']:.4f}  |  F1-Score:  {tst['f1_score']:.4f}")
    print(f"   Confusion Matrix: TP={tst['true_positive']}, FP={tst['false_positive']}, TN={tst['true_negative']}, FN={tst['false_negative']}")
    print("-" * 76)
    print("5. Step 5 vs. Step 6 Fair Test Set Comparison:")
    print(f"{'Model':<24} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<10}")
    for _, row in df_comp.iterrows():
        print(f"{row['model']:<24} {row['accuracy']:<10.4f} {row['precision']:<11.4f} {row['recall']:<9.4f} {row['f1_score']:<10.4f}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_svm_evaluation()
