"""
Supervised ANN Evaluation and Model Training Script.
Step 7: Performs candidate-aware dataset verification, hyperparameter exploration on validation,
threshold optimization on validation, final retraining on Train+Validation, held-out test
evaluation, model artifact persistence (.keras, .joblib, metadata JSON), and fair comparison
against Step 5 (TF-IDF Cosine Baseline) and Step 6 (Linear SVM).
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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
    roc_auc_score,
    average_precision_score,
)
from sklearn.svm import LinearSVC

from src.models.svm_dataset_splitter import split_candidate_dataset
from src.models.ann_classifier import ANNResumeJobClassifier, set_reproducible_seed
from src.models.svm_batch_predictor import build_pair_texts
from src.matching.resume_text_builder import build_resume_document
from src.matching.job_text_builder import build_job_document
from src.matching.tfidf_matcher import TFIDFMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_ann_evaluation(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes the complete end-to-end supervised ANN training, validation,
    threshold tuning, and held-out test evaluation workflow.
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

    set_reproducible_seed(random_state)

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

    # 2. Candidate-Aware 3-Way Split (Reuse Step 6 partition exactly)
    split_json_existing = output_dir / "svm_dataset_split.json"
    split_info = split_candidate_dataset(df_pairs, random_state=random_state)
    train_df = split_info["train_df"]
    val_df = split_info["val_df"]
    test_df = split_info["test_df"]
    train_val_df = split_info["train_val_df"]
    split_summary = split_info["summary"]

    # Verify zero candidate overlap
    train_cands = set(train_df["resume_id"].unique())
    val_cands = set(val_df["resume_id"].unique())
    test_cands = set(test_df["resume_id"].unique())

    assert len(train_cands.intersection(val_cands)) == 0, "Train and Validation candidates overlap!"
    assert len(train_cands.intersection(test_cands)) == 0, "Train and Test candidates overlap!"
    assert len(val_cands.intersection(test_cands)) == 0, "Validation and Test candidates overlap!"

    # Save ann_dataset_split.json
    ann_split_file = output_dir / "ann_dataset_split.json"
    with open(ann_split_file, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    logger.info("Verified zero candidate leakage. Saved split to: %s", ann_split_file)

    # Build pair text representations
    train_texts = build_pair_texts(train_df, df_resumes, df_jobs)
    val_texts = build_pair_texts(val_df, df_resumes, df_jobs)
    test_texts = build_pair_texts(test_df, df_resumes, df_jobs)
    train_val_texts = build_pair_texts(train_val_df, df_resumes, df_jobs)

    y_train = train_df["match_label"].values
    y_val = val_df["match_label"].values
    y_test = test_df["match_label"].values
    y_train_val = train_val_df["match_label"].values

    # 3. Phase A: Hyperparameter Exploration on Validation Partition
    # Fit TF-IDF strictly on Train partition
    vectorizer_train = TFIDFMatcher(ngram_range=(1, 2), min_df=1)
    vectorizer_train.fit(train_texts)

    configs_to_test = [
        {"name": "Config_A (128->64, drop 0.3/0.2, lr 0.001)", "hidden_units": (128, 64), "dropout": (0.30, 0.20), "lr": 0.001, "batch_size": 16},
        {"name": "Config_B (64->32, drop 0.2/0.2, lr 0.001)", "hidden_units": (64, 32), "dropout": (0.20, 0.20), "lr": 0.001, "batch_size": 16},
        {"name": "Config_C (128->64, drop 0.3/0.2, lr 0.0005)", "hidden_units": (128, 64), "dropout": (0.30, 0.20), "lr": 0.0005, "batch_size": 16},
        {"name": "Config_D (128->64, drop 0.0/0.0, lr 0.001)", "hidden_units": (128, 64), "dropout": (0.0, 0.0), "lr": 0.001, "batch_size": 16},
    ]

    hp_results = []
    best_hp_f1 = -1.0
    best_config = configs_to_test[0]
    best_trained_clf = None

    for cfg in configs_to_test:
        set_reproducible_seed(random_state)
        clf_candidate = ANNResumeJobClassifier(
            hidden_units=cfg["hidden_units"],
            dropout_rates=cfg["dropout"],
            learning_rate=cfg["lr"],
            batch_size=cfg["batch_size"],
            epochs=50,
            patience=8,
            threshold=0.50,
            class_weight_mode="balanced",
            random_state=random_state,
        )
        clf_candidate.train(
            train_texts=train_texts,
            y_train=y_train,
            val_texts=val_texts,
            y_val=y_val,
            vectorizer=vectorizer_train
        )

        p_val = clf_candidate.predict(val_texts, threshold=0.50)
        acc = float(accuracy_score(y_val, p_val))
        prec = float(precision_score(y_val, p_val, zero_division=0))
        rec = float(recall_score(y_val, p_val, zero_division=0))
        f1 = float(f1_score(y_val, p_val, zero_division=0))
        final_val_loss = float(clf_candidate.training_history.get("val_loss", [0.0])[-1])

        hp_results.append({
            "configuration": cfg["name"],
            "hidden_units": str(cfg["hidden_units"]),
            "dropout_rates": str(cfg["dropout"]),
            "learning_rate": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "actual_epochs": clf_candidate.actual_epochs,
            "best_epoch": clf_candidate.best_epoch,
            "validation_loss": round(final_val_loss, 4),
            "validation_accuracy": round(acc, 4),
            "validation_precision": round(prec, 4),
            "validation_recall": round(rec, 4),
            "validation_f1": round(f1, 4),
        })

        if f1 > best_hp_f1:
            best_hp_f1 = f1
            best_config = cfg
            best_trained_clf = clf_candidate

    df_hp = pd.DataFrame(hp_results)
    hp_csv_file = output_dir / "ann_hyperparameter_results.csv"
    df_hp.to_csv(hp_csv_file, index=False)
    logger.info("Saved hyperparameter exploration results to: %s", hp_csv_file)

    # 4. Phase A: Validation Threshold Selection (Sweeping threshold on validation set)
    val_scores = best_trained_clf.predict_score(val_texts)
    threshold_candidates = np.arange(0.30, 0.75, 0.05)
    thresh_records = []
    best_threshold = 0.50
    best_thresh_f1 = -1.0

    for t_val in threshold_candidates:
        t_float = float(round(t_val, 2))
        p_t = (val_scores >= t_float).astype(int)
        acc_t = float(accuracy_score(y_val, p_t))
        prec_t = float(precision_score(y_val, p_t, zero_division=0))
        rec_t = float(recall_score(y_val, p_t, zero_division=0))
        f1_t = float(f1_score(y_val, p_t, zero_division=0))

        thresh_records.append({
            "threshold": t_float,
            "validation_accuracy": round(acc_t, 4),
            "validation_precision": round(prec_t, 4),
            "validation_recall": round(rec_t, 4),
            "validation_f1": round(f1_t, 4),
        })

        if f1_t > best_thresh_f1 or (f1_t == best_thresh_f1 and abs(t_float - 0.50) < abs(best_threshold - 0.50)):
            best_thresh_f1 = f1_t
            best_threshold = t_float

    df_thresh = pd.DataFrame(thresh_records)
    thresh_csv_file = output_dir / "ann_threshold_results.csv"
    df_thresh.to_csv(thresh_csv_file, index=False)
    logger.info("Saved threshold exploration results to: %s (Selected threshold: %.2f)", thresh_csv_file, best_threshold)

    # 5. Phase B: Final Model Retraining on Train + Validation
    logger.info("Retraining final ANN on Train+Val (102 pairs) with %s and threshold %.2f...", best_config["name"], best_threshold)
    set_reproducible_seed(random_state)

    final_classifier = ANNResumeJobClassifier(
        hidden_units=best_config["hidden_units"],
        dropout_rates=best_config["dropout"],
        learning_rate=best_config["lr"],
        batch_size=best_config["batch_size"],
        epochs=50,
        patience=8,
        threshold=best_threshold,
        class_weight_mode="balanced",
        random_state=random_state,
        ngram_range=(1, 2),
        min_df=1,
    )
    final_classifier.train(
        train_texts=train_val_texts,
        y_train=y_train_val,
        val_texts=val_texts,
        y_val=y_val,
    )

    # Save model artifacts
    final_classifier.save(model_dir)
    final_classifier.save(secondary_model_dir)

    # Save training history CSV
    df_hist = pd.DataFrame(final_classifier.training_history)
    df_hist.insert(0, "epoch", range(1, len(df_hist) + 1))
    hist_csv_file = output_dir / "ann_training_history.csv"
    df_hist.to_csv(hist_csv_file, index=False)
    logger.info("Saved training history to: %s", hist_csv_file)

    # 6. Final Evaluation on Untouched Test Set
    test_scores = final_classifier.predict_score(test_texts)
    test_preds = (test_scores >= best_threshold).astype(int)

    test_acc = float(accuracy_score(y_test, test_preds))
    test_prec = float(precision_score(y_test, test_preds, zero_division=0))
    test_rec = float(recall_score(y_test, test_preds, zero_division=0))
    test_f1 = float(f1_score(y_test, test_preds, zero_division=0))

    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        test_roc_auc = float(roc_auc_score(y_test, test_scores))
    except Exception:
        test_roc_auc = float("nan")

    try:
        test_pr_auc = float(average_precision_score(y_test, test_scores))
    except Exception:
        test_pr_auc = float("nan")

    # Save test predictions CSV
    df_test_preds = pd.DataFrame({
        "pair_id": test_df.get("pair_id", [f"PAIR_{i+1}" for i in range(len(test_df))]),
        "resume_id": test_df["resume_id"],
        "job_id": test_df["job_id"],
        "true_label": y_test,
        "ann_score": np.round(test_scores, 4),
        "predicted_label": test_preds,
        "threshold": best_threshold,
        "model_name": "tfidf_ann",
    })
    test_preds_file = output_dir / "ann_test_predictions.csv"
    df_test_preds.to_csv(test_preds_file, index=False)
    logger.info("Saved test predictions to: %s", test_preds_file)

    # Save classification report text
    cls_report = classification_report(y_test, test_preds, target_names=["Non-Match (0)", "Match (1)"])
    report_file = output_dir / "ann_classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("ANN (MLP) Resume-Job Classification Report (Held-Out Test Set)\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Architecture: {best_config['name']}\n")
        f.write(f"Decision Threshold: {best_threshold:.2f}\n\n")
        f.write(cls_report)
    logger.info("Saved classification report to: %s", report_file)

    # 7. Model Comparison across Step 5, Step 6, and Step 7 on Same Held-Out Test Set
    res_docs_map = {r["resume_id"]: build_resume_document(r.to_dict()) for _, r in df_resumes.iterrows()}
    job_docs_map = {j["job_id"]: build_job_document(j.to_dict()) for _, j in df_jobs.iterrows()}

    # Step 5 Baseline on Test
    s5_train_val_corpus = (
        [res_docs_map[r["resume_id"]] for _, r in train_val_df.iterrows()]
        + [job_docs_map[r["job_id"]] for _, r in train_val_df.iterrows()]
    )
    s5_matcher = TFIDFMatcher(ngram_range=(1, 2), min_df=1)
    s5_matcher.fit(s5_train_val_corpus)

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

    # Step 6 SVM on Test
    svm_vectorizer = TFIDFMatcher(ngram_range=(1, 2), min_df=1)
    svm_vectorizer.fit(train_val_texts)
    X_tv_svm = svm_vectorizer.transform(train_val_texts)
    X_test_svm = svm_vectorizer.transform(test_texts)

    svm_clf = LinearSVC(C=1.0, class_weight="balanced", random_state=random_state, max_iter=2000)
    svm_clf.fit(X_tv_svm, y_train_val)
    svm_test_preds = svm_clf.predict(X_test_svm)

    svm_test_acc = float(accuracy_score(y_test, svm_test_preds))
    svm_test_prec = float(precision_score(y_test, svm_test_preds, zero_division=0))
    svm_test_rec = float(recall_score(y_test, svm_test_preds, zero_division=0))
    svm_test_f1 = float(f1_score(y_test, svm_test_preds, zero_division=0))
    cm_svm = confusion_matrix(y_test, svm_test_preds)
    tn6, fp6, fn6, tp6 = cm_svm.ravel()

    comparison_records = [
        {
            "model": "tfidf_cosine_baseline",
            "evaluation_protocol": f"Candidate-Aware Held-Out Test (Cosine Theta={best_s5_thresh:.2f})",
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
            "evaluation_protocol": "Candidate-Aware Held-Out Test (LinearSVC C=1.0 Balanced)",
            "accuracy": round(svm_test_acc, 4),
            "precision": round(svm_test_prec, 4),
            "recall": round(svm_test_rec, 4),
            "f1_score": round(svm_test_f1, 4),
            "true_positive": int(tp6),
            "false_positive": int(fp6),
            "true_negative": int(tn6),
            "false_negative": int(fn6),
        },
        {
            "model": "tfidf_ann",
            "evaluation_protocol": f"Candidate-Aware Held-Out Test (MLP {best_config['hidden_units']} Threshold={best_threshold:.2f})",
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
    comp_file = output_dir / "ann_model_comparison.csv"
    df_comp.to_csv(comp_file, index=False)
    logger.info("Saved 3-model comparison to: %s", comp_file)

    # 8. Complete Metrics JSON
    metrics_summary = {
        "dataset": split_summary,
        "training": {
            "selected_architecture": best_config["name"],
            "hidden_units": list(best_config["hidden_units"]),
            "dropout_rates": list(best_config["dropout"]),
            "learning_rate": best_config["lr"],
            "batch_size": best_config["batch_size"],
            "max_epochs": 50,
            "actual_epochs": final_classifier.actual_epochs,
            "best_epoch": final_classifier.best_epoch,
            "class_weight_mode": "balanced",
            "training_vocabulary_size": final_classifier.feature_count,
        },
        "validation": {
            "hyperparameter_results": hp_results,
            "best_validation_f1": round(best_hp_f1, 4),
        },
        "threshold_selection": {
            "selected_threshold": best_threshold,
            "validation_f1_at_threshold": round(best_thresh_f1, 4),
        },
        "test": {
            "accuracy": round(test_acc, 4),
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1_score": round(test_f1, 4),
            "specificity": round(specificity, 4),
            "roc_auc": round(test_roc_auc, 4) if not np.isnan(test_roc_auc) else None,
            "pr_auc": round(test_pr_auc, 4) if not np.isnan(test_pr_auc) else None,
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
            "total_test_pairs": len(test_df),
        },
        "comparison": {
            "tfidf_cosine_baseline": {"accuracy": round(s5_test_acc, 4), "precision": round(s5_test_prec, 4), "recall": round(s5_test_rec, 4), "f1_score": round(s5_test_f1, 4)},
            "tfidf_svm": {"accuracy": round(svm_test_acc, 4), "precision": round(svm_test_prec, 4), "recall": round(svm_test_rec, 4), "f1_score": round(svm_test_f1, 4)},
            "tfidf_ann": {"accuracy": round(test_acc, 4), "precision": round(test_prec, 4), "recall": round(test_rec, 4), "f1_score": round(test_f1, 4)},
        },
        "limitations": [
            "Academic benchmark with 25 candidates and 120 labeled pairs total.",
            "Held-out test set contains 4 candidates and 18 pairs; individual prediction flips impact metrics.",
            "TF-IDF input is lexical and does not capture long-range contextual or sequential semantic dependencies.",
            "ANN output represents model sigmoid activation, not an autonomous hiring probability.",
            "Model is designed for academic decision support benchmarking, not real-world automated recruitment.",
        ]
    }

    metrics_json_file = output_dir / "ann_metrics.json"
    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("Saved complete ANN metrics JSON to: %s", metrics_json_file)

    # 9. Visualizations
    _generate_ann_plots(final_classifier.training_history, cm, test_acc, test_f1, best_threshold, output_dir)

    # 10. Print Console Summary
    _print_ann_console_report(metrics_summary, df_hp, df_thresh, df_comp)

    return metrics_summary


def _generate_ann_plots(
    history: Dict[str, List[float]],
    cm: np.ndarray,
    test_acc: float,
    test_f1: float,
    threshold: float,
    output_dir: Path
):
    """Generates ANN training curves and confusion matrix heatmap."""
    # 1. Training Curves
    epochs_range = range(1, len(history.get("loss", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(epochs_range, history.get("loss", []), label="Training Loss", color="#e74c3c", linewidth=2)
    if "val_loss" in history and len(history["val_loss"]) == len(epochs_range):
        ax1.plot(epochs_range, history["val_loss"], label="Validation Loss", color="#3498db", linewidth=2, linestyle="--")
    ax1.set_title("ANN Binary Cross-Entropy Loss vs. Epochs", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy curve
    ax2.plot(epochs_range, history.get("accuracy", []), label="Training Accuracy", color="#2ecc71", linewidth=2)
    if "val_accuracy" in history and len(history["val_accuracy"]) == len(epochs_range):
        ax2.plot(epochs_range, history["val_accuracy"], label="Validation Accuracy", color="#9b59b6", linewidth=2, linestyle="--")
    ax2.set_title("ANN Classification Accuracy vs. Epochs", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    curves_file = output_dir / "ann_training_curves.png"
    plt.savefig(curves_file, dpi=200)
    plt.close()

    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Purples")
    plt.title(f"ANN Test Confusion Matrix (Theta={threshold:.2f})\n(Acc: {test_acc:.4f}, F1: {test_f1:.4f})", fontsize=12, fontweight="bold")
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
    cm_file = output_dir / "ann_confusion_matrix.png"
    plt.savefig(cm_file, dpi=200)
    plt.close()


def _print_ann_console_report(
    summary: Dict[str, Any],
    df_hp: pd.DataFrame,
    df_thresh: pd.DataFrame,
    df_comp: pd.DataFrame
):
    """Prints a clean, academic summary for Step 7."""
    spl = summary["dataset"]
    trn = summary["training"]
    tst = summary["test"]
    thr = summary["threshold_selection"]

    print("\n" + "=" * 78)
    print("AI-Based Resume Screening: Supervised ANN / MLP Classification (Step 7)")
    print("=" * 78)
    print("1. Candidate-Aware Partitioning (Reused Step 6 Partition):")
    print(f"   Train:      {spl['train']['candidates_count']} Candidates | {spl['train']['pairs_count']} Pairs (Pos: {spl['train']['positive_count']}, Neg: {spl['train']['negative_count']})")
    print(f"   Validation: {spl['validation']['candidates_count']} Candidates | {spl['validation']['pairs_count']} Pairs (Pos: {spl['validation']['positive_count']}, Neg: {spl['validation']['negative_count']})")
    print(f"   Test:       {spl['test']['candidates_count']} Candidates | {spl['test']['pairs_count']} Pairs (Pos: {spl['test']['positive_count']}, Neg: {spl['test']['negative_count']})")
    print("-" * 78)
    print(f"2. Architecture & Training ({trn['selected_architecture']}):")
    print(f"   Input Dim: {trn['training_vocabulary_size']} features | Hidden: {trn['hidden_units']} | Dropout: {trn['dropout_rates']}")
    print(f"   Loss: Binary Cross-Entropy | Optimizer: Adam (lr={trn['learning_rate']}) | Batch: {trn['batch_size']}")
    print(f"   Trained Epochs: {trn['actual_epochs']} | Best Epoch: {trn['best_epoch']} | Frozen Threshold: {thr['selected_threshold']:.2f}")
    print("-" * 78)
    print("3. Held-Out Test Evaluation (18 pairs / 4 candidates):")
    print(f"   Accuracy:  {tst['accuracy']:.4f}  |  Precision: {tst['precision']:.4f}")
    print(f"   Recall:    {tst['recall']:.4f}  |  F1-Score:  {tst['f1_score']:.4f}  |  Specificity: {tst['specificity']:.4f}")
    print(f"   Confusion Matrix: TP={tst['true_positive']}, FP={tst['false_positive']}, TN={tst['true_negative']}, FN={tst['false_negative']}")
    print("-" * 78)
    print("4. Step 5 vs. Step 6 vs. Step 7 Fair Test Set Comparison:")
    print(f"{'Model':<24} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<10}")
    for _, row in df_comp.iterrows():
        print(f"{row['model']:<24} {row['accuracy']:<10.4f} {row['precision']:<11.4f} {row['recall']:<9.4f} {row['f1_score']:<10.4f}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    run_ann_evaluation()
