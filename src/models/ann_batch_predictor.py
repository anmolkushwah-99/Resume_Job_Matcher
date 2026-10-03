"""
ANN Batch Predictor Module.
Provides batch prediction capability over datasets of resume-job pairs using
the trained TF-IDF Keras ANN classifier.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import pandas as pd
import numpy as np

from src.models.ann_classifier import ANNResumeJobClassifier
from src.models.svm_batch_predictor import build_pair_texts

logger = logging.getLogger(__name__)


def predict_pairs_ann_batch(
    pairs_df: pd.DataFrame,
    resumes_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    classifier: Optional[ANNResumeJobClassifier] = None,
    model_dir: Optional[Path] = None,
    model_name: str = "tfidf_ann",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Executes batch ANN prediction across a set of resume-job pairs.

    :param pairs_df: DataFrame containing pair_id, resume_id, job_id, optional match_label.
    :param resumes_df: DataFrame containing candidate resumes.
    :param jobs_df: DataFrame containing benchmark jobs.
    :param classifier: Optional pre-loaded ANNResumeJobClassifier.
    :param model_dir: Optional custom model directory.
    :param model_name: Name identifier for the model.
    :param threshold: Optional decision threshold.
    :return: DataFrame containing pair predictions and sigmoid scores.
    """
    if pairs_df is None or pairs_df.empty:
        raise ValueError("pairs_df cannot be empty.")

    if classifier is None:
        from src.models.ann_service import get_ann_classifier
        clf = get_ann_classifier(model_dir=model_dir)
    else:
        clf = classifier

    pair_texts = build_pair_texts(pairs_df, resumes_df, jobs_df)
    t = threshold if threshold is not None else clf.threshold
    scores = clf.predict_score(pair_texts)
    preds = (scores >= t).astype(int)

    results = []
    for idx, row in pairs_df.iterrows():
        entry = {
            "pair_id": row.get("pair_id", f"PAIR_{idx+1}"),
            "resume_id": str(row["resume_id"]).strip(),
            "job_id": str(row["job_id"]).strip(),
            "ann_score": float(round(scores[idx], 4)),
            "predicted_label": int(preds[idx]),
            "threshold": float(round(t, 4)),
            "model_name": model_name,
        }
        if "match_label" in row and pd.notna(row["match_label"]):
            entry["true_label"] = int(row["match_label"])

        results.append(entry)

    return pd.DataFrame(results)
