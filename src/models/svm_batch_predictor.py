"""
SVM Batch Predictor Module.
Provides batch prediction capability over datasets of resume-job pairs using
the trained TF-IDF SVM classifier.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import pandas as pd
import numpy as np

from src.models.svm_classifier import SVMResumeJobClassifier
from src.models.svm_service import get_svm_classifier
from src.matching.resume_text_builder import build_resume_document
from src.matching.job_text_builder import build_job_document

logger = logging.getLogger(__name__)


def build_pair_texts(
    pairs_df: pd.DataFrame,
    resumes_df: pd.DataFrame,
    jobs_df: pd.DataFrame
) -> List[str]:
    """
    Constructs a list of combined pair text documents for all rows in pairs_df.
    """
    res_docs: Dict[str, str] = {
        str(r["resume_id"]).strip(): build_resume_document(r.to_dict())
        for _, r in resumes_df.iterrows()
    }
    job_docs: Dict[str, str] = {
        str(j["job_id"]).strip(): build_job_document(j.to_dict())
        for _, j in jobs_df.iterrows()
    }

    pair_texts = []
    for _, row in pairs_df.iterrows():
        rid = str(row["resume_id"]).strip()
        jid = str(row["job_id"]).strip()

        if rid not in res_docs:
            raise KeyError(f"Resume ID '{rid}' not found in resumes dataset.")
        if jid not in job_docs:
            raise KeyError(f"Job ID '{jid}' not found in jobs dataset.")

        pair_texts.append(res_docs[rid] + "\n\n" + job_docs[jid])

    return pair_texts


def predict_pairs_batch(
    pairs_df: pd.DataFrame,
    resumes_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    classifier: Optional[SVMResumeJobClassifier] = None,
    model_dir: Optional[Path] = None,
    model_name: str = "tfidf_svm"
) -> pd.DataFrame:
    """
    Executes batch SVM prediction across a set of resume-job pairs.

    :param pairs_df: DataFrame containing pair_id, resume_id, job_id, optional match_label.
    :param resumes_df: DataFrame containing candidate resumes.
    :param jobs_df: DataFrame containing benchmark jobs.
    :param classifier: Optional pre-loaded SVMResumeJobClassifier.
    :param model_dir: Optional custom model directory.
    :param model_name: Name identifier for the model.
    :return: DataFrame containing pair predictions and decision scores.
    """
    if pairs_df is None or pairs_df.empty:
        raise ValueError("pairs_df cannot be empty.")

    clf = classifier or get_svm_classifier(model_dir=model_dir)

    pair_texts = build_pair_texts(pairs_df, resumes_df, jobs_df)
    preds = clf.predict(pair_texts)
    dec_scores = clf.decision_function(pair_texts)

    results = []
    for idx, row in pairs_df.iterrows():
        entry = {
            "pair_id": row.get("pair_id", f"PAIR_{idx+1}"),
            "resume_id": str(row["resume_id"]).strip(),
            "job_id": str(row["job_id"]).strip(),
            "predicted_label": int(preds[idx]),
            "decision_score": float(round(dec_scores[idx], 4)),
            "model_name": model_name,
        }
        if "match_label" in row and pd.notna(row["match_label"]):
            entry["true_label"] = int(row["match_label"])

        results.append(entry)

    return pd.DataFrame(results)
