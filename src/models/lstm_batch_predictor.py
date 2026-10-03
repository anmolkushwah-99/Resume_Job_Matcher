"""
LSTM Batch Predictor Module.
Provides batch prediction capability over datasets of resume-job pairs using
the trained Embedding + LSTM classifier.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import pandas as pd
import numpy as np

from src.models.lstm_classifier import LSTMResumeJobClassifier, build_pair_sequence_text
from src.matching.resume_text_builder import build_resume_document
from src.matching.job_text_builder import build_job_document

logger = logging.getLogger(__name__)


def build_pair_sequence_texts(
    pairs_df: pd.DataFrame,
    resumes_df: pd.DataFrame,
    jobs_df: pd.DataFrame
) -> List[str]:
    """
    Constructs a list of sequential text documents (<RESUME>...</RESUME> <JOB>...</JOB>)
    for all rows in pairs_df.
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

        pair_texts.append(build_pair_sequence_text(res_docs[rid], job_docs[jid]))

    return pair_texts


def predict_pairs_lstm_batch(
    pairs_df: pd.DataFrame,
    resumes_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    classifier: Optional[LSTMResumeJobClassifier] = None,
    model_dir: Optional[Path] = None,
    model_name: str = "embedding_lstm",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Executes batch LSTM prediction across a set of resume-job pairs.
    """
    if pairs_df is None or pairs_df.empty:
        raise ValueError("pairs_df cannot be empty.")

    if classifier is None:
        from src.models.lstm_service import get_lstm_classifier
        clf = get_lstm_classifier(model_dir=model_dir)
    else:
        clf = classifier

    pair_texts = build_pair_sequence_texts(pairs_df, resumes_df, jobs_df)
    t = threshold if threshold is not None else clf.threshold
    scores = clf.predict_score(pair_texts)
    preds = (scores >= t).astype(int)

    results = []
    for idx, row in pairs_df.iterrows():
        entry = {
            "pair_id": row.get("pair_id", f"PAIR_{idx+1}"),
            "resume_id": str(row["resume_id"]).strip(),
            "job_id": str(row["job_id"]).strip(),
            "lstm_score": float(round(scores[idx], 4)),
            "predicted_label": int(preds[idx]),
            "threshold": float(round(t, 4)),
            "model_name": model_name,
        }
        if "match_label" in row and pd.notna(row["match_label"]):
            entry["true_label"] = int(row["match_label"])

        results.append(entry)

    return pd.DataFrame(results)
