"""
Batch Matcher Module.
Provides high-performance, single-pass batch TF-IDF vectorization and cosine similarity
matching across multiple resume-job pairs using a unified shared vocabulary.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.matching.resume_text_builder import build_resume_document
from src.matching.job_text_builder import build_job_document
from src.matching.tfidf_matcher import TFIDFMatcher

logger = logging.getLogger(__name__)


def build_benchmark_corpus(
    resumes_df: pd.DataFrame,
    jobs_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Builds document dictionaries and a combined corpus list for all resumes and jobs.

    :param resumes_df: DataFrame of candidate resumes.
    :param jobs_df: DataFrame of benchmark jobs.
    :return: Dictionary containing resume_docs dict, job_docs dict, and combined corpus list.
    """
    resume_docs: Dict[str, str] = {}
    job_docs: Dict[str, str] = {}

    for _, row in resumes_df.iterrows():
        res_id = str(row["resume_id"]).strip()
        doc = build_resume_document(row.to_dict())
        resume_docs[res_id] = doc

    for _, row in jobs_df.iterrows():
        jb_id = str(row["job_id"]).strip()
        doc = build_job_document(row.to_dict())
        job_docs[jb_id] = doc

    all_corpus = list(resume_docs.values()) + list(job_docs.values())

    return {
        "resume_docs": resume_docs,
        "job_docs": job_docs,
        "corpus": all_corpus,
    }


def run_batch_matching(
    pairs_df: pd.DataFrame,
    resumes_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    model_name: str = "tfidf_cosine_baseline",
    ngram_range: tuple = (1, 2),
    min_df: int = 1,
) -> pd.DataFrame:
    """
    Executes batch matching over a set of pairs using a single fitted TFIDFMatcher.

    :param pairs_df: DataFrame containing pair_id, resume_id, job_id, and optional match_label.
    :param resumes_df: DataFrame containing candidate resumes.
    :param jobs_df: DataFrame containing benchmark jobs.
    :param model_name: Identifier for the model.
    :param ngram_range: TF-IDF n-gram range tuple.
    :param min_df: TF-IDF minimum document frequency.
    :return: DataFrame with scored pairs (pair_id, resume_id, job_id, similarity_score, etc.).
    """
    # 1. Build document dictionaries
    corpus_info = build_benchmark_corpus(resumes_df, jobs_df)
    resume_docs = corpus_info["resume_docs"]
    job_docs = corpus_info["job_docs"]
    corpus = corpus_info["corpus"]

    # 2. Fit TF-IDF once on shared corpus
    matcher = TFIDFMatcher(ngram_range=ngram_range, min_df=min_df)
    matcher.fit(corpus)

    # 3. Transform all unique resume and job documents into TF-IDF sparse matrices
    unique_res_ids = list(resume_docs.keys())
    unique_job_ids = list(job_docs.keys())

    res_matrix = matcher.transform([resume_docs[rid] for rid in unique_res_ids])
    job_matrix = matcher.transform([job_docs[jid] for jid in unique_job_ids])

    # Map ID to matrix row index
    res_id_to_idx = {rid: i for i, rid in enumerate(unique_res_ids)}
    job_id_to_idx = {jid: i for i, jid in enumerate(unique_job_ids)}

    # Precalculate full similarity matrix between all resumes and jobs
    sim_matrix = cosine_similarity(res_matrix, job_matrix)

    # 4. Iterate over target pairs and extract scores
    results = []
    for _, row in pairs_df.iterrows():
        pair_id = row.get("pair_id")
        res_id = str(row["resume_id"]).strip()
        jb_id = str(row["job_id"]).strip()
        match_label = row.get("match_label")

        if res_id not in res_id_to_idx:
            raise KeyError(f"Resume ID '{res_id}' not found in resumes dataset.")
        if jb_id not in job_id_to_idx:
            raise KeyError(f"Job ID '{jb_id}' not found in jobs dataset.")

        r_idx = res_id_to_idx[res_id]
        j_idx = job_id_to_idx[jb_id]

        score = float(sim_matrix[r_idx, j_idx])
        score = max(0.0, min(1.0, score))  # Ensure valid bounds

        entry = {
            "pair_id": pair_id if pair_id is not None else f"{res_id}_{jb_id}",
            "resume_id": res_id,
            "job_id": jb_id,
            "similarity_score": round(score, 4),
            "model_name": model_name,
        }
        if match_label is not None:
            entry["match_label"] = int(match_label)

        results.append(entry)

    return pd.DataFrame(results)
