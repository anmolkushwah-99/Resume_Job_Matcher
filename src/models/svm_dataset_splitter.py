"""
Candidate-Aware Dataset Splitting Module.
Implements leak-free grouped dataset partitioning based on candidate identity (resume_id).
Ensures no candidate appears in multiple partitions across train, validation, and test sets.
"""

import logging
from typing import Dict, Any, Tuple, Set, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


def split_candidate_dataset(
    pairs_df: pd.DataFrame,
    group_col: str = "resume_id",
    random_state: int = 42,
    test_size: float = 0.16,
    val_size: float = 0.19,
) -> Dict[str, Any]:
    """
    Performs a candidate-aware 3-way split (Train / Validation / Test) ensuring
    that all pairs belonging to the same candidate remain strictly in the same partition.

    :param pairs_df: DataFrame containing pair_id, resume_id, job_id, match_label.
    :param group_col: Column name representing the candidate grouping identity.
    :param random_state: Seed for reproducibility.
    :param test_size: Fraction of candidate groups allocated to the test set.
    :param val_size: Fraction of candidate groups allocated to the validation set from train+val.
    :return: Dictionary with train_df, val_df, test_df, candidate lists, and split statistics.
    """
    if pairs_df is None or pairs_df.empty:
        raise ValueError("pairs_df cannot be empty or None.")

    if group_col not in pairs_df.columns:
        raise KeyError(f"Grouping column '{group_col}' not found in dataset.")

    # 1. First split: (Train + Val) vs Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(pairs_df, groups=pairs_df[group_col]))

    train_val_df = pairs_df.iloc[train_val_idx].copy().reset_index(drop=True)
    test_df = pairs_df.iloc[test_idx].copy().reset_index(drop=True)

    # 2. Second split: Train vs Validation
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[group_col]))

    train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

    # 3. Verify absolute zero candidate leakage
    train_candidates: Set[str] = set(train_df[group_col].astype(str))
    val_candidates: Set[str] = set(val_df[group_col].astype(str))
    test_candidates: Set[str] = set(test_df[group_col].astype(str))

    leakage_train_val = train_candidates.intersection(val_candidates)
    leakage_train_test = train_candidates.intersection(test_candidates)
    leakage_val_test = val_candidates.intersection(test_candidates)

    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise RuntimeError(
            f"FATAL DATA LEAKAGE DETECTED in candidate splitting:\n"
            f"Train-Val Overlap: {leakage_train_val}\n"
            f"Train-Test Overlap: {leakage_train_test}\n"
            f"Val-Test Overlap: {leakage_val_test}"
        )

    # Helper function to compute partition stats
    def get_stats(sub_df: pd.DataFrame, cand_set: Set[str]) -> Dict[str, Any]:
        pos = int((sub_df["match_label"] == 1).sum()) if "match_label" in sub_df.columns else 0
        neg = int((sub_df["match_label"] == 0).sum()) if "match_label" in sub_df.columns else 0
        total = len(sub_df)
        return {
            "candidates_count": len(cand_set),
            "candidates": sorted(list(cand_set)),
            "pairs_count": total,
            "positive_count": pos,
            "negative_count": neg,
            "positive_ratio": round(pos / total, 4) if total > 0 else 0.0,
            "negative_ratio": round(neg / total, 4) if total > 0 else 0.0,
        }

    summary = {
        "strategy": "candidate_aware_grouped_split",
        "group_column": group_col,
        "random_state": random_state,
        "total_candidates": pairs_df[group_col].nunique(),
        "total_pairs": len(pairs_df),
        "train": get_stats(train_df, train_candidates),
        "validation": get_stats(val_df, val_candidates),
        "test": get_stats(test_df, test_candidates),
        "candidate_overlap": {
            "train_validation_overlap": len(leakage_train_val),
            "train_test_overlap": len(leakage_train_test),
            "validation_test_overlap": len(leakage_val_test),
        }
    }

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_val_df": train_val_df,
        "train_candidates": train_candidates,
        "val_candidates": val_candidates,
        "test_candidates": test_candidates,
        "summary": summary,
    }
