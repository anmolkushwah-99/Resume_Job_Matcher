"""
Unit Tests for Candidate-Aware Dataset Splitting Module.
Verifies zero candidate leakage across train, validation, and test partitions,
reproducibility, and distribution properties.
"""

import pytest
import pandas as pd
from src.models.svm_dataset_splitter import split_candidate_dataset


class TestSVMDatasetSplit:

    @pytest.fixture
    def sample_pairs_df(self):
        # 10 candidates with 3 pairs each = 30 pairs
        rows = []
        for i in range(1, 11):
            rid = f"RES_{i:02d}"
            rows.append({"pair_id": f"P_{i}_1", "resume_id": rid, "job_id": "JOB_01", "match_label": 1})
            rows.append({"pair_id": f"P_{i}_2", "resume_id": rid, "job_id": "JOB_02", "match_label": 0})
            rows.append({"pair_id": f"P_{i}_3", "resume_id": rid, "job_id": "JOB_03", "match_label": 0})
        return pd.DataFrame(rows)

    def test_zero_candidate_leakage(self, sample_pairs_df):
        split = split_candidate_dataset(sample_pairs_df, random_state=42, test_size=0.2, val_size=0.25)
        
        train_cands = split["train_candidates"]
        val_cands = split["val_candidates"]
        test_cands = split["test_candidates"]

        # Assert disjoint sets
        assert len(train_cands.intersection(val_cands)) == 0
        assert len(train_cands.intersection(test_cands)) == 0
        assert len(val_cands.intersection(test_cands)) == 0

        # Total unique candidates preserved
        total_cands = train_cands.union(val_cands).union(test_cands)
        assert len(total_cands) == 10

    def test_reproducibility_with_seed(self, sample_pairs_df):
        split1 = split_candidate_dataset(sample_pairs_df, random_state=42)
        split2 = split_candidate_dataset(sample_pairs_df, random_state=42)

        assert split1["train_candidates"] == split2["train_candidates"]
        assert split1["val_candidates"] == split2["val_candidates"]
        assert split1["test_candidates"] == split2["test_candidates"]

    def test_empty_or_invalid_input(self):
        with pytest.raises(ValueError):
            split_candidate_dataset(pd.DataFrame())
        with pytest.raises(ValueError):
            split_candidate_dataset(None)

    def test_missing_group_column_raises_key_error(self):
        df_no_group = pd.DataFrame([{"col1": 1, "col2": 2}])
        with pytest.raises(KeyError):
            split_candidate_dataset(df_no_group, group_col="resume_id")
