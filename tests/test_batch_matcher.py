"""
Unit Tests for Batch Matcher Module.
Verifies batch matching across DataFrame inputs, shared corpus generation,
and score boundaries.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.matching.batch_matcher import run_batch_matching, build_benchmark_corpus


class TestBatchMatcher:

    @pytest.fixture
    def sample_data(self):
        resumes_df = pd.DataFrame([
            {"resume_id": "RES001", "extracted_text": "Python Flask backend developer with SQL.", "skills": "Python;Flask;SQL"},
            {"resume_id": "RES002", "extracted_text": "Machine learning engineer with PyTorch and Pandas.", "skills": "Python;Machine Learning;PyTorch"},
        ])
        jobs_df = pd.DataFrame([
            {"job_id": "JOB001", "job_title": "Python Developer", "company": "Co A", "job_description": "Flask SQL REST APIs", "required_skills": "Python;Flask;SQL"},
            {"job_id": "JOB002", "job_title": "ML Engineer", "company": "Co B", "job_description": "PyTorch deep learning", "required_skills": "Python;Machine Learning;PyTorch"},
        ])
        pairs_df = pd.DataFrame([
            {"pair_id": "P1", "resume_id": "RES001", "job_id": "JOB001", "match_label": 1},
            {"pair_id": "P2", "resume_id": "RES001", "job_id": "JOB002", "match_label": 0},
            {"pair_id": "P3", "resume_id": "RES002", "job_id": "JOB001", "match_label": 0},
            {"pair_id": "P4", "resume_id": "RES002", "job_id": "JOB002", "match_label": 1},
        ])
        return resumes_df, jobs_df, pairs_df

    def test_build_benchmark_corpus(self, sample_data):
        resumes_df, jobs_df, _ = sample_data
        corpus_info = build_benchmark_corpus(resumes_df, jobs_df)
        assert len(corpus_info["resume_docs"]) == 2
        assert len(corpus_info["job_docs"]) == 2
        assert len(corpus_info["corpus"]) == 4

    def test_run_batch_matching(self, sample_data):
        resumes_df, jobs_df, pairs_df = sample_data
        scored_df = run_batch_matching(pairs_df, resumes_df, jobs_df)

        assert len(scored_df) == 4
        assert "similarity_score" in scored_df.columns
        assert "match_label" in scored_df.columns

        # Verify positive matches score higher than cross matches
        p1_score = scored_df[scored_df["pair_id"] == "P1"]["similarity_score"].iloc[0]
        p2_score = scored_df[scored_df["pair_id"] == "P2"]["similarity_score"].iloc[0]
        assert p1_score > p2_score

        p4_score = scored_df[scored_df["pair_id"] == "P4"]["similarity_score"].iloc[0]
        p3_score = scored_df[scored_df["pair_id"] == "P3"]["similarity_score"].iloc[0]
        assert p4_score > p3_score

        # Check score bounds
        assert (scored_df["similarity_score"] >= 0.0).all()
        assert (scored_df["similarity_score"] <= 1.0).all()
