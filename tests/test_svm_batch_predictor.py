"""
Unit Tests for SVM Batch Predictor Module.
Verifies batch DataFrame prediction, pair text construction,
and structured DataFrame output.
"""

import pytest
import pandas as pd
from src.models.svm_batch_predictor import predict_pairs_batch, build_pair_texts
from src.models.svm_classifier import SVMResumeJobClassifier


class TestSVMBatchPredictor:

    @pytest.fixture
    def sample_data(self):
        resumes_df = pd.DataFrame([
            {"resume_id": "RES001", "extracted_text": "Python Flask backend developer with SQL.", "skills": "Python;Flask;SQL"},
            {"resume_id": "RES002", "extracted_text": "Machine learning engineer with PyTorch.", "skills": "Python;Machine Learning;PyTorch"},
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

    def test_build_pair_texts(self, sample_data):
        resumes_df, jobs_df, pairs_df = sample_data
        pair_texts = build_pair_texts(pairs_df, resumes_df, jobs_df)
        assert len(pair_texts) == 4
        assert "Python" in pair_texts[0]
        assert "Flask" in pair_texts[0]

    def test_predict_pairs_batch(self, sample_data):
        resumes_df, jobs_df, pairs_df = sample_data
        pair_texts = build_pair_texts(pairs_df, resumes_df, jobs_df)
        
        clf = SVMResumeJobClassifier(C=1.0, random_state=42)
        clf.train(pair_texts, pairs_df["match_label"].values)

        pred_df = predict_pairs_batch(pairs_df, resumes_df, jobs_df, classifier=clf)

        assert len(pred_df) == 4
        assert "predicted_label" in pred_df.columns
        assert "decision_score" in pred_df.columns
        assert "true_label" in pred_df.columns
        assert pred_df["model_name"].iloc[0] == "tfidf_svm"
