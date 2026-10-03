"""
Unit tests for Skill Gap Explanation Module (src/matching/explanation.py).
"""

import pytest
from src.matching.explanation import calculate_skill_explanation


def test_calculate_skill_explanation_exact_match():
    """Verify skill explanation when all required skills are present."""
    resume_skills = ["Python", "Machine Learning", "SQL", "Docker"]
    job_skills = ["Python", "SQL"]

    explanation = calculate_skill_explanation(resume_skills, job_skills)
    assert set(explanation["matched_skills"]) == {"Python", "SQL"}
    assert explanation["missing_skills"] == []
    assert explanation["matched_count"] == 2
    assert explanation["missing_count"] == 0
    assert explanation["skill_overlap_ratio"] == 1.0


def test_calculate_skill_explanation_partial_match():
    """Verify skill explanation with partial overlap and missing skills."""
    resume_skills = ["Python", "Pandas", "Scikit-Learn"]
    job_skills = ["Python", "Docker", "Kubernetes", "AWS"]

    explanation = calculate_skill_explanation(resume_skills, job_skills)
    assert explanation["matched_skills"] == ["Python"]
    assert set(explanation["missing_skills"]) == {"Docker", "Kubernetes", "AWS"}
    assert explanation["matched_count"] == 1
    assert explanation["missing_count"] == 3
    assert explanation["skill_overlap_ratio"] == 0.25


def test_calculate_skill_explanation_case_insensitivity():
    """Verify skill matching is robust to case differences."""
    resume_skills = ["python", "sql", "tensorflow"]
    job_skills = ["Python", "SQL", "PyTorch"]

    explanation = calculate_skill_explanation(resume_skills, job_skills)
    assert set(explanation["matched_skills"]) == {"Python", "SQL"}
    assert explanation["missing_skills"] == ["PyTorch"]
    assert explanation["matched_count"] == 2
    assert explanation["missing_count"] == 1


def test_calculate_skill_explanation_empty_inputs():
    """Verify handling of empty resume or job skill lists."""
    res_empty = calculate_skill_explanation([], ["Python", "SQL"])
    assert res_empty["matched_skills"] == []
    assert set(res_empty["missing_skills"]) == {"Python", "SQL"}
    assert res_empty["skill_overlap_ratio"] == 0.0

    job_empty = calculate_skill_explanation(["Python", "SQL"], [])
    assert job_empty["matched_skills"] == []
    assert job_empty["missing_skills"] == []
    assert job_empty["skill_overlap_ratio"] == 0.0
