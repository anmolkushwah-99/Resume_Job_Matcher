"""
Unit Tests for Job Text Builder Module.
Verifies job document construction from raw text, dictionaries, and database records.
"""

import pytest
from src.matching.job_text_builder import build_job_document, build_job_document_from_db


class TestJobTextBuilder:

    def test_build_from_string(self):
        raw = "Seeking a Python Developer with Flask and SQL skills."
        doc = build_job_document(raw)
        assert "Python" in doc
        assert "Flask" in doc

    def test_build_from_dict_full(self):
        data = {
            "job_title": "Python Backend Developer",
            "company": "TechNova Solutions",
            "job_category": "Software Engineering",
            "job_description": "Design and maintain scalable REST APIs using Flask.",
            "required_skills": "Python;Flask;PostgreSQL;REST API",
            "preferred_skills": "Docker;Redis;AWS",
            "minimum_experience": 2.0,
            "education_requirement": "Bachelor in Computer Science",
        }
        doc = build_job_document(data)
        assert "Python Backend Developer at TechNova Solutions" in doc
        assert "Category: Software Engineering" in doc
        assert "Required Skills: Python, Flask, PostgreSQL, REST API" in doc
        assert "Preferred Skills: Docker, Redis, AWS" in doc
        assert "Experience Requirement: 2.0 years" in doc
        assert "Education Requirement: Bachelor in Computer Science" in doc

    def test_build_from_dict_minimal(self):
        data = {
            "job_title": "Data Analyst",
            "job_description": "Analyze company metrics.",
        }
        doc = build_job_document(data)
        assert "Data Analyst" in doc
        assert "Analyze company metrics." in doc

    def test_empty_and_none_input(self):
        assert build_job_document("") == ""
        assert build_job_document({}) == ""
        assert build_job_document(None) == ""

    def test_build_from_db_nonexistent(self):
        res = build_job_document_from_db("nonexistent-job-id-9999")
        assert res is None
