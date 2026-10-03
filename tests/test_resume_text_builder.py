"""
Unit Tests for Resume Text Builder Module.
Verifies resume document construction from raw text, dictionaries, and structured features.
"""

import pytest
from src.matching.resume_text_builder import build_resume_document, build_resume_document_from_db


class TestResumeTextBuilder:

    def test_build_from_string(self):
        raw = "Software developer with experience in Python and Flask."
        doc = build_resume_document(raw)
        assert "Python" in doc
        assert "Flask" in doc

    def test_build_from_dict_full(self):
        data = {
            "extracted_text": "Experienced full stack engineer.",
            "skills": ["Python", "Flask", "PostgreSQL", "Docker"],
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "experience_years": 3.5,
            "certifications": "AWS Certified Developer",
            "projects": "E-Commerce Platform",
        }
        doc = build_resume_document(data)
        assert "Experienced full stack engineer" in doc
        assert "Skills: Python, Flask, PostgreSQL, Docker" in doc
        assert "Education: B.Tech Computer Science" in doc
        assert "Experience: 3.5 years" in doc
        assert "Certifications: AWS Certified Developer" in doc
        assert "Projects: E-Commerce Platform" in doc

    def test_build_from_dict_minimal(self):
        data = {
            "extracted_text": "Junior developer.",
        }
        doc = build_resume_document(data)
        assert doc == "Junior developer."

    def test_technical_terms_preserved(self):
        data = {
            "extracted_text": "Proficient in C++, C#, .NET, Node.js, and CI/CD pipelines.",
            "skills": ["C++", "C#", ".NET", "Node.js", "CI/CD"],
        }
        doc = build_resume_document(data)
        assert "C++" in doc
        assert "C#" in doc
        assert ".NET" in doc
        assert "Node.js" in doc
        assert "CI/CD" in doc

    def test_empty_and_none_input(self):
        assert build_resume_document("") == ""
        assert build_resume_document({}) == ""
        assert build_resume_document(None) == ""

    def test_build_from_db_nonexistent(self):
        res = build_resume_document_from_db("nonexistent-resume-id-9999")
        assert res is None
