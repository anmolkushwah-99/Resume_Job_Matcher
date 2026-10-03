"""
Unit Tests for Resume Section Parser.
Tests heading variation, case-insensitivity, missing sections, and multi-word headings.
"""

from src.preprocessing.section_parser import parse_resume_sections


class TestSectionParser:
    """Test suite for resume section parsing."""

    def test_standard_headings(self):
        """Verify standard headings are cleanly parsed into logical sections."""
        resume_text = """
John Doe

SUMMARY
Experienced software engineer passionate about machine learning.

SKILLS
Python, Flask, SQL, TensorFlow, Scikit-learn, Git

EDUCATION
B.Tech in Computer Science and Engineering
ABC University

EXPERIENCE
Software Developer Intern at XYZ Corp
2 years of experience

PROJECTS
Resume Screener: Automated resume parser built with Flask

CERTIFICATIONS
AWS Certified Solutions Architect
"""
        sections = parse_resume_sections(resume_text)

        assert "Experienced software engineer" in sections["summary"]
        assert "Python, Flask, SQL" in sections["skills"]
        assert "B.Tech in Computer Science" in sections["education"]
        assert "Software Developer Intern" in sections["experience"]
        assert "Resume Screener" in sections["projects"]
        assert "AWS Certified Solutions Architect" in sections["certifications"]

    def test_case_insensitivity_and_multiword_headings(self):
        """Verify lowercase, mixed case, and multi-word headings like 'technical skills'."""
        resume_text = """
technical skills:
Java, Spring Boot, MySQL, Docker

professional experience:
3 years developing microservices

academic qualifications:
Master of Science in Data Analytics
"""
        sections = parse_resume_sections(resume_text)

        assert "Java, Spring Boot" in sections["skills"]
        assert "3 years developing microservices" in sections["experience"]
        assert "Master of Science" in sections["education"]

    def test_missing_sections(self):
        """Verify that absent sections return empty strings without errors."""
        resume_text = """
SKILLS
Python, FastAPI, Redis

EDUCATION
B.Sc in Mathematics
"""
        sections = parse_resume_sections(resume_text)

        assert "Python, FastAPI" in sections["skills"]
        assert "B.Sc in Mathematics" in sections["education"]
        assert sections["experience"] == ""
        assert sections["projects"] == ""
        assert sections["certifications"] == ""

    def test_resume_with_no_clear_headings(self):
        """Verify unformatted resume text defaults to summary/general without crashing."""
        resume_text = "Experienced Python developer with 3 years building web APIs and databases."
        sections = parse_resume_sections(resume_text)

        assert "Experienced Python developer" in sections["summary"]
        assert sections["skills"] == ""
        assert sections["education"] == ""

    def test_empty_input(self):
        """Verify empty string returns empty dictionary."""
        sections = parse_resume_sections("")
        assert sections["skills"] == ""
        assert sections["education"] == ""
        assert sections["summary"] == ""
