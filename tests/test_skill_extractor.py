"""
Unit Tests for Skill Extractor and Taxonomy Normalizer.
Tests exact matches, alias normalization, multi-word phrases, false-positive prevention, and deduplication.
"""

from src.preprocessing.skill_extractor import extract_skills, extract_skill_names


class TestSkillExtractor:
    """Test suite for skill extraction and alias normalization."""

    def test_exact_skill_matches(self):
        """Verify canonical skills are extracted directly."""
        text = "Skilled in Python, Java, MySQL, MongoDB, Docker, Git."
        skill_names = extract_skill_names(text)

        assert "Python" in skill_names
        assert "Java" in skill_names
        assert "MySQL" in skill_names
        assert "MongoDB" in skill_names
        assert "Docker" in skill_names
        assert "Git" in skill_names

    def test_alias_normalization(self):
        """Verify skill aliases are normalized to canonical master taxonomy names."""
        text = "Hands-on experience with sklearn, nodejs, react.js, golang, k8s, psql, and tf."
        skill_names = extract_skill_names(text)

        assert "Scikit-learn" in skill_names
        assert "Node.js" in skill_names
        assert "React" in skill_names
        assert "Go" in skill_names
        assert "Kubernetes" in skill_names
        assert "PostgreSQL" in skill_names
        assert "TensorFlow" in skill_names

    def test_multiword_skills(self):
        """Verify multi-word technical skills are recognized."""
        text = (
            "Specialized in Machine Learning, Natural Language Processing, "
            "Large Language Models, Spring Boot, Feature Engineering, and Data Warehousing."
        )
        skill_names = extract_skill_names(text)

        assert "Machine Learning" in skill_names
        assert "Natural Language Processing" in skill_names
        assert "Large Language Models" in skill_names
        assert "Spring Boot" in skill_names
        assert "Feature Engineering" in skill_names
        assert "Data Warehousing" in skill_names

    def test_false_positive_prevention(self):
        """Verify short single/double letter skills are not matched inside regular words."""
        # 'Developer' has 'r', 'good' has 'go', 'Computer' has 'c', 'Email' has 'ai'
        text = "Good developer working on a computer email project."
        skill_names = extract_skill_names(text)

        assert "R" not in skill_names
        assert "Go" not in skill_names
        assert "C" not in skill_names
        assert "Artificial Intelligence" not in skill_names

    def test_duplicate_prevention(self):
        """Verify that mentioning a skill and its aliases multiple times yields a single canonical skill."""
        text = "Proficient in Python, python3, py, Scikit-learn, sklearn, and scikit learn."
        skills = extract_skills(text)
        skill_names = [s["skill_name"] for s in skills]

        assert skill_names.count("Python") == 1
        assert skill_names.count("Scikit-learn") == 1
        assert len(skills) == 2

    def test_empty_input(self):
        """Verify empty text returns empty list."""
        assert extract_skills("") == []
        assert extract_skills(None) == []
        assert extract_skill_names("") == []
