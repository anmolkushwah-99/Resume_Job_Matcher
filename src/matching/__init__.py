"""
Matching Package.
Provides text representation builders, TF-IDF vectorization, cosine similarity
computation, match persistence, and batch evaluation.
"""

from src.matching.resume_text_builder import (
    build_resume_document,
    build_resume_document_from_db,
)
from src.matching.job_text_builder import (
    build_job_document,
    build_job_document_from_db,
)
from src.matching.tfidf_matcher import TFIDFMatcher
from src.matching.match_service import (
    calculate_and_store_match,
    calculate_matches_for_resume,
    MatchServiceError,
    ResumeNotFoundError,
    JobNotFoundError,
)
from src.matching.batch_matcher import run_batch_matching, build_benchmark_corpus
from src.matching.explanation import calculate_skill_explanation
from src.matching.model_registry import ModelRegistry
from src.matching.final_match_engine import FinalMatchEngine, get_final_match_engine

__all__ = [
    "build_resume_document",
    "build_resume_document_from_db",
    "build_job_document",
    "build_job_document_from_db",
    "TFIDFMatcher",
    "calculate_and_store_match",
    "calculate_matches_for_resume",
    "MatchServiceError",
    "ResumeNotFoundError",
    "JobNotFoundError",
    "run_batch_matching",
    "build_benchmark_corpus",
    "calculate_skill_explanation",
    "ModelRegistry",
    "FinalMatchEngine",
    "get_final_match_engine",
]

