"""
Models Package.
Provides supervised machine learning classifiers, candidate-aware dataset splitting,
prediction services, and batch inference utilities.
"""

from src.models.svm_classifier import SVMResumeJobClassifier
from src.models.svm_dataset_splitter import split_candidate_dataset
from src.models.svm_service import (
    predict_and_store_svm_match,
    predict_svm_matches_for_resume,
    get_svm_classifier,
    set_cached_svm_classifier,
)
from src.models.svm_batch_predictor import predict_pairs_batch, build_pair_texts
from src.models.ann_classifier import ANNResumeJobClassifier
from src.models.ann_service import (
    predict_and_store_ann_match,
    predict_ann_matches_for_resume,
    get_ann_classifier,
    set_cached_ann_classifier,
)
from src.models.ann_batch_predictor import predict_pairs_ann_batch
from src.models.lstm_classifier import (
    LSTMResumeJobClassifier,
    SequenceTokenizer,
    build_pair_sequence_text,
)
from src.models.lstm_service import (
    predict_and_store_lstm_match,
    predict_lstm_matches_for_resume,
    get_lstm_classifier,
    set_cached_lstm_classifier,
)
from src.models.lstm_batch_predictor import (
    predict_pairs_lstm_batch,
    build_pair_sequence_texts,
)

__all__ = [
    "SVMResumeJobClassifier",
    "split_candidate_dataset",
    "predict_and_store_svm_match",
    "predict_svm_matches_for_resume",
    "get_svm_classifier",
    "set_cached_svm_classifier",
    "predict_pairs_batch",
    "build_pair_texts",
    "ANNResumeJobClassifier",
    "predict_and_store_ann_match",
    "predict_ann_matches_for_resume",
    "get_ann_classifier",
    "set_cached_ann_classifier",
    "predict_pairs_ann_batch",
    "LSTMResumeJobClassifier",
    "SequenceTokenizer",
    "build_pair_sequence_text",
    "predict_and_store_lstm_match",
    "predict_lstm_matches_for_resume",
    "get_lstm_classifier",
    "set_cached_lstm_classifier",
    "predict_pairs_lstm_batch",
    "build_pair_sequence_texts",
]
