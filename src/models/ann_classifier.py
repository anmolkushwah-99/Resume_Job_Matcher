"""
ANN / MLP Classifier Module for Resume-Job Matching.
Implements supervised Multi-Layer Perceptron (Artificial Neural Network) classification
using TensorFlow/Keras on TF-IDF feature representations, with dynamic input dimension,
ReLU activations, Dropout regularization, Sigmoid output, Binary Cross-Entropy loss,
Adam optimizer, class weighting, early stopping, and artifact persistence.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import joblib
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from src.matching.tfidf_matcher import TFIDFMatcher

logger = logging.getLogger(__name__)


def set_reproducible_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_ann_architecture(
    input_dim: int,
    hidden_units: Tuple[int, ...] = (128, 64),
    dropout_rates: Tuple[float, ...] = (0.30, 0.20),
    learning_rate: float = 0.001
) -> models.Sequential:
    """
    Constructs a compiled Keras Sequential Multi-Layer Perceptron.

    Architecture:
    Input (input_dim)
      ↓
    Dense(128, activation='relu')
      ↓
    Dropout(0.30)
      ↓
    Dense(64, activation='relu')
      ↓
    Dropout(0.20)
      ↓
    Dense(1, activation='sigmoid')

    Loss: Binary Cross-Entropy
    Optimizer: Adam
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    # Hidden Layer 1
    model.add(layers.Dense(hidden_units[0], activation="relu", name="dense_hidden_1"))
    if dropout_rates and len(dropout_rates) > 0 and dropout_rates[0] > 0.0:
        model.add(layers.Dropout(dropout_rates[0], name="dropout_1"))

    # Hidden Layer 2
    if len(hidden_units) > 1:
        model.add(layers.Dense(hidden_units[1], activation="relu", name="dense_hidden_2"))
        if len(dropout_rates) > 1 and dropout_rates[1] > 0.0:
            model.add(layers.Dropout(dropout_rates[1], name="dropout_2"))

    # Output Layer (Binary Classification)
    model.add(layers.Dense(1, activation="sigmoid", name="dense_output"))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


class ANNResumeJobClassifier:
    """
    Supervised Artificial Neural Network (MLP) for Binary Resume-Job Matching.

    Features:
    - High-dimensional TF-IDF text representation dynamically converted to dense inputs.
    - Deep nonlinear feature learning via ReLU and Sigmoid activations.
    - Dropout regularization to mitigate overfitting on small academic datasets.
    - Training-only class weighting for class imbalance handling.
    - Early stopping to restore best validation weights.
    - Full artifact persistence (.keras model + .joblib vectorizer + metadata JSON).
    """

    def __init__(
        self,
        hidden_units: Tuple[int, ...] = (128, 64),
        dropout_rates: Tuple[float, ...] = (0.30, 0.20),
        learning_rate: float = 0.001,
        batch_size: int = 16,
        epochs: int = 50,
        patience: int = 8,
        threshold: float = 0.50,
        class_weight_mode: Optional[str] = "balanced",
        random_state: int = 42,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
    ):
        self.hidden_units = tuple(hidden_units)
        self.dropout_rates = tuple(dropout_rates)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.threshold = threshold
        self.class_weight_mode = class_weight_mode
        self.random_state = random_state
        self.ngram_range = ngram_range
        self.min_df = min_df

        self.model: Optional[models.Sequential] = None
        self.vectorizer: Optional[TFIDFMatcher] = None
        self.is_trained: bool = False
        self.feature_count: int = 0
        self.training_history: Dict[str, List[float]] = {}
        self.actual_epochs: int = 0
        self.best_epoch: int = 0

    def get_config(self) -> Dict[str, Any]:
        """Returns the classifier and vectorizer hyperparameter configuration."""
        return {
            "model_type": "ANN_MLP",
            "framework": f"TensorFlow_{tf.__version__}",
            "hidden_units": list(self.hidden_units),
            "dropout_rates": list(self.dropout_rates),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.epochs,
            "actual_epochs": self.actual_epochs,
            "best_epoch": self.best_epoch,
            "patience": self.patience,
            "threshold": round(self.threshold, 4),
            "class_weight_mode": self.class_weight_mode,
            "random_state": self.random_state,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "is_trained": self.is_trained,
            "feature_count": self.feature_count,
        }

    def train(
        self,
        train_texts: List[str],
        y_train: Union[List[int], np.ndarray],
        val_texts: Optional[List[str]] = None,
        y_val: Optional[Union[List[int], np.ndarray]] = None,
        vectorizer: Optional[TFIDFMatcher] = None,
    ) -> "ANNResumeJobClassifier":
        """
        Fits the TF-IDF vectorizer (if not pre-fitted) and trains the Keras ANN model.

        :param train_texts: List of combined resume-job text strings from the training partition.
        :param y_train: Binary match labels (0 or 1) for the training partition.
        :param val_texts: Optional validation pair text documents for early stopping and evaluation.
        :param y_val: Optional validation binary match labels.
        :param vectorizer: Optional pre-configured/fitted TFIDFMatcher.
        :return: self
        """
        if not train_texts or len(train_texts) == 0:
            raise ValueError("train_texts cannot be empty.")

        y_train_arr = np.asarray(y_train, dtype=int)
        unique_labels = np.unique(y_train_arr)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(f"Target labels must be binary [0, 1]. Found: {unique_labels}")

        set_reproducible_seed(self.random_state)

        # 1. Fit TF-IDF strictly on training partition
        if vectorizer is not None and vectorizer.is_fitted:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = TFIDFMatcher(ngram_range=self.ngram_range, min_df=self.min_df)
            self.vectorizer.fit(train_texts)

        # 2. Transform train texts to dense representation
        X_train_sparse = self.vectorizer.transform(train_texts)
        X_train = X_train_sparse.toarray().astype(np.float32)
        self.feature_count = int(X_train.shape[1])

        # 3. Handle validation data if provided
        validation_data = None
        if val_texts is not None and y_val is not None:
            if len(val_texts) > 0 and len(y_val) > 0:
                X_val_sparse = self.vectorizer.transform(val_texts)
                X_val = X_val_sparse.toarray().astype(np.float32)
                y_val_arr = np.asarray(y_val, dtype=int)
                validation_data = (X_val, y_val_arr)

        # 4. Compute class weights strictly on training labels
        class_weight_dict = None
        if self.class_weight_mode == "balanced":
            cw = compute_class_weight(
                class_weight="balanced",
                classes=np.array([0, 1]),
                y=y_train_arr
            )
            class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}

        # 5. Build Keras ANN model dynamically matching input_dim
        self.model = build_ann_architecture(
            input_dim=self.feature_count,
            hidden_units=self.hidden_units,
            dropout_rates=self.dropout_rates,
            learning_rate=self.learning_rate
        )

        # 6. Configure Early Stopping and Callbacks
        callback_list = []
        if validation_data is not None:
            early_stop = callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=0
            )
            callback_list.append(early_stop)
        elif self.patience > 0:
            early_stop = callbacks.EarlyStopping(
                monitor="loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=0
            )
            callback_list.append(early_stop)

        # 7. Train Keras Model
        history = self.model.fit(
            X_train,
            y_train_arr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            class_weight=class_weight_dict,
            callbacks=callback_list,
            verbose=0,
            shuffle=True
        )

        self.training_history = {
            k: [float(round(v, 6)) for v in vals]
            for k, vals in history.history.items()
        }
        self.actual_epochs = len(self.training_history.get("loss", []))
        
        # Best epoch determination
        if "val_loss" in self.training_history:
            self.best_epoch = int(np.argmin(self.training_history["val_loss"])) + 1
        else:
            self.best_epoch = int(np.argmin(self.training_history.get("loss", [0]))) + 1

        self.is_trained = True
        logger.info(
            "Trained ANN Classifier (%s) on %d samples with %d features. Trained epochs: %d, Best epoch: %d.",
            self.hidden_units, len(train_texts), self.feature_count, self.actual_epochs, self.best_epoch
        )
        return self

    def predict_score(self, texts: List[str]) -> np.ndarray:
        """
        Generates continuous ANN Sigmoid output scores in range [0.0, 1.0].

        :param texts: List of pair text documents.
        :return: 1D numpy array of float sigmoid scores.
        """
        if not self.is_trained or self.model is None or self.vectorizer is None:
            raise RuntimeError("ANNResumeJobClassifier must be trained before predict_score().")

        if not texts or len(texts) == 0:
            return np.array([], dtype=float)

        X_sparse = self.vectorizer.transform(texts)
        X_dense = X_sparse.toarray().astype(np.float32)
        scores = self.model.predict(X_dense, verbose=0).flatten()
        return np.asarray(scores, dtype=float)

    def predict(self, texts: List[str], threshold: Optional[float] = None) -> np.ndarray:
        """
        Predicts binary match classes (0 = Non-Match, 1 = Match) using the decision threshold.

        :param texts: List of pair text documents.
        :param threshold: Optional threshold override. Defaults to self.threshold.
        :return: 1D numpy array of binary predictions [0, 1].
        """
        scores = self.predict_score(texts)
        t = self.threshold if threshold is None else threshold
        return (scores >= t).astype(int)

    def set_threshold(self, threshold: float) -> None:
        """Updates the decision classification threshold."""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be in range [0.0, 1.0]. Given: {threshold}")
        self.threshold = float(threshold)

    def save(self, model_dir: Union[str, Path]) -> Path:
        """
        Persists the trained Keras ANN model, TF-IDF vectorizer, and metadata to disk.

        :param model_dir: Directory where artifacts should be saved.
        :return: Path to the directory.
        """
        if not self.is_trained or self.model is None or self.vectorizer is None:
            raise RuntimeError("Cannot save an untrained ANN classifier.")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # 1. Save Keras ANN model (.keras native format)
        self.model.save(model_path / "tfidf_ann_model.keras")

        # 2. Save TF-IDF Vectorizer (.joblib)
        joblib.dump(self.vectorizer, model_path / "tfidf_ann_vectorizer.joblib")

        # 3. Save Metadata (.json)
        metadata = self.get_config()
        with open(model_path / "tfidf_ann_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved ANN model artifacts to: %s", model_path)
        return model_path

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "ANNResumeJobClassifier":
        """
        Loads a persisted ANN classifier from disk.

        :param model_dir: Directory containing tfidf_ann_model.keras, tfidf_ann_vectorizer.joblib, tfidf_ann_metadata.json.
        :return: Loaded ANNResumeJobClassifier instance.
        """
        model_path = Path(model_dir)
        keras_file = model_path / "tfidf_ann_model.keras"
        vec_file = model_path / "tfidf_ann_vectorizer.joblib"
        meta_file = model_path / "tfidf_ann_metadata.json"

        if not keras_file.exists() or not vec_file.exists():
            raise FileNotFoundError(f"Missing ANN model artifacts in directory: {model_path}")

        meta: Dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        instance = cls(
            hidden_units=tuple(meta.get("hidden_units", [128, 64])),
            dropout_rates=tuple(meta.get("dropout_rates", [0.30, 0.20])),
            learning_rate=float(meta.get("learning_rate", 0.001)),
            batch_size=int(meta.get("batch_size", 16)),
            epochs=int(meta.get("max_epochs", 50)),
            patience=int(meta.get("patience", 8)),
            threshold=float(meta.get("threshold", 0.50)),
            class_weight_mode=meta.get("class_weight_mode", "balanced"),
            random_state=int(meta.get("random_state", 42)),
            ngram_range=tuple(meta.get("ngram_range", [1, 2])),
            min_df=int(meta.get("min_df", 1)),
        )

        instance.model = models.load_model(keras_file)
        instance.vectorizer = joblib.load(vec_file)
        instance.is_trained = True
        instance.feature_count = int(meta.get("feature_count", 0))
        if instance.feature_count == 0 and instance.vectorizer.is_fitted:
            instance.feature_count = len(instance.vectorizer.vectorizer.get_feature_names_out())

        instance.actual_epochs = int(meta.get("actual_epochs", 0))
        instance.best_epoch = int(meta.get("best_epoch", 0))

        logger.info("Loaded ANN classifier from %s with %d features.", model_path, instance.feature_count)
        return instance
