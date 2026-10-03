"""
LSTM / RNN-Based Sequence Classifier Module for Resume-Job Matching.
Implements sequential tokenization, integer sequence encoding, sequence padding/truncation,
trainable Embedding layer, LSTM recurrent layers, ReLU/Sigmoid activations, Binary Cross-Entropy
loss, Adam optimizer, class weighting, early stopping, and artifact persistence.
"""

import os
import re
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

logger = logging.getLogger(__name__)


def set_reproducible_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_pair_sequence_text(resume_text: str, job_text: str) -> str:
    """
    Constructs a deterministic sequential text representation combining resume and job.
    Uses structured delimiter tokens to preserve structural boundaries.
    """
    res_clean = str(resume_text).strip() if resume_text else ""
    jb_clean = str(job_text).strip() if job_text else ""
    return f"<RESUME> {res_clean} </RESUME> <JOB> {jb_clean} </JOB>"


class SequenceTokenizer:
    """
    Custom deterministic sequence tokenizer for integer token conversion.
    
    Features:
    - Special token handling: index 0 reserved for PAD, index 1 for <UNK>,
      index 2 for <RESUME>, index 3 for </RESUME>, index 4 for <JOB>, index 5 for </JOB>.
    - Preserves important technical alphanumeric tokens (e.g., C++, .NET, CI/CD, React.js).
    - Vocabulary construction strictly on training corpora.
    - Fully serializable with joblib.
    """

    def __init__(self, max_vocab_size: int = 5000, lowercase: bool = True):
        self.max_vocab_size = max_vocab_size
        self.lowercase = lowercase
        self.special_tokens = ["<PAD>", "<UNK>", "<RESUME>", "</RESUME>", "<JOB>", "</JOB>"]
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}
        for idx, sp_tok in enumerate(self.special_tokens):
            self.word_index[sp_tok] = idx
            self.word_index[sp_tok.lower()] = idx
            self.index_word[idx] = sp_tok
        self.word_counts: Dict[str, int] = {}
        self.is_fitted: bool = False

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text preserving words, technical symbols, and XML delimiter tags."""
        if not text:
            return []
        
        raw = text.lower() if self.lowercase else text
        
        # Token extraction pattern:
        # 1. XML-style boundary tags (e.g., <resume>, </resume>, <job>, </job>)
        # 2. Terms with trailing +/# symbols (e.g., c++, c#)
        # 3. Dotted/hyphenated/slashed technical terms (e.g., react.js, ci/cd, spring-boot)
        # 4. Leading dot terms (e.g., .net)
        # 5. Standard words / alphanumeric tokens
        token_pattern = r"</?[A-Za-z0-9_-]+>|[A-Za-z0-9]+[+#]+|[A-Za-z0-9]+(?:[\.+#/-][A-Za-z0-9]+)+|\.[A-Za-z0-9]+|\w+"
        tokens = re.findall(token_pattern, raw)
        return tokens

    def fit(self, texts: List[str]) -> "SequenceTokenizer":
        """
        Builds vocabulary strictly from the provided training texts.
        """
        self.word_counts = {}
        for text in texts:
            for token in self._tokenize(text):
                self.word_counts[token] = self.word_counts.get(token, 0) + 1

        # Initialize with special tokens
        self.word_index = {}
        self.index_word = {}
        for idx, sp_tok in enumerate(self.special_tokens):
            self.word_index[sp_tok] = idx
            self.word_index[sp_tok.lower()] = idx
            self.index_word[idx] = sp_tok

        # Sort tokens by frequency
        sorted_tokens = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        current_idx = len(self.special_tokens)

        for token, _ in sorted_tokens:
            if token in self.word_index:
                continue
            if current_idx >= self.max_vocab_size:
                break
            self.word_index[token] = current_idx
            self.index_word[current_idx] = token
            current_idx += 1

        self.is_fitted = True
        logger.info("Fitted SequenceTokenizer with vocabulary size: %d", len(self.word_index))
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.index_word)

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Converts text documents into lists of integer token IDs."""
        if not self.is_fitted:
            raise RuntimeError("SequenceTokenizer must be fitted before texts_to_sequences().")

        sequences = []
        unk_idx = self.word_index["<UNK>"]

        for text in texts:
            tokens = self._tokenize(text)
            seq = [self.word_index.get(token, unk_idx) for token in tokens]
            sequences.append(seq)

        return sequences

    def pad_sequences(
        self,
        sequences: List[List[int]],
        max_length: int,
        padding: str = "post",
        truncating: str = "post"
    ) -> np.ndarray:
        """
        Pads and truncates integer sequences to a uniform length.
        """
        padded = np.zeros((len(sequences), max_length), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            
            # Truncation
            if len(seq) > max_length:
                if truncating == "post":
                    trunc_seq = seq[:max_length]
                else:
                    trunc_seq = seq[-max_length:]
            else:
                trunc_seq = seq

            # Padding
            if padding == "post":
                padded[i, :len(trunc_seq)] = trunc_seq
            else:
                padded[i, -len(trunc_seq):] = trunc_seq

        return padded

    @property
    def word2idx(self) -> Dict[str, int]:
        """Convenience alias for word_index."""
        return self.word_index

    def save(self, file_path: Union[str, Path]) -> None:
        """Serializes the fitted SequenceTokenizer using joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "SequenceTokenizer":
        """Loads a persisted SequenceTokenizer instance."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SequenceTokenizer file not found at: {path}")
        return joblib.load(path)



def build_lstm_architecture(
    vocab_size: int,
    max_length: int,
    embedding_dim: int = 64,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout_rates: Tuple[float, float] = (0.30, 0.20),
    learning_rate: float = 0.001
) -> models.Sequential:
    """
    Constructs a compiled Keras Sequential LSTM neural network for sequence classification.

    Architecture:
    Input (max_length,)
      ↓
    Embedding (vocab_size, embedding_dim)
      ↓
    LSTM (lstm_units)
      ↓
    Dropout (dropout_rates[0])
      ↓
    Dense (dense_units, activation='relu')
      ↓
    Dropout (dropout_rates[1])
      ↓
    Dense (1, activation='sigmoid')

    Loss: Binary Cross-Entropy
    Optimizer: Adam
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(max_length,), dtype="int32", name="sequence_input"))
    model.add(layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="embedding_layer"
    ))
    model.add(layers.LSTM(lstm_units, name="lstm_layer"))

    if dropout_rates[0] > 0.0:
        model.add(layers.Dropout(dropout_rates[0], name="dropout_1"))

    model.add(layers.Dense(dense_units, activation="relu", name="dense_hidden"))

    if dropout_rates[1] > 0.0:
        model.add(layers.Dropout(dropout_rates[1], name="dropout_2"))

    model.add(layers.Dense(1, activation="sigmoid", name="dense_output"))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


class LSTMResumeJobClassifier:
    """
    Supervised LSTM-Based Binary Sequence Classifier for Resume-Job Matching.

    Features:
    - Sequential integer encoding with custom SequenceTokenizer.
    - Dynamic sequence length computation based on training distribution.
    - Trainable Embedding layer mapping tokens to dense semantic coordinates.
    - Long Short-Term Memory (LSTM) recurrent layer capturing sequential patterns and long-range dependencies.
    - Dropout regularization to prevent overfitting on small academic datasets.
    - Training-only class weighting for class imbalance compensation.
    - Early stopping to restore best validation weights.
    - Full artifact persistence (.keras model + .joblib tokenizer + metadata JSON).
    """

    def __init__(
        self,
        max_vocab_size: int = 5000,
        max_length: int = 512,
        embedding_dim: int = 64,
        lstm_units: int = 64,
        dense_units: int = 32,
        dropout_rates: Tuple[float, float] = (0.30, 0.20),
        learning_rate: float = 0.001,
        batch_size: int = 16,
        epochs: int = 50,
        patience: int = 8,
        threshold: float = 0.50,
        class_weight_mode: Optional[str] = "balanced",
        random_state: int = 42,
    ):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rates = tuple(dropout_rates)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.threshold = threshold
        self.class_weight_mode = class_weight_mode
        self.random_state = random_state

        self.model: Optional[models.Sequential] = None
        self.tokenizer: Optional[SequenceTokenizer] = None
        self.is_trained: bool = False
        self.vocab_size: int = 0
        self.training_history: Dict[str, List[float]] = {}
        self.actual_epochs: int = 0
        self.best_epoch: int = 0

    def get_config(self) -> Dict[str, Any]:
        """Returns the classifier hyperparameter and sequence configuration."""
        return {
            "model_type": "Embedding_LSTM",
            "framework": f"TensorFlow_{tf.__version__}",
            "max_vocab_size": self.max_vocab_size,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units,
            "dense_units": self.dense_units,
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
            "is_trained": self.is_trained,
        }

    def train(
        self,
        train_texts: List[str],
        y_train: Union[List[int], np.ndarray],
        val_texts: Optional[List[str]] = None,
        y_val: Optional[Union[List[int], np.ndarray]] = None,
        tokenizer: Optional[SequenceTokenizer] = None,
    ) -> "LSTMResumeJobClassifier":
        """
        Fits the SequenceTokenizer (if not provided) strictly on training texts,
        converts sequences to padded arrays, and trains the LSTM model.

        :param train_texts: List of paired text strings from the training partition.
        :param y_train: Binary match labels for training pairs.
        :param val_texts: Optional validation pair text documents.
        :param y_val: Optional validation binary match labels.
        :param tokenizer: Optional pre-fitted SequenceTokenizer.
        :return: self
        """
        if not train_texts or len(train_texts) == 0:
            raise ValueError("train_texts cannot be empty.")

        y_train_arr = np.asarray(y_train, dtype=int)
        unique_labels = np.unique(y_train_arr)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(f"Target labels must be binary [0, 1]. Found: {unique_labels}")

        set_reproducible_seed(self.random_state)

        # 1. Fit tokenizer strictly on training partition
        if tokenizer is not None and tokenizer.is_fitted:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SequenceTokenizer(max_vocab_size=self.max_vocab_size)
            self.tokenizer.fit(train_texts)

        self.vocab_size = self.tokenizer.vocab_size

        # 2. Convert and pad training sequences
        train_seqs = self.tokenizer.texts_to_sequences(train_texts)
        X_train = self.tokenizer.pad_sequences(train_seqs, max_length=self.max_length)

        # 3. Handle validation sequences if provided
        validation_data = None
        if val_texts is not None and y_val is not None and len(val_texts) > 0 and len(y_val) > 0:
            val_seqs = self.tokenizer.texts_to_sequences(val_texts)
            X_val = self.tokenizer.pad_sequences(val_seqs, max_length=self.max_length)
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

        # 5. Build Keras LSTM model
        self.model = build_lstm_architecture(
            vocab_size=self.vocab_size,
            max_length=self.max_length,
            embedding_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            dense_units=self.dense_units,
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

        # 7. Train Keras LSTM
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
        
        if "val_loss" in self.training_history:
            self.best_epoch = int(np.argmin(self.training_history["val_loss"])) + 1
        else:
            self.best_epoch = int(np.argmin(self.training_history.get("loss", [0]))) + 1

        self.is_trained = True
        logger.info(
            "Trained LSTM Classifier (emb=%d, lstm=%d) on %d samples with vocab=%d, max_len=%d. Trained epochs: %d, Best epoch: %d.",
            self.embedding_dim, self.lstm_units, len(train_texts), self.vocab_size, self.max_length, self.actual_epochs, self.best_epoch
        )
        return self

    def predict_score(self, texts: List[str]) -> np.ndarray:
        """
        Generates continuous LSTM Sigmoid output scores in range [0.0, 1.0].
        """
        if not self.is_trained or self.model is None or self.tokenizer is None:
            raise RuntimeError("LSTMResumeJobClassifier must be trained before predict_score().")

        if not texts or len(texts) == 0:
            return np.array([], dtype=float)

        seqs = self.tokenizer.texts_to_sequences(texts)
        X_padded = self.tokenizer.pad_sequences(seqs, max_length=self.max_length)
        scores = self.model.predict(X_padded, verbose=0).flatten()
        return np.asarray(scores, dtype=float)

    def predict(self, texts: List[str], threshold: Optional[float] = None) -> np.ndarray:
        """
        Predicts binary match classes (0 = Non-Match, 1 = Match) using the decision threshold.
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
        Persists the trained Keras LSTM model, SequenceTokenizer, and metadata to disk.
        """
        if not self.is_trained or self.model is None or self.tokenizer is None:
            raise RuntimeError("Cannot save an untrained LSTM classifier.")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # 1. Save Keras LSTM model (.keras native format)
        self.model.save(model_path / "embedding_lstm_model.keras")

        # 2. Save Sequence Tokenizer (.joblib)
        joblib.dump(self.tokenizer, model_path / "embedding_lstm_tokenizer.joblib")

        # 3. Save Metadata (.json)
        metadata = self.get_config()
        with open(model_path / "embedding_lstm_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved LSTM model artifacts to: %s", model_path)
        return model_path

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "LSTMResumeJobClassifier":
        """
        Loads a persisted LSTM classifier from disk.
        """
        model_path = Path(model_dir)
        keras_file = model_path / "embedding_lstm_model.keras"
        tok_file = model_path / "embedding_lstm_tokenizer.joblib"
        meta_file = model_path / "embedding_lstm_metadata.json"

        if not keras_file.exists() or not tok_file.exists():
            raise FileNotFoundError(f"Missing LSTM model artifacts in directory: {model_path}")

        meta: Dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        instance = cls(
            max_vocab_size=int(meta.get("max_vocab_size", 5000)),
            max_length=int(meta.get("max_length", 512)),
            embedding_dim=int(meta.get("embedding_dim", 64)),
            lstm_units=int(meta.get("lstm_units", 64)),
            dense_units=int(meta.get("dense_units", 32)),
            dropout_rates=tuple(meta.get("dropout_rates", [0.30, 0.20])),
            learning_rate=float(meta.get("learning_rate", 0.001)),
            batch_size=int(meta.get("batch_size", 16)),
            epochs=int(meta.get("max_epochs", 50)),
            patience=int(meta.get("patience", 8)),
            threshold=float(meta.get("threshold", 0.50)),
            class_weight_mode=meta.get("class_weight_mode", "balanced"),
            random_state=int(meta.get("random_state", 42)),
        )

        instance.model = models.load_model(keras_file)
        instance.tokenizer = joblib.load(tok_file)
        instance.is_trained = True
        instance.vocab_size = int(meta.get("vocab_size", instance.tokenizer.vocab_size))
        instance.actual_epochs = int(meta.get("actual_epochs", 0))
        instance.best_epoch = int(meta.get("best_epoch", 0))

        logger.info("Loaded LSTM classifier from %s with vocab_size=%d, max_length=%d.", model_path, instance.vocab_size, instance.max_length)
        return instance
