"""
Unit Tests for Sequence Tokenizer and LSTM Classifier Module (Step 8).
Tests tokenizer fitting, unknown-token handling, integer sequence conversion,
padding/truncation, LSTM architecture construction, trainable embedding layer,
sigmoid score outputs, threshold modification, model persistence, and reloading.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.models.lstm_classifier import (
    SequenceTokenizer,
    LSTMResumeJobClassifier,
    build_lstm_architecture,
)


class TestSequenceTokenizer:
    """Tests the training-only SequenceTokenizer."""

    def test_special_tokens_and_initialization(self):
        tokenizer = SequenceTokenizer(max_vocab_size=500, lowercase=True)
        assert tokenizer.word2idx["<PAD>"] == 0
        assert tokenizer.word2idx["<UNK>"] == 1
        assert tokenizer.word2idx["<RESUME>"] == 2
        assert tokenizer.word2idx["</RESUME>"] == 3
        assert tokenizer.word2idx["<JOB>"] == 4
        assert tokenizer.word2idx["</JOB>"] == 5
        assert tokenizer.vocab_size == 6

    def test_fit_and_technical_tokens(self):
        tokenizer = SequenceTokenizer()
        train_texts = [
            "<RESUME> python react.js c++ c# .net ci/cd docker </RESUME> <JOB> python docker ci/cd </JOB>",
            "<RESUME> java spring-boot sql aws </RESUME> <JOB> java aws sql </JOB>",
        ]
        tokenizer.fit(train_texts)

        # Ensure vocabulary expanded
        assert tokenizer.vocab_size > 6
        assert "python" in tokenizer.word2idx
        assert "docker" in tokenizer.word2idx
        assert "c++" in tokenizer.word2idx
        assert "react.js" in tokenizer.word2idx

    def test_unknown_token_mapping(self):
        tokenizer = SequenceTokenizer()
        train_texts = ["<RESUME> python machine learning </RESUME> <JOB> python data science </JOB>"]
        tokenizer.fit(train_texts)

        # Transform text with unseen/out-of-vocabulary word
        seq = tokenizer.texts_to_sequences(["<RESUME> unseentokenxyz python </RESUME>"])
        assert len(seq) == 1
        # unseentokenxyz should map to <UNK> index (1)
        assert tokenizer.word2idx["<UNK>"] in seq[0]
        assert tokenizer.word2idx["python"] in seq[0]

    def test_pad_sequences_post(self):
        tokenizer = SequenceTokenizer()
        tokenizer.fit(["apple banana orange"])
        seqs = [[2, 6, 7], [2, 6, 7, 8, 9]]

        # Pad with max_length = 4 (post-pad and post-truncate)
        padded = tokenizer.pad_sequences(seqs, max_length=4, padding="post", truncating="post")
        assert padded.shape == (2, 4)
        # First sequence should be padded with 0 at the end
        assert padded[0, 3] == 0
        assert np.array_equal(padded[0], [2, 6, 7, 0])
        # Second sequence should be truncated to length 4
        assert np.array_equal(padded[1], [2, 6, 7, 8])

    def test_tokenizer_save_and_load(self):
        tokenizer = SequenceTokenizer(max_vocab_size=300)
        tokenizer.fit(["python fastapi docker kubernetes postgresql"])

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "tokenizer.joblib"
            tokenizer.save(file_path)
            assert file_path.exists()

            loaded = SequenceTokenizer.load(file_path)
            assert loaded.vocab_size == tokenizer.vocab_size
            assert loaded.word2idx == tokenizer.word2idx
            assert loaded.max_vocab_size == tokenizer.max_vocab_size


class TestLSTMClassifier:
    """Tests the LSTMResumeJobClassifier implementation."""

    @pytest.fixture
    def sample_pair_texts_and_labels(self):
        texts = [
            "<RESUME> python machine learning neural networks deep learning tensorflow </RESUME> <JOB> python machine learning tensorflow </JOB>",
            "<RESUME> pytorch nlp natural language processing transformers bert </RESUME> <JOB> python pytorch nlp </JOB>",
            "<RESUME> accounting financial reporting general ledger audit tax </RESUME> <JOB> python data engineer sql spark </JOB>",
            "<RESUME> culinary chef cooking food sanitation kitchen management </RESUME> <JOB> backend software engineer java aws </JOB>",
            "<RESUME> frontend developer react javascript html css redux </RESUME> <JOB> react frontend engineer typescript </JOB>",
            "<RESUME> office manager receptionist customer service scheduling billing </RESUME> <JOB> machine learning engineer python </JOB>",
            "<RESUME> backend engineer python django postgresql docker redis </RESUME> <JOB> python backend developer fastapi </JOB>",
            "<RESUME> data analyst sql tableau excel bi statistics </RESUME> <JOB> data analyst sql business intelligence </JOB>",
        ]
        labels = np.array([1, 1, 0, 0, 1, 0, 1, 1])
        return texts, labels

    def test_initialization_and_config(self):
        clf = LSTMResumeJobClassifier(
            embedding_dim=64,
            lstm_units=64,
            dense_units=32,
            dropout_rates=(0.30, 0.20),
            learning_rate=0.001,
            batch_size=8,
            epochs=20,
            threshold=0.35,
            random_state=42,
        )
        cfg = clf.get_config()
        assert cfg["model_type"] == "Embedding_LSTM"
        assert cfg["embedding_dim"] == 64
        assert cfg["lstm_units"] == 64
        assert cfg["dense_units"] == 32
        assert cfg["dropout_rates"] == [0.30, 0.20]
        assert cfg["learning_rate"] == 0.001
        assert cfg["threshold"] == 0.35
        assert cfg["is_trained"] is False

    def test_build_lstm_architecture_shapes(self):
        vocab_size = 150
        max_length = 80
        model = build_lstm_architecture(
            vocab_size=vocab_size,
            max_length=max_length,
            embedding_dim=32,
            lstm_units=48,
            dense_units=24,
            dropout_rates=(0.2, 0.1),
        )
        assert model.input_shape == (None, max_length)
        assert model.output_shape == (None, 1)

    def test_train_and_predict_score_range(self, sample_pair_texts_and_labels):
        texts, labels = sample_pair_texts_and_labels
        clf = LSTMResumeJobClassifier(
            embedding_dim=16,
            lstm_units=16,
            dense_units=8,
            dropout_rates=(0.1, 0.1),
            learning_rate=0.01,
            epochs=10,
            batch_size=4,
            threshold=0.50,
            max_length=30,
            random_state=42,
        )
        clf.train(train_texts=texts, y_train=labels)

        assert clf.is_trained is True
        assert clf.vocab_size > 0
        assert clf.max_length == 30

        scores = clf.predict_score(texts)
        assert len(scores) == len(texts)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

        preds = clf.predict(texts)
        assert len(preds) == len(texts)
        assert set(preds).issubset({0, 1})

    def test_threshold_modification(self, sample_pair_texts_and_labels):
        texts, labels = sample_pair_texts_and_labels
        clf = LSTMResumeJobClassifier(
            embedding_dim=16,
            lstm_units=16,
            dense_units=8,
            epochs=5,
            batch_size=4,
            threshold=0.50,
            max_length=30,
            random_state=42,
        )
        clf.train(texts, labels)

        clf.set_threshold(0.0)
        assert np.all(clf.predict(texts) == 1)

        clf.set_threshold(1.0)
        assert np.all(clf.predict(texts) == 0)

        with pytest.raises(ValueError):
            clf.set_threshold(1.5)

    def test_model_save_and_reload(self, sample_pair_texts_and_labels):
        texts, labels = sample_pair_texts_and_labels
        clf = LSTMResumeJobClassifier(
            embedding_dim=16,
            lstm_units=16,
            dense_units=8,
            dropout_rates=(0.1, 0.1),
            epochs=8,
            batch_size=4,
            threshold=0.40,
            max_length=25,
            random_state=42,
        )
        clf.train(texts, labels)
        scores_orig = clf.predict_score(texts)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            clf.save(save_path)

            assert (save_path / "embedding_lstm_model.keras").exists()
            assert (save_path / "embedding_lstm_tokenizer.joblib").exists()
            assert (save_path / "embedding_lstm_metadata.json").exists()

            loaded_clf = LSTMResumeJobClassifier.load(save_path)
            assert loaded_clf.is_trained is True
            assert loaded_clf.threshold == 0.40
            assert loaded_clf.vocab_size == clf.vocab_size
            assert loaded_clf.max_length == clf.max_length

            scores_loaded = loaded_clf.predict_score(texts)
            np.testing.assert_allclose(scores_orig, scores_loaded, rtol=1e-4, atol=1e-4)

    def test_untrained_model_raises_runtime_error(self):
        clf = LSTMResumeJobClassifier()
        with pytest.raises(RuntimeError):
            clf.predict_score(["<RESUME> text </RESUME> <JOB> text </JOB>"])
        with pytest.raises(RuntimeError):
            clf.save("tmp/models")

    def test_invalid_target_labels_raises_error(self):
        clf = LSTMResumeJobClassifier()
        with pytest.raises(ValueError, match="binary"):
            clf.train(["text a", "text b"], [0, 2])

    def test_empty_train_texts_raises_error(self):
        clf = LSTMResumeJobClassifier()
        with pytest.raises(ValueError):
            clf.train([], [])
