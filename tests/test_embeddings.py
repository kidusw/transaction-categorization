import pickle

import numpy as np
import pandas as pd
import pytest

from txcat.models import embeddings
from txcat.models.embeddings import EmbeddingEncoder, embedding_features


class FakeBackend:
    def __init__(self):
        self.seen: list[str] = []

    def embed(self, texts, batch_size=256):
        self.seen.extend(texts)
        for t in texts:
            yield np.array([len(t), t.count("a")], dtype=float)


@pytest.fixture
def fake_model(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setitem(embeddings._MODELS, "fake", backend)
    monkeypatch.setitem(embeddings._CACHES, "fake", {})
    return backend


def test_encoder_caches_unique_texts(fake_model):
    encoder = EmbeddingEncoder("fake")
    out = encoder.fit_transform(pd.Series(["aa", "b", "aa"]))
    assert out.tolist() == [[2, 2], [1, 0], [2, 2]]
    encoder.transform(pd.Series(["b", "ccc"]))
    assert fake_model.seen == ["aa", "b", "ccc"]


def test_encoder_pickles_without_model(fake_model):
    encoder = EmbeddingEncoder("fake")
    encoder.transform(pd.Series(["x"]))
    clone = pickle.loads(pickle.dumps(encoder))
    assert vars(clone) == {"model_name": "fake", "batch_size": 256}


def test_embedding_features_adds_direction(fake_model):
    X = pd.DataFrame({"text": ["aa", "b"], "amount": [5.0, np.nan]})
    out = embedding_features("fake").fit_transform(X)
    assert out.tolist() == [[2, 2, 1, 0, 0], [1, 0, 0, 0, 1]]


@pytest.mark.skipif(not embeddings.fastembed_available(), reason="fastembed extra not installed")
def test_real_model_is_multilingual():
    encoder = EmbeddingEncoder()
    vectors = encoder.transform(pd.Series(["salary payment", "ደመወዝ", "salary payment"]))
    assert vectors.shape[0] == 3 and vectors.shape[1] > 100
    assert np.allclose(vectors[0], vectors[2])
