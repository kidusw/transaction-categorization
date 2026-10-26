"""Multilingual sentence embeddings via fastembed (ONNX Runtime, no torch).

Optional: install with `uv sync --extra embeddings`. The encoder is a scikit-learn
transformer so an embedding model pickles/logs like any other pipeline; the ONNX model
itself is not pickled, only its name, and is loaded (and downloaded once) on first use.
"""

import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# 384-dim, ~220 MB ONNX download, 50+ languages including Amharic.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# fastembed defaults to the OS temp dir; keep the model next to the (git-ignored) data.
CACHE_DIR = os.environ.get("FASTEMBED_CACHE_PATH", "data/models/fastembed")

# Process-wide, keyed by model name: loaded ONNX sessions and text -> vector caches. Kept
# outside the estimator so refits/grid points reuse them and pickles stay small.
_MODELS: dict[str, object] = {}
_CACHES: dict[str, dict[str, np.ndarray]] = {}


def fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


class EmbeddingEncoder(TransformerMixin, BaseEstimator):
    """Text column -> dense embeddings, with an in-memory cache keyed by text.

    Transaction descriptions repeat heavily (same merchant, same template), so caching
    unique texts cuts encoding time by an order of magnitude.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 256):
        self.model_name = model_name
        self.batch_size = batch_size

    def _backend(self):
        if self.model_name not in _MODELS:
            from fastembed import TextEmbedding

            _MODELS[self.model_name] = TextEmbedding(
                model_name=self.model_name, cache_dir=CACHE_DIR
            )
        return _MODELS[self.model_name]

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._backend().embed(texts, batch_size=self.batch_size)))

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        texts = pd.Series(np.asarray(X).reshape(-1)).fillna("").astype(str)
        cache = _CACHES.setdefault(self.model_name, {})
        missing = [t for t in texts.unique() if t not in cache]
        if missing:
            cache.update(zip(missing, self._embed(missing), strict=True))
        return np.vstack([cache[t] for t in texts])


def embedding_features(model_name: str = DEFAULT_MODEL):
    """Sentence embedding of the text, plus amount direction (same as the TF-IDF models)."""
    from sklearn.compose import ColumnTransformer

    from txcat.models.features import AMOUNT, TEXT, direction_transformer

    return ColumnTransformer(
        [
            ("embedding", EmbeddingEncoder(model_name), TEXT),
            ("direction", direction_transformer(), [AMOUNT]),
        ]
    )
