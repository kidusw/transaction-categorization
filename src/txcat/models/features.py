"""Model inputs: a frame with the cleaned description and the signed amount.

Every model takes the same two columns, so they are interchangeable in training, MLflow
and (later) the API. The amount is only used for its direction: money in vs money out
separates e.g. "CBE transfer from X" (income-like) from "CBE transfer to X". Lexicon
phrases have no amount, so their direction is "unknown".
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

TEXT = "text"
AMOUNT = "amount"
FEATURES = [TEXT, AMOUNT]


def model_frame(df: pd.DataFrame, text_col: str = "description_clean") -> pd.DataFrame:
    """Select model inputs from a processed transactions (or lexicon) frame."""
    amount = df[AMOUNT] if AMOUNT in df else pd.Series(np.nan, index=df.index)
    return pd.DataFrame({TEXT: df[text_col].fillna("").astype(str), AMOUNT: amount.astype(float)})


def direction_features(amount: pd.DataFrame | np.ndarray) -> np.ndarray:
    """One-hot [money in, money out, unknown] from the signed amount."""
    values = np.asarray(amount, dtype=float).reshape(-1)
    known = ~np.isnan(values)
    return np.column_stack(
        [known & (values > 0), known & (values < 0), ~known],
    ).astype(float)


def direction_transformer() -> FunctionTransformer:
    return FunctionTransformer(direction_features, feature_names_out=_direction_names)


def _direction_names(_transformer, _input_features) -> np.ndarray:
    return np.array(["dir_in", "dir_out", "dir_unknown"])


def tfidf_features(
    word_ngrams: tuple[int, int] = (1, 2),
    char_ngrams: tuple[int, int] = (2, 5),
    min_df: int = 2,
) -> ColumnTransformer:
    """Word + char_wb n-gram TF-IDF on the text, plus amount direction.

    Character n-grams within word boundaries cope with typos ("supermrket"), glued tokens
    and Ge'ez script without any tokenizer or pretrained model.
    """
    return ColumnTransformer(
        [
            (
                "word",
                TfidfVectorizer(ngram_range=word_ngrams, min_df=min_df, sublinear_tf=True),
                TEXT,
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=char_ngrams, min_df=min_df, sublinear_tf=True
                ),
                TEXT,
            ),
            ("direction", direction_transformer(), [AMOUNT]),
        ]
    )
