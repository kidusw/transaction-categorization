"""Fuzzy merchant normalisation: noisy cleaned description -> canonical merchant."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from txcat.data.catalog import MERCHANTS, Merchant

DEFAULT_CUTOFF = 88


class MerchantNormalizer:
    """Matches cleaned descriptions against every alias of every catalog merchant.

    `token_set_ratio` scores an alias as a full match when all its tokens appear in the
    description, so "kaldis coffee bole branch" -> "Kaldis Coffee", while typos still
    score high via the underlying Levenshtein ratio.
    """

    def __init__(self, merchants: Iterable[Merchant] = MERCHANTS, cutoff: int = DEFAULT_CUTOFF):
        self.cutoff = cutoff
        self.aliases: list[str] = []
        self.owners: list[Merchant] = []
        for merchant in merchants:
            for alias in merchant.aliases:
                self.aliases.append(alias.lower())
                self.owners.append(merchant)
        # Prefer longer (more specific) aliases when scores tie.
        self._alias_len = np.array([len(a) for a in self.aliases])

    def match(self, text: str) -> Merchant | None:
        return self.match_many([text])[0]

    def match_many(self, texts: Iterable[str]) -> list[Merchant | None]:
        texts = list(texts)
        if not texts:
            return []
        scores = process.cdist(texts, self.aliases, scorer=fuzz.token_set_ratio, workers=-1)
        ranked = scores * 1000 + self._alias_len  # score first, alias length as tie-break
        best = ranked.argmax(axis=1)
        best_scores = scores[np.arange(len(texts)), best]
        return [
            self.owners[b] if s >= self.cutoff else None
            for b, s in zip(best, best_scores, strict=True)
        ]

    def normalize(self, cleaned: pd.Series) -> pd.Series:
        """Canonical merchant name per row (None when nothing matches confidently)."""
        matches = self.match_many(cleaned.fillna("").tolist())
        return pd.Series([m.name if m else None for m in matches], index=cleaned.index)
