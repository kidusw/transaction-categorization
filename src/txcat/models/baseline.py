"""Rule/keyword baseline: merchant catalog first, then keyword rules, then a fallback.

This is what a team would ship before any ML. It is scikit-learn compatible so it can be
evaluated, logged to MLflow and served exactly like the learned models.

Caveat for the synthetic slice: the synthetic generator renders descriptions from the same
merchant catalog this baseline matches against, so the baseline is unrealistically strong
there. The Kaggle (real-format) slice is the honest comparison.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from txcat.data.merchants import MerchantNormalizer
from txcat.models.features import AMOUNT, TEXT

# (category, pattern) checked in order on the cleaned description. English + Amharic.
KEYWORD_RULES: tuple[tuple[str, str], ...] = (
    ("Income", r"salary|payroll|paycheck|ደመወዝ|remittance|western union"),
    ("Cash Withdrawal", r"\batm\b|cash wdl|withdrawal|ገንዘብ ወጪ"),
    ("Loan & Card Payments", r"credit card|card payment|\bloan\b|ብድር"),
    ("Housing", r"\brent\b|mortgage|ኪራይ"),
    ("Savings", r"equb|ekub|saving|ቁጠባ"),
    ("Community & Charity", r"iddir|edir|church|mosque|donation|tithe|charity|እድር"),
    ("Insurance", r"insurance|ኢንሹራንስ"),
    ("Utilities", r"electric|water|utility|utilities|power|ኤሌክትሪክ|ውሃ"),
    ("Telecom & Airtime", r"airtime|recharge|mobile|internet|phone|telecom|wireless|data pack"),
    ("Groceries", r"grocer|supermarket|market|ሱፐርማርኬት"),
    ("Dining & Coffee", r"restaurant|cafe|coffee|pizza|burger|grill|bar\b|food|ቡና|ምግብ"),
    ("Transport & Fuel", r"fuel|\bgas\b|petrol|taxi|uber|lyft|ride|parking|ነዳጅ"),
    ("Entertainment & Subscriptions", r"movie|cinema|music|netflix|spotify|hulu|tv\b|dstv"),
    ("Health & Personal Care", r"pharm|hospital|clinic|doctor|dental|haircut|barber|salon"),
    ("Education", r"school|tuition|university|college|ትምህርት"),
    ("Travel", r"hotel|airline|airways|flight|resort|travel"),
    ("Government & Fees", r"\btax\b|revenue|permit|passport|service charge|\bfee\b"),
    ("Transfers", r"transfer|\btrf\b|p2p|sent to|received from|wallet|ማስተላለፍ"),
    ("Shopping", r"shop|store|mall|amazon|target|walmart"),
)

# Confidence a rule hit reports; used for the review-queue threshold like any model.
MERCHANT_CONFIDENCE = 0.95
KEYWORD_CONFIDENCE = 0.7
FALLBACK_CONFIDENCE = 0.2


class RuleBaseline(ClassifierMixin, BaseEstimator):
    """Merchant match -> catalog category; else first keyword rule; else fallback."""

    def __init__(self, merchant_cutoff: int = 88):
        self.merchant_cutoff = merchant_cutoff

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y))
        self.classes_ = np.array(sorted(y.unique()))
        amount = X[AMOUNT].to_numpy(dtype=float)
        # Fallbacks learned from training data: most frequent class for money in / out.
        self.fallback_in_ = _mode(y[amount > 0], y)
        self.fallback_out_ = _mode(y[~(amount > 0)], y)
        # Plain strings (not compiled patterns) so the model serialises with skops.
        self.rules_ = [(cat, p) for cat, p in KEYWORD_RULES if cat in self.classes_]
        return self

    def _predict_with_confidence(self, X: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        texts = X[TEXT].fillna("").tolist()
        amounts = X[AMOUNT].to_numpy(dtype=float)
        normalizer = MerchantNormalizer(cutoff=self.merchant_cutoff)
        merchants = normalizer.match_many(texts)
        labels, conf = [], np.empty(len(texts))
        for i, (text, merchant) in enumerate(zip(texts, merchants, strict=True)):
            if merchant is not None and merchant.category in self.classes_:
                labels.append(merchant.category)
                conf[i] = MERCHANT_CONFIDENCE
                continue
            hit = next((cat for cat, pattern in self.rules_ if re.search(pattern, text)), None)
            if hit is not None:
                labels.append(hit)
                conf[i] = KEYWORD_CONFIDENCE
            else:
                labels.append(self.fallback_in_ if amounts[i] > 0 else self.fallback_out_)
                conf[i] = FALLBACK_CONFIDENCE
        return labels, conf

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(self._predict_with_confidence(X)[0])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Rule confidence on the predicted class, the remainder spread over the others.

        With very few classes the spread remainder could exceed a low confidence, so the
        confidence is floored at 2/(n+1): the predicted class always stays the argmax.
        """
        labels, conf = self._predict_with_confidence(X)
        n_classes = len(self.classes_)
        conf = np.maximum(conf, 2 / (n_classes + 1))
        proba = np.repeat(((1 - conf) / max(n_classes - 1, 1))[:, None], n_classes, axis=1)
        index = {c: i for i, c in enumerate(self.classes_)}
        proba[np.arange(len(labels)), [index[label] for label in labels]] = conf
        return proba


def _mode(values: pd.Series, default: pd.Series) -> str:
    return (values if len(values) else default).value_counts().idxmax()
