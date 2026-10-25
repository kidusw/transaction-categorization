"""Text cleaning for raw transaction descriptions (English and Amharic/Ge'ez script)."""

import re
import unicodedata

import pandas as pd

_PATTERNS = [
    # Bank/wallet references such as FT24123ABCD, TXN-99812, REF:7731
    re.compile(r"\b(?:ft|txn|ref|trx|tt)[\s:#-]*[a-z0-9]*\d[a-z0-9]*\b", re.I),
    # Dates: 12/01/2024, 2024-01-12, 12.01.24
    re.compile(r"\b\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}\b"),
    # Masked or full phone/card numbers: 0911xxxxxx, +251911223344, ****1234
    re.compile(r"(?:\+?\d[\dx*]{5,}|\*{2,}\d+)", re.I),
    # Store numbers like "#1234"
    re.compile(r"#\s*\d+"),
]
# Keep latin letters/digits, Ethiopic syllables (U+1200-U+137F) and a few joiners.
_NON_WORD = re.compile(r"[^0-9a-zሀ-ፚᎀ-᎟&'+.-]+")
_ETHIOPIC_PUNCT = re.compile(r"[፠-፨]")
_STRAY = re.compile(r"(?<!\w)[.'+-]+|[.'+-]+(?!\w)")
_SPACES = re.compile(r"\s+")


def clean_description(text: str) -> str:
    """Normalise one description: NFKC, lower-case, strip references/dates/numbers."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _ETHIOPIC_PUNCT.sub(" ", text)
    for pattern in _PATTERNS:
        text = pattern.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    text = _STRAY.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def clean_descriptions(descriptions: pd.Series) -> pd.Series:
    return descriptions.map(clean_description)


def has_ethiopic(text: str) -> bool:
    return any("ሀ" <= ch <= "፿" for ch in text)
