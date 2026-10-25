import pytest

from txcat.data.clean import clean_description, has_ethiopic


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CBE transfer to Abebe K. FT24123ABCD", "cbe transfer to abebe k"),
        ("telebirr airtime +251911223344", "telebirr airtime"),
        ("Ethio Telecom 0911xxxxxx 12/01/2024", "ethio telecom"),
        ("STARBUCKS #1234 SEATTLE", "starbucks seattle"),
        ("POS  Shoa   Supermarket  REF:7731", "pos shoa supermarket"),
        ("NETFLIX.COM", "netflix.com"),
        ("Chick-Fil-A", "chick-fil-a"),
        ("Wendy's", "wendy's"),
    ],
)
def test_strips_references_numbers_and_noise(raw, expected):
    assert clean_description(raw) == expected


def test_keeps_amharic_and_drops_ethiopic_punctuation():
    assert clean_description("ቴሌብር የአየር ሰዓት ግዢ። TT55555ABC") == "ቴሌብር የአየር ሰዓት ግዢ"


def test_nfkc_normalises_full_width_text():
    full_width = "".join(chr(0xFF00 + ord(c) - 0x20) for c in "KALDIS")
    assert clean_description(full_width + " Coffee") == "kaldis coffee"


def test_has_ethiopic():
    assert has_ethiopic("ገንዘብ ማስተላለፍ cbe")
    assert not has_ethiopic("cbe transfer")
