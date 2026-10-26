import numpy as np
import pandas as pd
import pytest

from txcat.insights.anomaly import AnomalyConfig, isolation_forest_scores, score_anomalies
from txcat.insights.evaluate import account_halves, evaluation_table
from txcat.insights.recurring import detect_recurring, mark_recurring, series_key
from txcat.insights.spend import monthly_by_category, spend_table


def _txns(rows: list[tuple]) -> pd.DataFrame:
    """rows of (account_id, date, amount, category, merchant)."""
    df = pd.DataFrame(rows, columns=["account_id", "date", "amount", "category", "merchant"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _monthly(merchant, category, amount, day=5, months=8, account="A", jitter=(0,)):
    return [
        (account, f"2024-{m:02d}-{day + jitter[m % len(jitter)]:02d}", amount, category, merchant)
        for m in range(1, months + 1)
    ]


# -- recurring --------------------------------------------------------------------------


def test_detects_monthly_bill_with_jitter_and_next_date():
    df = _txns(_monthly("DStv", "Entertainment & Subscriptions", -700.0, jitter=(0, 2, -1)))
    found = detect_recurring(df)

    assert len(found) == 1
    row = found.iloc[0]
    assert (row.counterparty, row.direction, row.period) == ("DStv", "out", "monthly")
    assert row.typical_amount == 700
    assert row.next_expected == row.last_seen + pd.Timedelta(days=round(row.median_gap_days))


def test_salary_without_merchant_falls_back_to_category():
    df = _txns(_monthly(None, "Income", 20000.0, day=26))
    found = detect_recurring(df)
    assert found[["counterparty", "direction"]].values.tolist() == [["Income", "in"]]


def test_weekly_series():
    dates = pd.date_range("2024-01-01", periods=10, freq="7D")
    df = _txns([("A", d, -300.0, "Savings", "Equb") for d in dates])
    assert detect_recurring(df)["period"].tolist() == ["weekly"]


def test_irregular_gaps_are_not_recurring():
    dates = ["2024-01-01", "2024-01-03", "2024-02-20", "2024-02-21", "2024-04-30", "2024-05-02"]
    df = _txns([("A", d, -250.0, "Dining & Coffee", "Kaldis Coffee") for d in dates])
    assert detect_recurring(df).empty


def test_unstable_amount_is_not_recurring():
    amounts = [-100.0, -900.0, -250.0, -2000.0, -60.0, -1200.0]
    rows = [("A", f"2024-{m:02d}-10", a, "Shopping", "Merkato") for m, a in enumerate(amounts, 1)]
    assert detect_recurring(_txns(rows)).empty


def test_one_spike_does_not_break_a_recurring_bill():
    rows = _monthly("House Rent", "Housing", -8000.0)
    rows[4] = (*rows[4][:2], -80000.0, *rows[4][3:])
    assert len(detect_recurring(_txns(rows))) == 1


def test_too_few_occurrences():
    assert detect_recurring(_txns(_monthly("Iddir", "Community & Charity", -100.0, months=2))).empty


def test_series_are_per_account_and_direction():
    rows = _monthly("Commercial Bank of Ethiopia", "Transfers", -500.0, account="A")
    rows += _monthly("Commercial Bank of Ethiopia", "Transfers", 500.0, account="B")
    found = detect_recurring(_txns(rows))
    assert sorted(zip(found.account_id, found.direction, strict=True)) == [
        ("A", "out"),
        ("B", "in"),
    ]


def test_mark_recurring_flags_member_rows():
    rows = _monthly("DStv", "Entertainment & Subscriptions", -700.0)
    rows.append(("A", "2024-03-14", -250.0, "Dining & Coffee", "Tomoca Coffee"))
    df = _txns(rows)
    marked = mark_recurring(df, detect_recurring(df))
    assert marked.tolist() == [True] * 8 + [False]
    assert series_key(df).iloc[-1] == "Tomoca Coffee | out"


# -- monthly spend ----------------------------------------------------------------------


def test_monthly_spend_and_income():
    df = _txns(
        [
            ("A", "2024-01-03", -100.0, "Groceries", None),
            ("A", "2024-01-20", -50.0, "Groceries", None),
            ("A", "2024-01-26", 9000.0, "Income", None),
            ("A", "2024-02-02", -70.0, "Groceries", None),
            ("B", "2024-01-05", -999.0, "Groceries", None),
        ]
    )
    monthly = monthly_by_category(df)
    jan = monthly[(monthly.account_id == "A") & (monthly.month == "2024-01")].set_index("category")
    assert jan.loc["Groceries", ["spend", "income", "transactions"]].tolist() == [150, 0, 2]
    assert jan.loc["Income", ["spend", "income"]].tolist() == [0, 9000]

    table = spend_table(df, "A", months=1)
    assert table.columns.astype(str).tolist() == ["2024-02"]
    assert table.to_dict()[table.columns[0]] == {"Groceries": 70}


# -- anomalies --------------------------------------------------------------------------


def test_spike_is_flagged_against_prior_history_only():
    rng = np.random.default_rng(0)
    amounts = list(-rng.uniform(400, 600, 12))
    amounts[9] = -5000.0  # 10x spike
    rows = [("A", f"2024-{m:02d}-12", a, "Utilities", "EEU") for m, a in enumerate(amounts, 1)]
    scored = score_anomalies(_txns(rows))

    assert scored["is_anomaly"].tolist() == [i == 9 for i in range(12)]
    assert scored["anomaly_z"].iloc[:5].isna().all()  # not enough history yet
    assert (scored["anomaly_basis"].iloc[5:] == "series").all()


def test_small_amounts_and_refunds_are_not_flagged():
    amounts = [-500.0] * 8 + [-5.0, 300.0]
    rows = [("A", f"2024-01-{d:02d}", a, "Groceries", "Shoa") for d, a in enumerate(amounts, 1)]
    assert not score_anomalies(_txns(rows))["is_anomaly"].any()


def test_falls_back_to_category_history():
    rows = [("A", f"2024-01-{d:02d}", -200.0, "Groceries", f"shop{d}") for d in range(1, 9)]
    rows.append(("A", "2024-01-20", -4000.0, "Groceries", "new shop"))
    scored = score_anomalies(_txns(rows))
    assert scored["anomaly_basis"].iloc[-1] == "category"
    assert scored["is_anomaly"].iloc[-1]


def test_threshold_is_configurable():
    amounts = [-100.0, -110.0, -90.0, -105.0, -95.0, -100.0, -400.0]
    rows = [("A", f"2024-01-{d:02d}", a, "Groceries", "Shoa") for d, a in enumerate(amounts, 1)]
    df = _txns(rows)
    assert score_anomalies(df, AnomalyConfig(threshold=3.0))["is_anomaly"].iloc[-1]
    assert not score_anomalies(df, AnomalyConfig(threshold=50.0))["is_anomaly"].iloc[-1]


def test_isolation_forest_ranks_spike_highest():
    rows = [("A", f"2024-01-{d:02d}", -100.0 - d, "Groceries", "Shoa") for d in range(1, 28)]
    rows.append(("A", "2024-01-28", -3000.0, "Groceries", "Shoa"))
    scored = score_anomalies(_txns(rows))
    assert isolation_forest_scores(scored, scored).argmax() == len(rows) - 1


# -- evaluation on synthetic data -------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_processed():
    from txcat.data.clean import clean_descriptions
    from txcat.data.merchants import MerchantNormalizer
    from txcat.data.synthetic import SyntheticGenerator

    df = SyntheticGenerator(7).generate(n_accounts=6)
    df["merchant"] = MerchantNormalizer().normalize(clean_descriptions(df["description"]))
    return df


def test_account_halves_split_by_id(synthetic_processed):
    dev, held = account_halves(synthetic_processed)
    assert set(dev.account_id) == {"SYN-0001", "SYN-0003", "SYN-0005"}
    assert set(held.account_id) == {"SYN-0002", "SYN-0004", "SYN-0006"}


def test_detectors_beat_chance_on_synthetic(synthetic_processed):
    table, best_z = evaluation_table(synthetic_processed)
    recurring = table[(table.detector == "recurring") & (table.accounts == "held-out")].iloc[0]
    assert recurring.precision > 0.95 and recurring.recall > 0.9
    held = table[(table.detector == "anomaly: robust z") & (table.accounts == "held-out")]
    assert set(held["slice"]) == {"all", "recurring", "discretionary"}
    assert best_z > 0
