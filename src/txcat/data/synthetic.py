"""Seeded generator of SYNTHETIC Ethiopian-style bank transactions.

Every row is tagged `source="synthetic"`. Descriptions mix English and Amharic, use
real-world statement conventions (telebirr airtime, CBE transfers with FT references,
Ethio Telecom packages, ATM withdrawals) and realistic noise (case, references, phone
numbers, typos). Account holders and counterparties are randomly generated people;
no real customer data is used.

Each account follows a profile: monthly salary, optional rent/iddir/equb/DStv and
utilities on a schedule (so recurring-payment detection has something to find), plus
Poisson-distributed discretionary spending. A small share of rows are amount spikes,
flagged in `is_injected_anomaly` for evaluating the anomaly detector; rows from the
recurring profile are flagged `is_recurring_true` for evaluating the recurring detector.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from txcat.data.catalog import MERCHANTS, Merchant

FIRST_NAMES = (
    "Abebe",
    "Almaz",
    "Bekele",
    "Bethlehem",
    "Biniyam",
    "Dawit",
    "Eyerusalem",
    "Eyob",
    "Fikirte",
    "Getachew",
    "Hana",
    "Henok",
    "Kidist",
    "Lemlem",
    "Liya",
    "Mekdes",
    "Meron",
    "Nahom",
    "Rahel",
    "Samuel",
    "Saron",
    "Selam",
    "Tesfaye",
    "Tigist",
    "Yonas",
    "Zewdu",
    "Hamza",
    "Amina",
    "Ahmed",
    "Fatuma",
    "Chaltu",
    "Gemechu",
    "Lensa",
    "Tolosa",
)
EMPLOYERS = (
    "Abay Trading PLC",
    "Sheba Tech PLC",
    "Entoto Consulting",
    "Awash Agro PLC",
    "Gihon Engineering",
    "Lalibela Logistics",
    "Addis Ababa City Admin",
    "Rift Valley NGO",
)
BRANCHES = ("bole", "piassa", "megenagna", "kazanchis", "cmc", "sarbet", "4 kilo", "mexico")
MONTHS_AM = (
    "መስከረም",
    "ጥቅምት",
    "ህዳር",
    "ታህሳስ",
    "ጥር",
    "የካቲት",
    "መጋቢት",
    "ሚያዝያ",
    "ግንቦት",
    "ሰኔ",
    "ሐምሌ",
    "ነሐሴ",
)

# Description templates per category. {m}=merchant alias, {name}=counterparty,
# {ref}=bank reference, {phone}=phone number, {branch}=area, {mon}=month name.
TEMPLATES: dict[str, tuple[str, ...]] = {
    "Telecom & Airtime": (
        "{m} {phone}",
        "{m} airtime {phone}",
        "{m} recharge",
        "{m} 1gb weekly package",
        "{m} የአየር ሰዓት ግዢ",
        "airtime top-up {m} {ref}",
        "{m} ካርድ ግዢ",
    ),
    "Transfers": (
        "{m} transfer to {name} {ref}",
        "{m} trf {name}",
        "ገንዘብ ማስተላለፍ ለ{name} {m}",
        "{m} send money {phone}",
        "{m} p2p {name} {ref}",
        "transfer from {name} via {m}",
    ),
    "Cash Withdrawal": ("{m} {branch} branch", "{m} {ref}", "{m} ገንዘብ ወጪ {branch}"),
    "Loan & Card Payments": ("{m} {ref}", "{m} {mon}", "monthly {m}"),
    "Housing": ("{m} {mon}", "{m} to {name}", "{m} {name} {ref}"),
    "Utilities": ("{m} {ref}", "{m} bill {mon}", "{m} postpaid", "{m} {mon}"),
    "Groceries": ("{m}", "{m} {branch}", "pos purchase {m}", "{m} ግዢ", "pos {m} {ref}"),
    "Dining & Coffee": ("{m}", "{m} {branch}", "pos {m}", "{m} ቁርስ", "{m} {ref}"),
    "Transport & Fuel": ("{m} trip", "{m} {branch}", "{m} {ref}", "{m} ክፍያ"),
    "Shopping": ("{m}", "pos {m} {ref}", "{m} purchase", "{m} ግዢ"),
    "Entertainment & Subscriptions": ("{m} {mon}", "{m} subscription renewal", "{m} {ref}"),
    "Health & Personal Care": ("{m}", "{m} {branch}", "{m} card", "{m} ክፍያ"),
    "Insurance": ("{m} premium {ref}", "{m} motor policy", "{m} {mon}"),
    "Education": ("{m} {mon}", "{m} term fee {ref}", "{m} for {name}"),
    "Travel": ("{m} ticket {ref}", "{m} booking", "{m} {ref}"),
    "Government & Fees": ("{m} {ref}", "{m} {mon}", "{m}"),
    "Community & Charity": ("{m} {mon}", "{m} monthly contribution", "{m} መዋጮ", "{m}"),
    "Savings": ("{m} {mon}", "{m} contribution {name}", "{m} {ref}"),
    "Income": (
        "salary {employer}",
        "payroll credit {employer} {ref}",
        "ደመወዝ {employer}",
        "{employer} salary {mon}",
    ),
}

# (median ETB, log-normal sigma) per category for discretionary amounts.
AMOUNTS: dict[str, tuple[float, float]] = {
    "Telecom & Airtime": (100, 0.6),
    "Transfers": (1500, 1.0),
    "Cash Withdrawal": (2000, 0.7),
    "Groceries": (900, 0.7),
    "Dining & Coffee": (250, 0.6),
    "Transport & Fuel": (350, 0.8),
    "Shopping": (1800, 0.9),
    "Health & Personal Care": (600, 0.9),
    "Travel": (12000, 0.6),
    "Government & Fees": (400, 1.2),
    "Education": (6000, 0.5),
    "Insurance": (3000, 0.5),
}
# Expected discretionary transactions per month per category.
MONTHLY_RATES: dict[str, float] = {
    "Telecom & Airtime": 6,
    "Transfers": 4,
    "Cash Withdrawal": 2,
    "Groceries": 5,
    "Dining & Coffee": 6,
    "Transport & Fuel": 5,
    "Shopping": 1.5,
    "Health & Personal Care": 0.6,
    "Travel": 0.1,
    "Government & Fees": 0.8,
    "Education": 0.2,
    "Insurance": 0.1,
}
ANOMALY_RATE = 0.005


@dataclass
class AccountProfile:
    account_id: str
    holder: str
    employer: str
    salary: float
    rent: float | None
    iddir: float | None
    equb: float | None
    dstv: bool
    loan: tuple[str, float] | None
    payday: int
    rent_day: int


def _merchants_by_category(region: str = "ET") -> dict[str, list[Merchant]]:
    grouped: dict[str, list[Merchant]] = {}
    for merchant in MERCHANTS:
        if merchant.region == region:
            grouped.setdefault(merchant.category, []).append(merchant)
    return grouped


class SyntheticGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.fake = Faker()
        self.fake.seed_instance(seed)
        self.merchants = _merchants_by_category()

    # -- small random helpers ------------------------------------------------------------
    def _choice(self, seq):
        return seq[self.rng.integers(len(seq))]

    def _name(self) -> str:
        return f"{self._choice(FIRST_NAMES)} {self._choice(FIRST_NAMES)[0]}."

    def _ref(self) -> str:
        return self._choice(("FT", "TT", "REF")) + self.fake.bothify("##%##???").upper()

    def _phone(self) -> str:
        prefix = self._choice(("09", "07", "+2519"))
        return prefix + "".join(str(d) for d in self.rng.integers(0, 10, 8))

    def _noise(self, text: str) -> str:
        r = self.rng.random()
        if r < 0.25:
            text = text.upper()
        elif r < 0.35:
            text = text.title()
        if self.rng.random() < 0.08 and len(text) > 8:  # drop one character (typo)
            i = int(self.rng.integers(1, len(text) - 1))
            text = text[:i] + text[i + 1 :]
        if self.rng.random() < 0.1:
            text = "  ".join(text.split(" ", 1)) if " " in text else text
        return text

    def _describe(self, category: str, merchant: Merchant | None, day: date, prof) -> str:
        template = self._choice(TEMPLATES[category])
        alias = self._choice(merchant.aliases) if merchant else ""
        text = template.format(
            m=alias,
            name=self._name(),
            ref=self._ref(),
            phone=self._phone(),
            branch=self._choice(BRANCHES),
            mon=self._choice((day.strftime("%b"), MONTHS_AM[day.month - 1])),
            employer=prof.employer,
        )
        return self._noise(text)

    # -- generation ----------------------------------------------------------------------
    def _profile(self, idx: int) -> AccountProfile:
        salary = float(np.round(self.rng.lognormal(np.log(22000), 0.6), -2))
        return AccountProfile(
            account_id=f"SYN-{idx:04d}",
            holder=f"{self._choice(FIRST_NAMES)} {self._choice(FIRST_NAMES)}",
            employer=self._choice(EMPLOYERS),
            salary=salary,
            rent=float(np.round(salary * self.rng.uniform(0.2, 0.4), -2))
            if self.rng.random() < 0.7
            else None,
            iddir=float(self._choice((50, 100, 150, 200))) if self.rng.random() < 0.6 else None,
            equb=float(self._choice((500, 1000, 2000, 5000))) if self.rng.random() < 0.4 else None,
            dstv=bool(self.rng.random() < 0.5),
            loan=(
                self._choice(("CBE Loan", "Condominium Loan")),
                float(np.round(salary * self.rng.uniform(0.1, 0.25), -1)),
            )
            if self.rng.random() < 0.3
            else None,
            payday=int(self.rng.integers(25, 29)),
            rent_day=int(self.rng.integers(1, 6)),
        )

    def _row(self, prof, day, category, amount, merchant_name=None, desc=None):
        merchant = None
        if merchant_name:
            merchant = next(m for m in MERCHANTS if m.name == merchant_name)
        elif category in self.merchants:
            merchant = self._choice(self.merchants[category])
        return {
            "account_id": prof.account_id,
            "date": day,
            "description": desc or self._describe(category, merchant, day, prof),
            "amount": round(amount, 2),
            "currency": "ETB",
            "category": category,
            "merchant_true": merchant.name if merchant else prof.employer,
            "is_injected_anomaly": False,
            "is_recurring_true": False,
        }

    def _on(self, month: date, day_of_month: int, jitter: int = 1) -> date:
        d = day_of_month + int(self.rng.integers(-jitter, jitter + 1))
        return month.replace(day=min(max(d, 1), 28))

    def _around(self, base: float, spread: float) -> float:
        return base * (1 + self.rng.uniform(-spread, spread))

    def _recurring(self, prof: AccountProfile, start: date, end: date) -> list[dict]:
        rows = []
        month = date(start.year, start.month, 1)
        while month <= end:
            rows.append(
                self._row(
                    prof, self._on(month, prof.payday), "Income", self._around(prof.salary, 0.02)
                )
            )
            if prof.rent:
                rows.append(
                    self._row(
                        prof, self._on(month, prof.rent_day), "Housing", -prof.rent, "House Rent"
                    )
                )
            rows.append(
                self._row(
                    prof,
                    self._on(month, 12, 3),
                    "Utilities",
                    -self._around(450, 0.35),
                    "Ethiopian Electric Utility",
                )
            )
            rows.append(
                self._row(
                    prof,
                    self._on(month, 18, 3),
                    "Utilities",
                    -self._around(180, 0.3),
                    "AAWSA Water",
                )
            )
            if prof.iddir:
                rows.append(
                    self._row(prof, self._on(month, 3), "Community & Charity", -prof.iddir, "Iddir")
                )
            if prof.equb:
                rows.append(
                    self._row(
                        prof, self._on(month, prof.payday + 1, 0), "Savings", -prof.equb, "Equb"
                    )
                )
            if prof.loan:
                rows.append(
                    self._row(
                        prof,
                        self._on(month, 10, 2),
                        "Loan & Card Payments",
                        -prof.loan[1],
                        prof.loan[0],
                    )
                )
            if prof.dstv:
                rows.append(
                    self._row(
                        prof,
                        self._on(month, 8, 0),
                        "Entertainment & Subscriptions",
                        -float(self._choice((700, 1100))),
                        "DStv",
                    )
                )
            month = (month + timedelta(days=32)).replace(day=1)
        for r in rows:
            r["is_recurring_true"] = True  # ground truth for the recurring-payment detector
        return [r for r in rows if start <= r["date"] <= end]

    def _discretionary(self, prof: AccountProfile, start: date, end: date) -> list[dict]:
        rows = []
        n_days = (end - start).days + 1
        months = n_days / 30.4
        for category, rate in MONTHLY_RATES.items():
            median, sigma = AMOUNTS[category]
            for _ in range(self.rng.poisson(rate * months)):
                day = start + timedelta(days=int(self.rng.integers(n_days)))
                amount = float(self.rng.lognormal(np.log(median), sigma))
                sign = 1 if category == "Transfers" and self.rng.random() < 0.3 else -1
                rows.append(self._row(prof, day, category, sign * amount))
        return rows

    def generate(
        self,
        n_accounts: int = 30,
        start: date = date(2023, 1, 1),
        end: date = date(2025, 6, 30),
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for idx in range(1, n_accounts + 1):
            prof = self._profile(idx)
            rows += self._recurring(prof, start, end)
            rows += self._discretionary(prof, start, end)

        df = pd.DataFrame(rows)
        spikes = self.rng.random(len(df)) < ANOMALY_RATE
        spikes &= df["category"].ne("Income").to_numpy()
        df.loc[spikes, "amount"] = (df.loc[spikes, "amount"] * self.rng.uniform(6, 12)).round(2)
        df.loc[spikes, "is_injected_anomaly"] = True

        df["date"] = pd.to_datetime(df["date"])
        df["source"] = "synthetic"
        df["dataset"] = "synthetic_et"
        return df.sort_values(["date", "account_id"], kind="stable").reset_index(drop=True)
