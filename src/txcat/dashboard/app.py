"""txcat dashboard. Run: `uv run streamlit run src/txcat/dashboard/app.py`.

Every number on screen comes from the HTTP API (TXCAT_API_URL), so the dashboard shows
exactly what an API consumer would get.
"""

import pandas as pd
import streamlit as st

from txcat.dashboard.client import ApiClient, ApiError
from txcat.dashboard.tables import comparison_table

st.set_page_config(page_title="txcat", page_icon="💳", layout="wide")

EXAMPLES = pd.DataFrame(
    {
        "description": [
            "TELEBIRR AIRTIME RECHARGE FT24123ABCD",
            "ቤት ኪራይ ጥቅምት",
            "Kaldis Coffee Bole branch",
            "CBE transfer to Abebe K. FT24123ABCD",
            "Starbucks Store #1234",
            "xyz qq 77",
        ],
        "amount": [-100.0, -8000.0, -180.0, -2500.0, -6.5, -40.0],
    }
)


@st.cache_resource
def client() -> ApiClient:
    return ApiClient()


def _show_error(exc: ApiError) -> None:
    st.error(str(exc))
    st.caption("Start the API with `uv run uvicorn txcat.api.app:app` or `docker compose up`.")


def page_insights(api: ApiClient) -> None:
    st.header("Account insights")
    accounts = api.accounts()
    if not accounts:
        st.info("No transactions stored yet. Run `uv run txcat bootstrap` to load the data.")
        return
    labels = {
        f"{a['account_id']}  ({a['transactions']} txns, {a['first_date']} → {a['last_date']})": a
        for a in accounts
    }
    # Default to the first synthetic account: they have the richest history (2.5 years).
    default = next((i for i, a in enumerate(accounts) if a["account_id"].startswith("SYN-")), 0)
    account = labels[st.selectbox("Account", list(labels), index=default)]
    months = st.slider("Months of spend", 3, 24, 12)
    data = api.insights(account["account_id"], months=months, anomalies=20)
    synthetic = data["account_id"].startswith("SYN-")

    cols = st.columns(4)
    cols[0].metric("Transactions", f"{data['transactions']:,}")
    cols[1].metric("Recurring payments", len(data["recurring"]))
    cols[2].metric("Unusual transactions", len(data["anomalies"]))
    first, last = pd.Timestamp(data["first_date"]), pd.Timestamp(data["last_date"])
    cols[3].metric(
        "Months of history", (last.year - first.year) * 12 + last.month - first.month + 1
    )
    if synthetic:
        st.caption("Synthetic account: generated data, not a real customer.")

    st.subheader("Recurring payments")
    recurring = pd.DataFrame(data["recurring"])
    if recurring.empty:
        st.write("None detected.")
    else:
        st.dataframe(
            recurring[
                [
                    "counterparty",
                    "direction",
                    "category",
                    "period",
                    "typical_amount",
                    "occurrences",
                    "last_seen",
                    "next_expected",
                ]
            ],
            hide_index=True,
            column_config={
                "typical_amount": st.column_config.NumberColumn("typical amount", format="%.0f"),
                "next_expected": st.column_config.DateColumn("next due"),
            },
        )

    st.subheader("Monthly spend by category")
    monthly = pd.DataFrame(data["monthly_spend"])
    if not monthly.empty:
        chart = monthly.pivot_table(
            index="month", columns="category", values="spend", aggfunc="sum", fill_value=0
        )
        chart = chart.loc[:, chart.sum() > 0]
        st.bar_chart(chart, stack=True, height=380)

    st.subheader("Unusual transactions")
    st.caption(
        "Amount far above this account's own history for the same merchant or category "
        "(robust z-score ≥ 5, computed only from earlier transactions)."
    )
    anomalies = pd.DataFrame(data["anomalies"])
    if anomalies.empty:
        st.write("None flagged.")
    else:
        st.dataframe(
            anomalies.sort_values("date", ascending=False),
            hide_index=True,
            column_config={"anomaly_z": st.column_config.NumberColumn("z", format="%.1f")},
        )


def page_try(api: ApiClient) -> None:
    st.header("Try it: categorize descriptions")
    st.caption(
        "Edit the table or upload a CSV with `description` and optional `amount` columns "
        "(negative = money out). Low-confidence rows are sent to the review queue."
    )
    upload = st.file_uploader("CSV", type="csv")
    rows = pd.read_csv(upload) if upload else EXAMPLES
    edited = st.data_editor(rows, num_rows="dynamic", hide_index=True, key="rows")
    if not st.button("Categorize", type="primary"):
        return
    edited = edited.dropna(subset=["description"])
    edited = edited[edited["description"].astype(str).str.strip() != ""]
    payload = [
        {
            "description": str(r["description"]),
            "amount": None if pd.isna(r.get("amount")) else float(r["amount"]),
        }
        for r in edited.to_dict("records")
    ]
    if not payload:
        st.warning("Add at least one description.")
        return
    result = api.categorize(payload)
    predictions = pd.DataFrame(result["predictions"])
    queued = int(predictions["needs_review"].sum())
    st.success(
        f"{len(predictions)} categorized with {result['model_version']}; "
        f"{queued} below the {result['threshold']:.2f} threshold went to the review queue."
    )
    st.dataframe(
        predictions[["description", "category", "confidence", "merchant", "needs_review"]],
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )


def page_review(api: ApiClient) -> None:
    st.header("Review queue")
    model = api.model()
    st.caption(
        f"Predictions with confidence below {model['threshold']:.2f}. Corrections are stored "
        "as labelled training data for the next model."
    )
    pending = api.review_queue("pending", limit=50)
    if not pending:
        st.success("Nothing to review.")
    categories = sorted(model["categories"])
    for item in pending:
        with st.form(f"review-{item['id']}", border=True):
            st.markdown(f"**{item['description']}**")
            amount = "" if item["amount"] is None else f" · amount {item['amount']:,.2f}"
            st.caption(
                f"#{item['id']} · predicted {item['predicted_category']} "
                f"({item['confidence']:.0%}){amount} · {item['model_version']}"
            )
            default = (
                categories.index(item["predicted_category"])
                if item["predicted_category"] in categories
                else 0
            )
            choice = st.selectbox("Correct category", categories, index=default)
            if st.form_submit_button("Save"):
                api.review(item["id"], choice)
                st.toast(f"#{item['id']} saved as {choice}")
                st.rerun()

    with st.expander("Recently reviewed"):
        reviewed = pd.DataFrame(api.review_queue("reviewed", limit=50))
        if reviewed.empty:
            st.write("None yet.")
        else:
            st.dataframe(
                reviewed[
                    ["id", "description", "predicted_category", "reviewed_category", "reviewed_at"]
                ],
                hide_index=True,
            )


def page_model(api: ApiClient) -> None:
    st.header("Model")
    health = api.health()
    model = api.model()
    cols = st.columns(3)
    cols[0].metric("Served model", model["model_version"])
    cols[1].metric("Review threshold", f"{model['threshold']:.3f}")
    cols[2].metric("Database", health["database"])
    st.markdown(
        "Time-based evaluation: each source is split by date (oldest 70% train, next 15% "
        "validation, newest 15% test). Hyperparameters and the threshold are chosen on "
        "validation; the table shows the future **test** slice. *Unseen text* = test rows "
        "whose description never appeared in training, the most honest column."
    )
    comparison = comparison_table(model["comparison"])
    if comparison.empty:
        st.info("No comparison table shipped with this model.")
        return
    f1_columns = [c for c in comparison.columns if c.startswith("test macro-F1")]
    st.dataframe(
        comparison.style.format("{:.3f}", subset=list(comparison.columns[1:])).highlight_max(
            subset=f1_columns, color="#c8e6c9"
        ),
        hide_index=True,
    )
    st.caption(
        "Synthetic scores are near-perfect because that data is template-generated; the "
        "Kaggle test slice is small (123 rows). See MODEL_CARD.md for limitations."
    )


PAGES = {
    "Account insights": page_insights,
    "Try it": page_try,
    "Review queue": page_review,
    "Model": page_model,
}


def main() -> None:
    api = client()
    st.sidebar.title("txcat")
    st.sidebar.caption("Bank-transaction categorization & insights")
    # Deep links: ?page=review etc. (first word of the page name, lower-case).
    slugs = [name.split()[0].lower() for name in PAGES]
    wanted = st.query_params.get("page", "")
    start = slugs.index(wanted) if wanted in slugs else 0
    page = st.sidebar.radio("Page", list(PAGES), index=start, label_visibility="collapsed")
    st.sidebar.caption(f"API: {api.base_url}")
    try:
        PAGES[page](api)
    except ApiError as exc:
        _show_error(exc)


main()
