import os
import html
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

st.set_page_config(
    page_title="Spending Assistant",
    page_icon="💳",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f0f;
    color: #f0ece4;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
section[data-testid="stSidebar"] {
    background-color: #161616;
    border-right: 1px solid #2a2a2a;
}
.user-bubble {
    background: #1e3a2f;
    border: 1px solid #2d5c45;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0 8px 20%;
    color: #d4f5e2;
    font-size: 0.95rem;
}
.assistant-bubble {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 20% 8px 0;
    color: #f0ece4;
    font-size: 0.95rem;
    line-height: 1.6;
}
.bubble-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
    opacity: 0.5;
}
.metric-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #7effc4;
}
.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.5;
    margin-top: 4px;
}
div[data-testid="stTextInput"] input {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 10px !important;
    color: #f0ece4 !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stButton"] > button {
    background: #7effc4 !important;
    color: #0f0f0f !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stButton"] > button:hover {
    background: #5de0a8 !important;
}
</style>
""", unsafe_allow_html=True)


def escape_text(value: str) -> str:
    return html.escape(str(value)).replace("\n", "<br>")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "transactiondate": "transaction_date",
        "transaction_date": "transaction_date",
        "posting_date": "posting_date",
        "description": "merchant",
        "merchant": "merchant",
        "spend_category": "category",
        "category": "category",
        "amount_($)": "amount",
        "amount($)": "amount",
        "amount": "amount",
    }
    df = df.rename(columns=rename_map)

    required = ["transaction_date", "merchant", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    if "category" not in df.columns:
        df["category"] = "Uncategorized"

    raw_dates = df["transaction_date"].astype(str)
    parsed = pd.to_datetime(df["transaction_date"], errors="coerce")
    if parsed.isna().mean() > 0.3:
        parsed = pd.to_datetime(raw_dates + "-2025", format="%d-%b-%Y", errors="coerce")
    df["transaction_date"] = parsed

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["merchant"] = df["merchant"].astype(str).str.strip().str.title()
    df["category"] = df["category"].astype(str).str.strip().str.title()
    df = df.dropna(subset=["transaction_date", "merchant", "amount"]).copy()
    df = df.sort_values("transaction_date").reset_index(drop=True)
    return df


def build_prompt(question: str, df: pd.DataFrame) -> tuple[str, str]:
    q = question.lower().strip()

    if "top" in q and "category" in q:
        top = df.groupby("category", dropna=False)["amount"].sum().sort_values(ascending=False).head(5)
        context = top.to_string()
        prompt = (
            f"Here are the top spending categories with totals:\n{context}\n\n"
            "Explain these results in one clear paragraph for a personal spending dashboard."
        )
    elif "merchant" in q and ("most" in q or "top" in q):
        top_m = df.groupby("merchant", dropna=False)["amount"].sum().sort_values(ascending=False).head(5)
        context = top_m.to_string()
        prompt = (
            f"Here are the top merchants with totals:\n{context}\n\n"
            "Answer the question briefly in 2-3 sentences."
        )
    elif "total" in q and ("november" in q or "nov" in q):
        nov = df[df["transaction_date"].dt.month == 11]
        total = nov["amount"].sum()
        context = f"Total spending in November: ${total:,.2f}"
        prompt = f"{context}\n\nExplain this result in one clear sentence."
    elif "total" in q and ("december" in q or "dec" in q):
        dec = df[df["transaction_date"].dt.month == 12]
        total = dec["amount"].sum()
        context = f"Total spending in December: ${total:,.2f}"
        prompt = f"{context}\n\nExplain this result in one clear sentence."
    elif "total spending" in q or q == "total" or q.startswith("what is my total"):
        total = df["amount"].sum()
        context = f"Total spending across all months: ${total:,.2f}"
        prompt = f"{context}\n\nExplain this result in one clear sentence."
    elif "predict" in q or "forecast" in q or "2026" in q:
        monthly = df.groupby(df["transaction_date"].dt.to_period("M"))["amount"].sum()
        context = monthly.to_string()
        prompt = (
            f"Here is monthly spending data:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Based on these trends, give a concise forecast with a specific dollar estimate and brief reasoning."
        )
    else:
        summary = df.groupby("category")["amount"].sum().sort_values(ascending=False).head(10).to_string()
        context = summary
        prompt = (
            f"Answer the user's spending question using this category summary:\n{context}\n\n"
            f"Question: {question}\n\nProvide a concise, helpful answer in 2-3 sentences."
        )
    return context, prompt


def ask_gemini(question: str, df: pd.DataFrame, api_key: str, model: str) -> tuple[str, str, str]:
    if genai is None:
        raise ImportError("Missing package: install with pip install google-genai")
    if not api_key:
        raise ValueError("Missing Gemini API key. Add it in the sidebar or set GEMINI_API_KEY.")

    context, prompt = build_prompt(question, df)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        ),
    )

    answer = getattr(response, "text", "") or "No response returned by Gemini."
    return context, prompt, answer.strip()

def make_category_chart(df: pd.DataFrame):
    cat = df.groupby("category")["amount"].sum().sort_values(ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#161616")
    ax.set_facecolor("#161616")
    ax.barh(cat.index[::-1], cat.values[::-1], color="#7effc4", height=0.6)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#2a2a2a", linewidth=0.8)
    plt.tight_layout()
    return fig


def make_monthly_chart(df: pd.DataFrame):
    monthly = df.groupby(df["transaction_date"].dt.strftime("%b"))["amount"].sum()
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = monthly.reindex([m for m in month_order if m in monthly.index])
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#161616")
    ax.set_facecolor("#161616")
    ax.plot(monthly.index, monthly.values, color="#7effc4", linewidth=2.5, marker="o", markersize=5)
    ax.fill_between(monthly.index, monthly.values, alpha=0.1, color="#7effc4")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#2a2a2a", linewidth=0.8)
    plt.tight_layout()
    return fig


if "messages" not in st.session_state:
    st.session_state.messages = []

csv_candidates = [
    "sample_transactions.csv",
    "unified_transactions-4.csv",
    "transactions_clean_candidate-2025-all-months-4.csv",
]
default_csv = next((p for p in csv_candidates if os.path.exists(p)), csv_candidates[0])

# API key is never hardcoded. Set it via the GEMINI_API_KEY environment
# variable (recommended) or paste it into the sidebar field at runtime.
default_key = os.getenv("GEMINI_API_KEY", "")

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    csv_path = st.text_input("CSV path", value=default_csv)
    model_name = st.text_input("Gemini model", value="gemini-2.5-flash")
    api_key = st.text_input("Gemini API key", value=default_key, type="password")
    st.caption("Tip: you can also set GEMINI_API_KEY in your environment.")
    st.markdown("---")
    if st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**How to use**")
    st.caption("1. Add your Gemini API key")
    st.caption("2. Check the CSV path")
    st.caption("3. Ask a question below")

try:
    df = load_data(csv_path)
    data_ok = True
except Exception as e:
    st.error(f"Could not load `{csv_path}`: {e}")
    data_ok = False

st.markdown("# 💳 Spending Assistant")
st.caption("Powered by your data + Gemini API")

if data_ok:
    total = df["amount"].sum()
    top_cat = df.groupby("category")["amount"].sum().idxmax() if not df.empty else "N/A"
    months = df["transaction_date"].dt.to_period("M").nunique() if not df.empty else 0
    avg_monthly = total / months if months else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"${total:,.0f}", "Total Spent"),
        (c2, f"${avg_monthly:,.0f}", "Avg / Month"),
        (c3, str(len(df)), "Transactions"),
        (c4, top_cat.split()[0] if top_cat != "N/A" else "N/A", "Top Category"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{escape_text(val)}</div>'
            f'<div class="metric-label">{escape_text(label)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**Spend by Category**")
        st.pyplot(make_category_chart(df))
    with ch2:
        st.markdown("**Monthly Trend**")
        st.pyplot(make_monthly_chart(df))

    st.markdown("---")
    st.markdown("### 💬 Ask a question")

    suggestions = [
        "What are my top spending categories?",
        "What is the total spending in November?",
        "Which merchant did I spend the most on?",
        "Predict my expenses in November 2026",
    ]

    cols = st.columns(len(suggestions))
    for col, s in zip(cols, suggestions):
        if col.button(s, key=s):
            st.session_state.messages.append({"role": "user", "content": s})
            with st.spinner("Thinking..."):
                try:
                    _, _, answer = ask_gemini(s, df, api_key, model_name)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Gemini error: {e}"})
            st.rerun()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble"><div class="bubble-label">You</div>{escape_text(msg["content"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble"><div class="bubble-label">Assistant</div>{escape_text(msg["content"])}</div>',
                unsafe_allow_html=True,
            )

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask anything about your spending...", label_visibility="collapsed")
        submitted = st.form_submit_button("Send →")

    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            try:
                _, _, answer = ask_gemini(user_input, df, api_key, model_name)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Gemini error: {e}"})
        st.rerun()
else:
    st.info("Fix the CSV path in the sidebar first, then the dashboard will appear.")
