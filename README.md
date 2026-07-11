# 💳 Spending Assistant

An LLM-powered personal finance dashboard. It ingests raw bank/credit-card transaction exports, cleans and categorizes them, visualizes spending trends, and answers natural-language questions about the data using Google's Gemini API.

Try it in two ways:
- **`app.py`** — an interactive Streamlit dashboard with charts and a chat interface.
- **`spending_assistant_notebook.ipynb`** — the underlying data-cleaning and EDA workflow, plus a Gemini Q&A helper, in notebook form.

## Why I built this

I wanted a fast way to answer questions like "what did I spend the most on last month?" or "what's my spending likely to look like next quarter?" without manually pivoting spreadsheets. The project became a small end-to-end exercise in: cleaning messy real-world transaction data, building a lightweight retrieval layer that hands an LLM only the relevant summary stats (not raw rows), and wrapping it in a usable UI.

## How it works

1. **Ingest & clean** — reads a CSV export, normalizes column names/date formats across different bank export styles, deduplicates, and coerces types.
2. **Summarize** — computes monthly totals, top categories, and top merchants with pandas.
3. **Answer** — routes the user's question to a small prompt-building function that hands Gemini only the relevant aggregated context (e.g. category totals, not every row), then asks it to answer or forecast in plain language.
4. **Visualize** — Streamlit renders spend-by-category and monthly-trend charts alongside a chat-style Q&A panel.

## Tech stack

Python, pandas, Streamlit, matplotlib/seaborn, Google Gemini API (`google-genai`).

## Getting started

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"   # get one at https://aistudio.google.com/apikey
streamlit run app.py
```

For the notebook, launch Jupyter/VS Code from this same folder so the relative path to `sample_transactions.csv` resolves, and set `GEMINI_API_KEY` in your environment before running the Gemini cells.

The project ships with `sample_transactions.csv` — a small synthetic dataset — so it runs out of the box without any personal data. Point it at your own export by changing the CSV path in the sidebar (app) or `DATA_CANDIDATES` (notebook).

## Notes on the data

This repo intentionally ships only synthetic sample data. The version I use day-to-day runs against my real bank export locally, but that file is excluded from version control (see `.gitignore`) since it contains personal financial information.

## Possible next steps

Category auto-tagging with a classifier instead of relying on the bank's own labels, multi-account support, and a proper spend-forecast model instead of prompting an LLM for a point estimate.
