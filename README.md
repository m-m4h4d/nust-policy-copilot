# NUST Policy Copilot

A strictly grounded RAG assistant for NUST University handbooks built with Gemini and LangChain. Get instant answers about Attendance, Grading, Fees, and Hostels with zero hallucinations.

## How to run

1. Install dependencies (Python 3.11+): `pip install -r requirements.txt`
2. Set your Gemini API key, either in a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your-key-here
   ```
   or in `.streamlit/secrets.toml` (used on Streamlit Cloud):
   ```
   GOOGLE_API_KEY = "your-key-here"
   ```
3. Run `streamlit run app.py` to start.

## Evaluation

Run `python eval_final.py`. Retrieval metrics (Recall@5, nDCG@5) run locally and need no API key; the generation and abstention checks call Gemini.

The eval set is `data/eval_set_handwritten.json`: hand-written questions with reference answers, plus a few unanswerable questions for the abstention check. Regenerate it with `python data/build_eval_set.py`.
