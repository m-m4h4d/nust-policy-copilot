import os
from dotenv import load_dotenv

# --- GOOGLE API KEY SETUP ---
# Priority: existing env var > .env file > Streamlit secrets (Cloud).
# Loaded here (not in app.py) so every entry point, including eval_final.py, gets the key.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

if not os.environ.get("GOOGLE_API_KEY"):
    try:
        import streamlit as st
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass  # No secrets.toml / key missing: generation calls will fail, retrieval still works

# --- MODEL CONFIG ---
SEED = 42
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 5
RRF_K = 60  # Standard RRF smoothing constant (Cormack et al., 2009)
# With RETRIEVAL_K=5 per retriever, DENSE_WEIGHT=1.05 means BM25's top hit outranks dense's 5th
# result but nothing above it: roughly "dense top 4 + best BM25 hit". At >= 1.07 the fused top 5
# is always the dense top 5. Tuned on data/eval_set_handwritten.json (Recall@5 0.957 vs 0.935 dense-only).
DENSE_WEIGHT = 1.05
SPARSE_WEIGHT = 1.0
MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

# --- SYSTEM SETTINGS ---
STRICT_MODE_THRESHOLD = 0.7
