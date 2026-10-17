import os
import streamlit as st

# --- GOOGLE API KEY SETUP ---
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# # Try to get key from Streamlit Secrets (Cloud) OR Environment (Local)
# if "GOOGLE_API_KEY" in st.secrets:
#     os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
# elif "GOOGLE_API_KEY" in os.environ:
#     pass # Key already set locally
# else:
#     # Fallback for local testing if not using env vars
#     # os.environ["GOOGLE_API_KEY"] = "AIza..." # Only uncomment locally!
#     pass

# --- MODEL CONFIG ---
SEED = 42
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 5
RRF_K = 60  # Standard RRF smoothing constant (Cormack et al., 2009)
MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

# --- SYSTEM SETTINGS ---
STRICT_MODE_THRESHOLD = 0.7