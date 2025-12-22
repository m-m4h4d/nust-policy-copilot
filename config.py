import os

# --- AUTHENTICATION ---
GOOGLE_API_KEY = "AIzaSyBgmbd9UgOlCtpvW7i5aGq_wOPW9zZCz7I"

# Set it in the environment so libraries can find it automatically
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- MODEL CONFIG ---
SEED = 42
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 5
MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

# --- SYSTEM SETTINGS ---
STRICT_MODE_THRESHOLD = 0.7