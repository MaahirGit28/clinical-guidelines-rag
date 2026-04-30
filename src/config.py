"""Central configuration for the Clinical Guidelines RAG system."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --- API keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to .env")

# --- Models ---
# bge-m3: 1024-dim, multilingual, ~2.3GB. Already cached from your ingest run.
# Alternative: "BAAI/bge-small-en-v1.5" (130MB, faster, English-only).
EMBED_MODEL = "BAAI/bge-m3"

# Groq's current flagship general-purpose model. Alternatives:
#   "llama-3.1-70b-versatile", "llama-3.1-8b-instant" (faster, weaker)
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1  # Low — clinical answers should be conservative

# --- Chunking ---
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50

# --- Retrieval ---
TOP_K = 8

# --- Chroma ---
COLLECTION_NAME = "aacap_guidelines"