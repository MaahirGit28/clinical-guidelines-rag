"""Central configuration for the Clinical Guidelines RAG system."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# --- API keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to .env")

QDRANT_URL = os.getenv("QDRANT_URL")
if not QDRANT_URL:
    raise ValueError("QDRANT_URL not found. Add it to .env")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY not found. Add it to .env")

# --- Models ---
EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1

# --- Chunking ---
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50

# --- Retrieval ---
TOP_K = 8

# --- Qdrant ---
COLLECTION_NAME = "aacap_guidelines"