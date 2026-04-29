"""Centralized configuration.

Everything tunable lives here so Day-2 chunking/retrieval iteration is one-file edits.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_PDF_DIR: Path = DATA_DIR / "raw"
CHROMA_DIR: Path = DATA_DIR / "chroma_db"

RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --- Vector store ------------------------------------------------------------
COLLECTION_NAME: str = "aacap_guidelines"

# --- Embedding model ---------------------------------------------------------
# BGE-M3: 1024-dim, multilingual, supports dense + sparse + multi-vector in one model.
# Downloads ~2.3 GB on first use to ~/.cache/huggingface/.
EMBED_MODEL_NAME: str = "BAAI/bge-m3"

# --- LLM ---------------------------------------------------------------------
# Sonnet 4.6 = balanced default. Swap to claude-opus-4-7 for hardest queries,
# claude-haiku-4-5-20251001 for cheap iteration during dev.
LLM_MODEL: str = "claude-sonnet-4-6"
LLM_MAX_TOKENS: int = 1024

# --- Chunking (tune on Day 2) ------------------------------------------------
# Clinical guidelines have dense, structured prose. 512 tokens with 100-token
# overlap is a reasonable starting point; expect to revisit.
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 100

# --- Retrieval (tune on Day 2) -----------------------------------------------
SIMILARITY_TOP_K: int = 5

