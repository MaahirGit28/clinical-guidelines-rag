"""Build the vector index from PDFs in data/raw/.

Idempotent: drops and rebuilds the Chroma collection every run. Safe to re-run
after dropping new PDFs into data/raw/ or after tweaking chunking in config.py.

Usage:
    python -m src.ingest
"""
from __future__ import annotations

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    RAW_PDF_DIR,
)


def build_index() -> VectorStoreIndex:
    # 1. Configure embeddings + chunker globally
    print(f"[1/4] Loading embedding model: {EMBED_MODEL_NAME}")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # 2. Load PDFs. PyMuPDFReader yields one Document per page, with page_label
    #    metadata — essential for citation accuracy.
    print(f"[2/4] Reading PDFs from {RAW_PDF_DIR}")
    pdfs = list(RAW_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {RAW_PDF_DIR}. "
            "Run scripts/download_guidelines.py first."
        )
    reader = SimpleDirectoryReader(
        input_dir=str(RAW_PDF_DIR),
        file_extractor={".pdf": PyMuPDFReader()},
    )
    docs = reader.load_data()
    n_files = len({d.metadata.get("file_name") for d in docs})
    print(f"      Loaded {len(docs)} pages across {n_files} guidelines")

    # 3. Reset the Chroma collection for a clean rebuild
    print(f"[3/4] Resetting Chroma collection '{COLLECTION_NAME}'")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 — fine if it didn't exist yet
        pass
    chroma_collection = chroma_client.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Chunk + embed + persist
    print(f"[4/4] Building index (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=True,
    )
    print(f"\n✓ Done. Collection '{COLLECTION_NAME}' has {chroma_collection.count()} chunks.")
    print(f"  Persisted to {CHROMA_DIR}")
    return index


if __name__ == "__main__":
    build_index()
