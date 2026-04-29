"""Parse PDFs from data/raw/, chunk, embed, and store in ChromaDB."""
import logging
from pathlib import Path

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
    EMBED_MODEL,
    RAW_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def build_index() -> VectorStoreIndex:
    # Global LlamaIndex settings — no LLM needed for ingestion
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    Settings.llm = None

    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {RAW_DIR}. Add PDFs or run "
            "scripts/download_guidelines.py first."
        )
    log.info(f"Found {len(pdf_files)} PDF(s)")

    # PyMuPDF gives much cleaner text than pypdf for scientific PDFs
    reader = PyMuPDFReader()
    documents = []
    for pdf_path in pdf_files:
        pages = reader.load_data(file_path=pdf_path)
        for page_num, page_doc in enumerate(pages, start=1):
            page_doc.metadata["file_name"] = pdf_path.name
            page_doc.metadata["page_label"] = str(page_num)
            documents.append(page_doc)
    log.info(f"Loaded {len(documents)} page(s) across {len(pdf_files)} PDF(s)")

    # Normalize metadata so citations are clean
    for doc in documents:
        if "file_path" in doc.metadata:
            doc.metadata["source"] = Path(doc.metadata["file_path"]).name
        # PyMuPDFReader sets "source" too sometimes; ensure page label exists
        doc.metadata.setdefault(
            "page_label", str(doc.metadata.get("page", "?"))
        )

    # Set up Chroma — wipe + recreate the collection for clean re-runs
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        db.delete_collection(COLLECTION_NAME)
        log.info(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass
    collection = db.create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    log.info("Embedding + indexing… (first run downloads the embedding model)")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    log.info(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}'")
    return index


if __name__ == "__main__":
    build_index()