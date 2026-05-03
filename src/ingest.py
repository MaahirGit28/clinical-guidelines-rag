"""Parse PDFs from data/raw/, chunk, embed, and store in Qdrant Cloud."""
import logging
from pathlib import Path

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
    RAW_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# Page 1 of each AACAP PDF = abstract + journal masthead. The dense topical
# vocabulary of the abstract dominates retrieval as a near-tie for any query
# about that paper, pushing the actual recommendation pages down to ranks
# 2-6. Day-2 eval (run_20260501_112155) confirmed this: p1 chunks took rank
# 1 on 6+ of 12 questions. Set this to False to revert.
SKIP_FIRST_PAGE = True


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
    skipped_pages = 0
    for pdf_path in pdf_files:
        pages = reader.load_data(file_path=pdf_path)
        for page_num, page_doc in enumerate(pages, start=1):
            if SKIP_FIRST_PAGE and page_num == 1:
                skipped_pages += 1
                continue
            page_doc.metadata["file_name"] = pdf_path.name
            page_doc.metadata["page_label"] = str(page_num)
            documents.append(page_doc)

    if SKIP_FIRST_PAGE:
        log.info(
            f"Loaded {len(documents)} page(s) across {len(pdf_files)} PDF(s) "
            f"(skipped {skipped_pages} first page(s) — abstract bleed mitigation)"
        )
    else:
        log.info(f"Loaded {len(documents)} page(s) across {len(pdf_files)} PDF(s)")

    # Normalize metadata so citations are clean
    for doc in documents:
        if "file_path" in doc.metadata:
            doc.metadata["source"] = Path(doc.metadata["file_path"]).name
        doc.metadata.setdefault(
            "page_label", str(doc.metadata.get("page", "?"))
        )

    # Set up Qdrant — wipe + recreate the collection for clean re-runs.
    # QdrantVectorStore auto-creates the collection on first write.
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        client.delete_collection(COLLECTION_NAME)
        log.info(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    log.info("Embedding + indexing… (first run downloads the embedding model)")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    # Verify by hitting Qdrant directly for the canonical count
    collection_info = client.get_collection(COLLECTION_NAME)
    log.info(
        f"Indexed {collection_info.points_count} chunks into '{COLLECTION_NAME}'"
    )
    return index


if __name__ == "__main__":
    build_index()