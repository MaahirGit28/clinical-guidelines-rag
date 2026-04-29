"""Query the indexed clinical guidelines via Groq."""
import logging

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


CLINICAL_QA_PROMPT = PromptTemplate(
    "You are a clinical assistant answering questions about psychiatric "
    "clinical practice guidelines. Use ONLY the context below. If the "
    "context does not contain the answer, say so explicitly — do not "
    "speculate or use outside knowledge.\n\n"
    "Cite sources inline using the format [source: <filename>, p. <n>]. "
    "Every clinical claim must carry a citation.\n\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


def load_query_engine():
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = OpenAILike(
        model=LLM_MODEL,
        api_base="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
        is_chat_model=True,
        context_window=8192,
    )

    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = db.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    return index.as_query_engine(
        similarity_top_k=TOP_K,
        text_qa_template=CLINICAL_QA_PROMPT,
        response_mode="compact",
    )


def ask(question: str):
    qe = load_query_engine()
    response = qe.query(question)

    print("\n=== ANSWER ===")
    print(response.response)
    print("\n=== SOURCES ===")
    for i, node in enumerate(response.source_nodes, 1):
        meta = node.node.metadata
        src = (
            meta.get("file_name")
            or meta.get("file_path", "unknown").split("/")[-1]
        )
        page = meta.get("page_label") or meta.get("page") or "?"
        score = node.score if node.score is not None else 0.0
        snippet = node.node.text.replace("\n", " ").strip()[:200]
        print(f"[{i}] {src} (p. {page}) — score={score:.3f}")
        print(f"    {snippet}…")
    return response


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or (
        "What are the recommended first-line treatments for ADHD in "
        "school-age children?"
    )
    ask(q)
