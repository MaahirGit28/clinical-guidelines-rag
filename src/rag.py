"""Query the AACAP guidelines index with citation-disciplined responses.

Usage:
    python -m src.rag                         # runs the demo question below
    from src.rag import query
    result = query("What is first-line treatment for adolescent MDD?")
"""
from __future__ import annotations

from typing import Any

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import (
    ANTHROPIC_API_KEY,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    SIMILARITY_TOP_K,
)

# Citation discipline matters more than prose here. The prompt explicitly
# (a) restricts to provided context, (b) requires inline source attribution,
# (c) tells the model to abstain when context is insufficient, and
# (d) flags the AACAP 5-year currency convention.
CITATION_PROMPT = PromptTemplate(
    "You are a clinical reference assistant for child and adolescent mental health.\n"
    "Answer the question using ONLY the context below, drawn from AACAP clinical "
    "practice guidelines.\n\n"
    "Rules:\n"
    "1. After every factual claim, cite the source as [filename, p. N].\n"
    "2. If the context does not contain enough information, say so explicitly. "
    "Do not fabricate.\n"
    "3. If the cited material is older than 5 years, note that AACAP guidance "
    "is conventionally considered outdated after 5 years and recommend verification.\n"
    "4. Do not give individualized medical advice; this is a reference tool.\n\n"
    "---\n"
    "Context:\n"
    "{context_str}\n"
    "---\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


def load_index() -> VectorStoreIndex:
    """Wire up Settings and rehydrate the persisted Chroma index."""
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = Anthropic(
        model=LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=LLM_MAX_TOKENS,
    )
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store)


def query(question: str, top_k: int = SIMILARITY_TOP_K) -> dict[str, Any]:
    """Return a dict with the answer and structured source attributions."""
    index = load_index()
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=CITATION_PROMPT,
    )
    response = query_engine.query(question)
    return {
        "answer": str(response),
        "sources": [
            {
                "file": node.metadata.get("file_name"),
                "page": node.metadata.get("page_label") or node.metadata.get("source"),
                "score": round(node.score or 0.0, 4),
                "preview": (node.text[:240] + "…") if len(node.text) > 240 else node.text,
            }
            for node in response.source_nodes
        ],
    }


def _print_result(question: str, result: dict[str, Any]) -> None:
    print(f"\nQ: {question}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  • {s['file']} (p. {s['page']}, score={s['score']})")


if __name__ == "__main__":
    demo_q = (
        "What are the first-line treatments for major depressive disorder "
        "in adolescents?"
    )
    _print_result(demo_q, query(demo_q))
