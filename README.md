# Clinical Guidelines RAG: Child & Adolescent Mental Health

A retrieval-augmented Q&A system over AACAP clinical practice guidelines, built to demonstrate
production-quality RAG patterns for clinical decision support.

## Why this exists

Clinical guidelines are dense, versioned, and require trustworthy attribution — making them an
ideal RAG use case. This project ingests AACAP clinical practice guidelines and answers questions
against them with explicit page-level citations, refusing to answer when context is insufficient.

## Stack

| Layer         | Choice              | Why                                                                  |
| ------------- | ------------------- | -------------------------------------------------------------------- |
| Orchestration | LlamaIndex          | Strongest retrieval primitives among 2026 RAG frameworks             |
| Vector store  | Chroma (local)      | Zero-ops; sufficient for a ~10–50 PDF corpus                         |
| Embeddings    | BGE-M3              | Dense + sparse in one model; strong on technical English             |
| LLM           | Claude Sonnet 4.6   | 200k context, strong instruction-following for citation discipline   |
| PDF parsing   | PyMuPDF             | Fast, reliable; preserves page-level metadata for citation           |
| UI (optional) | Streamlit           | Demo-able in <100 lines                                              |
| Eval          | RAGAS               | Faithfulness + answer-relevancy as quantitative gates                |

## Notes on responsible use

This system retrieves from public clinical guidelines only. It does **not** process PHI and is
not HIPAA-compliant as deployed. It is a research/portfolio prototype and is **not** for clinical
use. Per AACAP convention, practice parameters and guidelines should be considered outdated after
five years; the system surfaces source filenames so users can verify currency.
