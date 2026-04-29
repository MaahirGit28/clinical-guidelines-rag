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

## Repo layout

```
clinical-guidelines-rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/                          # gitignored; created on first run
│   ├── raw/                       # downloaded PDFs land here
│   └── chroma_db/                 # persistent vector store
├── src/
│   ├── __init__.py
│   ├── config.py                  # paths, model names, hyperparameters
│   ├── ingest.py                  # PDF → chunks → embeddings → Chroma
│   └── rag.py                     # query interface w/ citation prompt
└── scripts/
    └── download_guidelines.py     # fetch AACAP PDFs into data/raw/
```

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in ANTHROPIC_API_KEY

# 2. Fetch guidelines (~5–10 PDFs, ~25 MB)
python scripts/download_guidelines.py

# 3. Build the index (one-time; ~2 min on CPU, faster on GPU)
python -m src.ingest

# 4. Ask a question
python -m src.rag
```

## Three-day plan

- **Day 1** — Ingestion + indexing working end-to-end (this README)
- **Day 2** — Citation prompt iteration, chunking experiments, retrieval-quality eyeballing
- **Day 3** — Streamlit UI + RAGAS evaluation harness + writeup

## Future extensions (deliberately out of scope for the MVP)

- Hybrid retrieval (Qdrant + BM25) for exact-match clinical terms (drug names, ICD codes)
- GraphRAG over the NIMH RDoC dimensional framework
- Domain-adapted embeddings fine-tuned on clinical Q&A pairs
- Version-diff awareness (this guideline supersedes that one)

## Notes on responsible use

This system retrieves from public clinical guidelines only. It does **not** process PHI and is
not HIPAA-compliant as deployed. It is a research/portfolio prototype and is **not** for clinical
use. Per AACAP convention, practice parameters and guidelines should be considered outdated after
five years; the system surfaces source filenames so users can verify currency.
