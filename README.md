# Clinical Guidelines RAG

A retrieval-augmented Q&A system over the AACAP Practice Parameters — pediatric psychiatry clinical guidelines spanning 10 conditions and 17 years of publication. Built with an emphasis on **evaluation-driven iteration** and **refusal over extrapolation** when asked about conditions outside the corpus.

The interesting part isn't the stack. It's the eval set.

## What it does

Given a clinical question (e.g. *"What are first-line pharmacotherapy options for adolescent MDD?"*), the system retrieves the most relevant chunks from the indexed practice parameters and generates an answer with inline page-level citations in the form `[source: depression.pdf, p. 12]`. If asked about a condition not covered by the corpus, the system declines rather than hallucinating from adjacent material.

The project is structured around a **32-question hand-verified evaluation set**, with each retrieval change measured against it before being kept.

## Corpus

10 AACAP Practice Parameters published in the *Journal of the American Academy of Child & Adolescent Psychiatry* between 2007 and 2024:

| Disorder | Year | Pages | Grading format |
|---|---|---|---|
| Major depressive disorder | 2022 | 24 | `[CS]/[MS]/[CG]/[OP]/[NE]` |
| Anxiety disorders | 2020 | 18 | `[CS]/[MS]/[CG]/[OP]/[NE]` |
| Bipolar disorder | 2007 | 19 | `[CS]/[MS]/[CG]/[OP]/[NE]` |
| Autism spectrum disorder | 2014 | 23 | `[CS]/[MS]/[CG]/[OP]/[NE]` |
| PTSD | 2010 | 17 | `[CS]/[MS]/[CG]/[OP]/[NE]` |
| OCD | 2012 | 16 | `[CS]/[MS]/[CG]/[OP]/[NE]` |
| Schizophrenia | 2013 | 15 | `[CS]/[MS]/[CG]/[OP]/[NE]` |
| Tic disorders | 2013 | 19 | `[CS]/[MS]/[CG]/[OP]/[NE]` |
| Eating disorders | 2015 | 14 | `[CS]/[MS]/[CG]/[OP]/[NE]` |
| Substance use disorders | 2024 | 34 | AHRQ/Brown systematic review (e.g. `2C`) |

The 2024 SUD parameter uses a different rating scheme than the older parameters. The eval set spans both formats so the system has to handle the heterogeneity that real clinical reference work entails.

## Stack

| Layer | Choice |
|---|---|
| Orchestration | LlamaIndex (`llama-index-core`) |
| Vector store | Qdrant Cloud (managed) |
| Embeddings | `BAAI/bge-m3` via HuggingFace (local, 1024-dim, cosine) |
| Generation | `llama-3.3-70b-versatile` on Groq, via the `OpenAILike` adapter |
| PDF parsing | PyMuPDF |
| UI | Streamlit |

Three deliberate choices worth calling out:

**Qdrant Cloud over local ChromaDB.** The project started on a local persistent Chroma client, which works fine in development but does not survive a stateless deployment — the container filesystem is ephemeral, so the index disappears on every restart. Migrating to managed Qdrant made the vector store external to the app process. The migration touched four files (`src/config.py`, `src/rag.py`, `src/ingest.py`, `requirements.txt`).

**`OpenAILike` adapter for Groq, not the dedicated wrapper.** The official `llama-index-llms-groq` package has dependency conflicts with current `llama-index-core`. Groq's API is OpenAI-compatible, so pointing the generic `OpenAILike` adapter at `https://api.groq.com/openai/v1` sidesteps the conflict without pinning either library to an older version.

**`llama-index-core` plus explicit subpackages, not the `llama-index` meta-package.** The meta-package pulls in agent, program, and question-generation subpackages that this project never imports. `requirements.txt` is hand-curated to direct dependencies with pinned versions rather than a `pip freeze` dump.

## Evaluation

The eval set (`eval/questions.json`) contains 32 hand-verified questions: roughly 3 per indexed disorder, 2 cross-PDF discrimination questions that require pulling chunks from multiple papers, and 1 out-of-corpus refusal probe. Each entry carries a verified **primary page**, an **expected page range**, **key terms** a correct answer should include, a **`must_not_recommend`** list of forbidden terms, and (for the refusal probe) **`refusal_indicators`**.

Every page number was verified by reading the PDF text directly, not from model output.

Aggregate metrics below are reported over the **29 single-source questions**. The refusal probe and the 2 cross-PDF questions are scored separately, since "was the expected PDF retrieved at top-1" is ill-defined for a question that requires two papers at once.

### Iteration 1 — abstract-bleed fix (4-PDF corpus, 12 questions)

A baseline run on the original 4-PDF corpus surfaced a clear pattern: the abstract page of each journal PDF was dominating top-1 retrieval for almost any topical query about that paper, pushing actual recommendation pages down to ranks 2–6. Skipping page 1 during ingestion (`SKIP_FIRST_PAGE = True` in `src/ingest.py`) produced:

| metric | baseline | no-abstract |
|---|---|---|
| mean first-correct rank | 2.27 | **1.82** |
| primary page retrieved | 10/11 | **11/11** |
| top-1 source PDF | 11/11 | 11/11 |
| refusal probe passed | ✓ | ✓ |
| rank distribution | `[1,1,1,1,1,2,2,2,2,6,6]` | `[1,1,1,1,1,1,1,1,2,5,5]` |

Because both runs cover the same questions, the right test is paired rather than a comparison of independent intervals. **Mean paired first-correct rank improved by 0.46 (95% bootstrap CI: 0.18–0.73)**, excluding zero cleanly.

Notably, the binary metrics detect nothing here — top-1 source was 11/11 in both runs, McNemar p = 1.0. Rank distinguishes position 1 from position 8 where hit-rate collapses both to "hit," so it is far more sensitive at this sample size. This is the metric to watch when evaluating future retrieval changes.

One question (`mdd_003`) appeared to drop a key term in its answer, but inspection showed the new answer was substantively *more* complete, citing actual ideation rates from the source. The apparent regression was a heuristic-scorer artifact — worth flagging as a reason automated eval scores need spot-checks.

### Iteration 2 — corpus expansion (10 PDFs, 32 questions)

The corpus was expanded from 4 to 10 PDFs and the eval set from 12 to 32 questions. The refusal probe was replaced: v1 used PTSD-treatment-as-out-of-scope, but PTSD is now *in* the corpus, so v2 uses **nocturnal enuresis**, verified to have zero mentions across all 10 PDFs.

| metric | 4-PDF (post-fix) | 10-PDF (v2) | 95% CI (10-PDF) |
|---|---|---|---|
| Top-1 source PDF | 11/11 (100%) | 23/29 (79%) | [62%, 90%] |
| Primary page retrieved | 11/11 (100%) | 24/29 (83%) | [65%, 92%] |
| Correct PDF anywhere in top-8 | 11/11 (100%) | 30/31 (97%) | [84%, 99%] |
| Mean first-correct rank | 1.82 | 2.19 | [1.5, 2.9] |
| Refusal probe passed | ✓ | ✓ | — |
| Cross-PDF discrimination | n/a | 1 of 2 | — |

Intervals are Wilson score intervals, which behave better than the normal approximation at small n and proportions near 1.

**The drop is expansion, not regression.** The 11 original questions still retrieve at mean rank ~1.8 in the 10-PDF run — the abstract-skip gains held. The drop is concentrated entirely in the new questions, with a clear named cause: a 2.5x larger corpus introduced overlapping vocabulary across related conditions, and pure dense retrieval without lexical signal or metadata filtering cannot reliably discriminate.

Worth stating plainly: at n=29, roughly six questions must flip wrong→right with none regressing before a binary hit-rate change reaches p < 0.05. This eval set can detect large effects, not subtle ones. That is a reason to prefer rank-based metrics and to report effect sizes with intervals rather than significance verdicts.

The cross-PDF case that worked is instructive: `xpdf_002` (evidence grades for SSRIs across PTSD vs MDD) retrieved from both source PDFs, correctly identified that the grades differ, and produced a complete answer. That is dense retrieval in its favorable regime — broad semantic similarity to multiple sources without conflicting topical pull.

## Refusal handling

Generic RAG systems will happily answer questions about conditions adjacent to but not in their corpus, by retrieving plausibly-related chunks and reasoning across them. For clinical use that is worse than no answer.

The refusal probe asks for a first-line pharmacological treatment for nocturnal enuresis, which no indexed parameter covers. The system prompt instructs the model to refuse when retrieved context does not contain explicit recommendations for the condition asked about. The probe passes on the v2 run with a clean *"the context does not contain information about..."* response — no extrapolation from the SSRI material in the depression or anxiety papers, no synthesis from the SUD paper's medication management discussion.

The refusal discipline held across the 2.5x corpus expansion, which is a more meaningful test than the original 4-PDF version.

## Known failure modes

Three structural patterns surfaced in the 10-PDF run. Each is reproducible against named eval questions and each has a credible architectural fix. None are implemented.

**1. Cross-PDF retrieval bleed** — `ptsd_001`, `ocd_001`, `tic_002`.
Related conditions share substantial vocabulary: anxiety ↔ OCD on CBT, SSRI, and exposure language; tics ↔ OCD on antipsychotic augmentation; MDD ↔ eating disorders on SSRI use. Pure dense retrieval cannot discriminate when the query embedding sits closer to a related-but-wrong paper. The system declined or hedged when retrieval went wrong rather than confidently hallucinating — credit to the prompt, not the retriever. *Fix:* hybrid search, or metadata filtering by disorder when the query names a condition.

**2. Recommendation pages losing to elaboration pages** — `ed_002` (rank 7), `sud_003` (rank 6).
Same shape as the abstract-bleed problem. The recommendation statement on page X is shorter and lower-similarity than the elaboration on page X+1. Skip-page-1 fixed the abstract case; this is the same effect on different pages. *Fix:* flag recommendation chunks at ingestion via the AACAP grade markers and weight them in retrieval.

**3. Cross-PDF discrimination partially broken** — `xpdf_001` failed, `xpdf_002` worked.
When the question names a condition explicitly ("dose of fluoxetine for OCD"), retrieval can still be pulled to the wrong paper if it has higher overall similarity to the query embedding. Same root cause as #1.

## Quickstart

### 1. Clone and set up the environment

```bash
git clone https://github.com/MaahirGit28/clinical-guidelines-rag.git
cd clinical-guidelines-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
```

Then edit `.env` with three values:

```
GROQ_API_KEY=gsk_...
QDRANT_URL=https://your-cluster-id.region.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=...
```

No quotes around values and no spaces around `=` — python-dotenv treats quote characters as part of the value. Get a Groq key at [console.groq.com](https://console.groq.com/keys) and a Qdrant cluster at [cloud.qdrant.io](https://cloud.qdrant.io); both have free tiers sufficient for this project.

`src/config.py` raises on startup if any of the three is missing.

### 3. Provide the source PDFs

The 10 AACAP Practice Parameters are not redistributed here (JAACAP copyright). Place the PDFs you want to index in `data/raw/`. For the eval set to score correctly, you need the same 10 papers — filenames are referenced in `eval/questions.json`. Otherwise swap in your own clinical PDFs and write your own eval questions.

### 4. Build the index

```bash
python -m src.ingest
```

Parses the PDFs, skips the abstract page of each, tags page labels, chunks at 256 tokens with 50 overlap, embeds with BGE-M3, and writes to the Qdrant collection. The script wipes and recreates the collection on each run, so re-ingesting is safe and leaves no stale chunks. First run downloads BGE-M3 (~2.3 GB) into your HuggingFace cache.

### 5. Launch the UI

```bash
streamlit run app.py
```

Open `http://localhost:8501`. First query takes ~30s while the embedding model loads; subsequent queries are fast.

### 6. (Optional) Run the eval

```bash
python -m eval.run_eval --tag my-experiment
python -m eval.stats eval/results/run_<timestamp>_my-experiment.json
```

## Repository structure

```
clinical-guidelines-rag/
├── app.py                  # Streamlit UI
├── requirements.txt        # hand-curated, pinned direct dependencies
├── README.md
├── .env.example
├── docs/
├── data/
│   └── raw/                # 10 AACAP PDFs (gitignored — JAACAP copyright)
├── eval/
│   ├── README.md           # eval methodology, scoring, iteration loop
│   ├── questions.json      # 32 hand-verified Q&A entries
│   ├── run_eval.py         # runner, with secret redaction on the config snapshot
│   ├── stats.py            # Wilson CIs, bootstrap, paired McNemar
│   └── results/            # run outputs (gitignored)
├── scripts/
│   └── download_guidelines.py
└── src/
    ├── config.py           # chunking, retrieval, model, and Qdrant settings
    ├── ingest.py           # PDF parsing + chunking + embedding + upsert
    └── rag.py              # query engine + prompt template
```

## Configuration

```python
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
TOP_K = 8
LLM_TEMPERATURE = 0.1
EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama-3.3-70b-versatile"
COLLECTION_NAME = "Clinical_RAG1"
```

Plus `SKIP_FIRST_PAGE = True` in `src/ingest.py`.

Moving from `CHUNK_SIZE=512 / OVERLAP=100` to `256 / 50` was a meaningful retrieval improvement — smaller chunks surfaced specific AACAP recommendation language that larger chunks diluted with surrounding prose.

## Future work

In priority order, with the bottleneck each addresses:

- **Hybrid retrieval (lexical BM25 + semantic dense).** Highest leverage. Addresses failure modes 1 and 3. Queries naming specific medications or DSM terminology should be biased toward chunks literally containing those tokens; the current pure-dense setup discards that signal. Qdrant supports sparse vectors natively, so this no longer requires a vector store change.
- **Metadata filtering by disorder.** Tag each chunk with its source disorder at ingestion; restrict retrieval when the query names a condition. Cheaper than hybrid, addresses the same failures.
- **Recommendation-chunk weighting.** Parse AACAP grade markers (`[CS]`, `[MS]`, `2C`) at chunk creation and boost those chunks. Addresses failure mode 2.
- **Reference-list filtering.** Drop or downweight pages where a high fraction of tokens match citation patterns. Journal back-matter still competes with body content for some screening questions.
- **RAGAS faithfulness scoring**, layered on the existing heuristic scores, to catch answers that ignore retrieval and draw on model priors.
- **GraphRAG extension.** Clinical guidelines have rich entity-relation structure (condition → first-line treatment → contraindication → monitoring) that a graph layer over the dense index could exploit.

## Sources & licensing

The AACAP Practice Parameters indexed during development are published in the *Journal of the American Academy of Child & Adolescent Psychiatry* and are not redistributed here. The code in this repository is mine; the indexed content is not.
