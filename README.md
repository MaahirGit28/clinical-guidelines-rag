# Clinical Guidelines RAG

A retrieval-augmented Q&A system over the AACAP Practice Parameters — pediatric psychiatry clinical guidelines spanning 10 conditions and 17 years of publication. Built with an emphasis on **evaluation-driven iteration** and **refusal over extrapolation** when asked about conditions outside the corpus.

The interesting part isn't the stack. It's the eval set.

## What it does

Given a clinical question (e.g. *"What are first-line pharmacotherapy options for adolescent MDD?"*), the system retrieves the most relevant chunks from the indexed practice parameters and generates an answer with inline page-level citations in the form `[source: depression.pdf, p. 12]`. If asked about a condition not covered by the corpus, the system declines rather than hallucinating from adjacent material.

The project is structured around a **32-question hand-verified evaluation set**, with each retrieval/prompt change measured against it before being kept.

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

The SUD parameter intentionally uses a different rating scheme than the older parameters. The eval set spans both formats so the system has to handle the heterogeneity that real clinical reference work entails.

## Stack

| Layer | Choice |
|---|---|
| Orchestration | LlamaIndex |
| Vector store | ChromaDB (local persistent) |
| Embeddings | `BAAI/bge-m3` via HuggingFace (local) |
| Generation | `llama-3.3-70b-versatile` on Groq, via the `OpenAILike` adapter |
| PDF parsing | PyMuPDF |
| UI | Streamlit |

A note on the Groq wiring: the official `llama-index-llms-groq` package has dependency conflicts with current `llama-index-core`. Pointing the `OpenAILike` adapter at `https://api.groq.com/openai/v1` works around this cleanly without pinning either library.

## Evaluation

The eval set (`eval/questions.json`) contains 32 hand-verified questions: 3 per indexed disorder, 2 cross-PDF discrimination questions that require pulling chunks from multiple papers to answer correctly, and 1 out-of-corpus refusal probe. Each entry carries:

- A verified **primary page** in the source PDF where the answer appears
- An **expected page range** (some answers span 2–3 pages)
- **Key terms** a correct answer should include
- A **`must_not_recommend`** field listing forbidden terms (e.g. medications contraindicated for the population)
- For the refusal probe, **`refusal_indicators`** — phrases a proper decline should contain

Every page number was verified by reading the PDF text directly, not from training memory.

The runner (`eval/run_eval.py`) computes per-question scores including `source_hit`, `primary_page_hit`, `first_correct_rank`, `key_terms_hit`, `forbidden_mentioned`, and `passes_refusal`, then writes the full run as JSON to `eval/results/`.

### Iteration 1 — abstract-bleed fix (4-PDF corpus, 12 questions)

A baseline run on the original 4-PDF corpus surfaced a clear pattern: the abstract page of each journal PDF was dominating top-1 retrieval for almost any topical query about that paper, pushing actual recommendation pages down to ranks 2–6. Skipping page 1 during ingestion (`SKIP_FIRST_PAGE = True` in `src/ingest.py`) produced this delta:

| metric | baseline | no-abstract | Δ |
|---|---|---|---|
| mean first-correct rank | 2.27 | **1.82** | −0.45 |
| primary page retrieved | 10/11 | **11/11** | +1 |
| top-1 source PDF | 11/11 | 11/11 | — |
| refusal probe passed | ✓ | ✓ | — |
| rank distribution | `[1,1,1,1,1,2,2,2,2,6,6]` | `[1,1,1,1,1,1,1,1,2,5,5]` | tighter |

5 ranks improved, 0 regressed. (The rank distribution lists 11 entries because the 12th question is the refusal probe, scored separately.)

One question (`mdd_003`) appeared to drop a key term in its answer, but inspection showed the new answer was substantively *more* complete, citing actual ideation rates from the source. The "regression" was a heuristic-scorer artifact — the kind of thing that makes purely-automated eval scoring misleading without spot-checks. Worth flagging.

### Iteration 2 — corpus expansion (10 PDFs, 32 questions)

After the abstract-bleed fix landed, the corpus was expanded from 4 to 10 PDFs and the eval set from 12 to 32 questions. The refusal probe was replaced — v1 used PTSD-treatment-as-out-of-scope, but PTSD is now in the corpus, so v2 uses **nocturnal enuresis treatment**, verified to have zero mentions across all 10 PDFs.

Comparing the post-fix 4-PDF run to the 10-PDF v2 baseline:

| metric | 4-PDF (post-fix) | 10-PDF (v2) | Δ |
|---|---|---|---|
| Source PDF top-1 | 11/11 (100%) | 23/29 (79%) | −21pp |
| Primary page retrieved | 11/11 (100%) | 24/29 (83%) | −17pp |
| Mean first-correct rank | 1.82 | 2.19 | +0.37 |
| Questions with no correct hit in top-8 | 0 | 3 | +3 |
| Refusal probe passed | ✓ | ✓ | — |
| Cross-PDF discrimination (new) | n/a | 1/2 | new |

The drop is not regression. The 11 original questions still retrieve at mean rank ~1.8 in the 10-PDF run — the abstract-skip gains held. The drop is concentrated entirely in the new questions, and it has a clear, named cause: the 2.5x larger corpus introduced overlapping vocabulary across related conditions, and pure dense retrieval without metadata filtering or lexical signal can't reliably discriminate. This is what the next iteration would address (see *Known failure modes* below).

The cross-PDF discrimination case is genuinely interesting: `xpdf_002` (asking about evidence grades for SSRIs across PTSD vs MDD) retrieved chunks from both source PDFs, correctly identified that the grades differ between papers, and produced a substantively complete answer. That's the dense retriever working in its favorable regime — broad semantic similarity to multiple sources without conflicting topical pull.

## Refusal handling

Generic RAG systems will happily answer questions about conditions adjacent to but not actually in their corpus, by retrieving plausibly-related chunks and reasoning across them. For clinical use that's worse than no answer.

The refusal probe asks for a first-line pharmacological treatment for nocturnal enuresis. Enuresis is not covered by any of the 10 indexed practice parameters and no chunk in the index discusses it. The system prompt instructs the model to refuse when retrieved context does not contain explicit recommendations for the condition asked about. The probe currently passes on the v2 run with a clean *"the context does not contain information about..."* response — no extrapolation from the SSRI material in the depression or anxiety papers, no synthesis from the SUD paper's medication management discussion.

The refusal discipline holds across the 2.5x corpus expansion, which is the more meaningful test than the original 4-PDF version.

## Known failure modes

The 10-PDF run surfaced three structural failure patterns. Each is reproducible against named eval questions and each has a credible architectural fix.

**1. Cross-PDF retrieval bleed** — 3 questions: `ptsd_001`, `ocd_001`, `tic_002`.
PDFs covering related conditions share substantial vocabulary: anxiety ↔ OCD overlap on CBT, SSRI, and exposure language; tics ↔ OCD overlap on antipsychotic augmentation; MDD ↔ eating disorders overlap on SSRI use. Pure dense retrieval can't reliably discriminate when the query embedding is closer to a related-but-wrong PDF's content. The system declined or hedged when retrieval went wrong, rather than confidently hallucinating — credit to the prompt, not to the retriever. *Fix:* hybrid search (BM25 + dense), or metadata filtering by disorder topic when the query names a specific condition.

**2. Recommendation pages losing to elaboration pages** — 2 questions: `ed_002` (rank 7), `sud_003` (rank 6).
Same pattern as the abstract-bleed problem from Iteration 1. The recommendation statement on page X is shorter and lower-similarity than the elaboration discussion on page X+1. Skip-page-1 fixed the abstract case; this is the same effect surfacing on different pages. *Fix:* chunk-level metadata flagging recommendation chunks (parseable by the `[CS]/[MS]/[CG]` markers in the older format and `2C`-style markers in the newer one), then weight them in retrieval.

**3. Cross-PDF discrimination is partially broken** — `xpdf_001` failed, `xpdf_002` worked.
When the question explicitly names a specific condition ("dose of fluoxetine for OCD"), retrieval can still get pulled to the wrong PDF if another paper has higher overall similarity to the query embedding. Same root cause as #1. *Fix:* same as #1.

These were measured but deliberately deferred — the fixes are half-day-each minimum and the project benefits more from shipping than from one more retrieval iteration.

## Quickstart

### 1. Clone and set up the environment

```bash
git clone https://github.com/MaahirGit28/clinical-guidelines-rag.git
cd clinical-guidelines-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your Groq API key

```bash
cp .env.example .env
```

Then edit `.env` and paste your key after `GROQ_API_KEY=`. Get one from [console.groq.com](https://console.groq.com/keys) — the free tier is sufficient for this project.

### 3. Provide the source PDFs

The 10 AACAP Practice Parameters used in development are not redistributed in this repo (JAACAP copyright). Place the PDFs you want to index in `data/raw/`. For the eval set to score correctly out of the box, you'll need the same 10 papers — see the filenames referenced in `eval/questions.json`. Otherwise, swap in any clinical PDFs you have access to and write your own eval questions.

### 4. Build the index

```bash
python -m src.ingest
```

This parses the PDFs, skips the abstract page of each, chunks them (256 tokens, 50 overlap), embeds with BGE-M3, and persists to `data/chroma_db/`. First run downloads the BGE-M3 model (~2.3 GB) into your HuggingFace cache. Re-running is idempotent.

### 5. Launch the UI

```bash
streamlit run app.py
```

Open `http://localhost:8501`. The first query takes ~30 seconds while the embedding model loads; subsequent queries are fast.

### 6. (Optional) Run the eval

```bash
python -m eval.run_eval --tag my-experiment
```

Writes a timestamped JSON to `eval/results/`.

## Repository structure

```
clinical-guidelines-rag/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                # 10 AACAP PDFs (gitignored — JAACAP copyright)
│   └── chroma_db/          # vector store (gitignored)
├── eval/
│   ├── questions.json      # 32 hand-verified Q&A entries
│   ├── run_eval.py         # eval runner (with secret redaction)
│   └── results/            # run outputs (gitignored)
├── scripts/
│   └── download_guidelines.py
└── src/
    ├── config.py           # chunk size, top_k, model names
    ├── ingest.py           # PDF parsing + chunking + embedding
    └── rag.py              # query engine + prompt template
```

## Configuration

The settings that move retrieval quality the most live in `src/config.py`:

```python
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
TOP_K = 8
LLM_TEMPERATURE = 0.1
```

Going from `CHUNK_SIZE=512 / OVERLAP=100` to the current `256 / 50` was a meaningful retrieval improvement on the eval set — smaller chunks surfaced specific AACAP recommendation language that larger chunks diluted with surrounding prose.

## Future work

In rough priority order, with the bottleneck each one addresses:

- **Hybrid retrieval (BM25 + dense)**. The single highest-leverage change. Addresses cross-PDF bleed (failure mode #1) and cross-PDF discrimination (#3). For queries containing exact medication names or DSM terminology, lexical match should dominate; the current pure-dense setup throws that signal away. Qdrant has cleaner support for hybrid than Chroma — likely a vector store swap as part of this.
- **Metadata filtering by disorder topic**. Tag each chunk with its source disorder at ingestion time. When the query mentions a specific condition by name, restrict retrieval to that disorder's chunks. Cheaper than hybrid and addresses the same failures.
- **Recommendation-chunk weighting**. Parse the AACAP grade markers (`[CS]`, `[MS]`, `2C`, etc.) at chunk creation, tag those chunks as recommendation chunks, and boost their retrieval scores. Addresses failure mode #2.
- **RAGAS automated scoring** for faithfulness and answer relevance, layered on top of the existing heuristic scores. Useful for tracking generation quality as retrieval changes.
- **GraphRAG / Neo4j extension**. Clinical guidelines have rich entity-relation structure (condition → first-line treatment → contraindication → monitoring requirement) that a graph layer over the dense index could exploit. Originally scoped out of the build timeline; remains an interesting extension and the most ambitious of these.

## Sources & licensing

The AACAP Practice Parameters indexed during development are published in the *Journal of the American Academy of Child & Adolescent Psychiatry* and are not redistributed here. The code in this repository is mine; the indexed content is not.

## Lessons worth carrying forward

- **The eval set is the unit of progress, not single queries.** Single-query iteration is too noisy to drive decisions. The 32-question set surfaced both the abstract-bleed problem and the cross-PDF bleed problem with named, reproducible failure cases.
- **A drop in numbers can be a win in narrative.** 100% → 79% top-1 looks like regression. It's actually expansion into harder territory with a clearly identified bottleneck. Honest measurement and named failures are more valuable than clean numbers.
- **Refusal discipline is a separable axis from retrieval quality.** The refusal probe passed on every run regardless of what was happening with primary-page-hit rates. Worth tracking independently.
