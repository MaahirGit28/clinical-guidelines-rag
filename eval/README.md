# Eval

Hand-written evaluation set for the Clinical Guidelines RAG. 12 questions across the four AACAP guidelines in the corpus, plus one out-of-scope refusal question.

## Files

- `questions.json` — the eval set. 12 questions, each with verified expected source PDF + page numbers and key terms a correct answer should contain.
- `run_eval.py` — runs every question through the RAG, scores results, prints a summary, and saves the run to `eval/results/run_<timestamp>.json` so iterations can be diffed.
- `results/` — created on first run; one JSON file per run.

## Wire-up

`run_eval.py` imports `load_query_engine` from your existing `src/rag.py`. No code changes needed in `src/rag.py` — it already exposes the right function. If you rename `load_query_engine` later, update the import at the top of `run_eval.py` (search for `[WIRE-UP]`).

## Running

```bash
# from repo root
python -m eval.run_eval

# tag a run so you can identify it later
python -m eval.run_eval --tag baseline-256-50
python -m eval.run_eval --tag chunk-384-overlap-75
```

Each run writes `eval/results/run_<timestamp>_<tag>.json`. The file contains the full question, generated answer, retrieved sources, scores, and a config snapshot — enough to fully reconstruct what changed between runs.

## What gets scored

For each non-refusal question:

- **`source_hit`** — was the expected PDF retrieved at all (in top-k)?
- **`source_hit_top1`** — was the expected PDF the top-1 result?
- **`primary_page_hit`** — was the *exact* expected page retrieved?
- **`any_expected_page_hit`** — was *any* of the expected page range retrieved?
- **`first_correct_rank`** — what was the rank of the first correct (file, page) hit?
- **`key_terms_hit / key_terms_total`** — did the generated answer mention the key terms (e.g., `fluoxetine`, `SSRI`)?
- **`forbidden_mentioned`** — did the answer recommend something it shouldn't (e.g., paroxetine for MDD)?

For the refusal question:

- **`refusal_signal_hit`** — did the answer contain phrases like "does not cover" / "context does not"?
- **`forbidden_drug_mentioned`** — did the answer name a PTSD drug despite no PTSD recommendation existing in the corpus?
- **`passes_refusal`** — both of the above check out.

These are heuristic scores. They tell you which questions to pay attention to; manual review is still the source of truth for whether a generated answer is correct.

## Iteration loop for Day 2

1. `python -m eval.run_eval --tag baseline` — establishes the starting point.
2. Look at the per-question detail and the summary table. Identify the worst-performing questions.
3. Form one hypothesis at a time (e.g., "smaller chunks will help precision on the SCARED-naming question") and change only one thing in `src/config.py`.
4. Re-ingest if you changed chunking; re-run eval with a descriptive tag.
5. Diff the two runs by `id`. The questions with the biggest score deltas are where the change had real effect — that's the signal you're optimizing on.

Don't tune the prompt or chunking against a single bad query. The whole point of this set is that single-query iteration is too noisy.
