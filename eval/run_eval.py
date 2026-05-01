"""Run the eval question set against the RAG system, score results, save run.

Usage (from repo root):
    python -m eval.run_eval
    python -m eval.run_eval --questions eval/questions.json
    python -m eval.run_eval --tag baseline-256-50  # label this run

What it does:
    1. Loads questions from JSON.
    2. Builds the RAG query engine using src/config.py + src/rag.py.
    3. For each question, runs the query and captures:
         - generated answer
         - retrieved source nodes (file + page + score)
    4. Scores each result against the expected source/page/key_terms.
    5. Prints a per-question report and summary table.
    6. Saves the full run to eval/results/run_<timestamp>_<tag>.json
       so you can diff between iterations.

Wiring assumption:
    src/rag.py exposes a `build_query_engine()` function that returns a
    LlamaIndex query engine ready to call with .query(question_str). If
    your function is named differently, edit the import on line marked
    [WIRE-UP] below — that's the only place this script is coupled to
    your code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import textwrap
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

# ---- repo path setup so this script works whether run as module or directly ----
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# [WIRE-UP] Imports your existing src/rag.py:load_query_engine().
# If you rename that function later, update the import here.
try:
    from src.rag import load_query_engine
except ImportError as e:
    print("=" * 72)
    print("Could not import load_query_engine from src.rag.")
    print(f"Original error: {e}")
    print("=" * 72)
    sys.exit(1)

try:
    from src import config as cfg
except ImportError:
    cfg = None  # config snapshot is best-effort


# ---------- data structures ----------

@dataclass
class RetrievedSource:
    file_name: str
    page_label: str | None
    score: float | None
    text_preview: str  # first ~200 chars of the chunk

@dataclass
class QuestionResult:
    id: str
    topic: str
    category: str
    question: str
    expected: dict
    generated_answer: str
    retrieved: list[RetrievedSource]
    scores: dict
    elapsed_s: float
    error: str | None = None


# ---------- scoring ----------

def score_result(q: dict, answer: str, retrieved: list[RetrievedSource]) -> dict:
    """Compute simple heuristic scores. Manual review still primary."""
    s: dict[str, Any] = {}
    answer_lc = (answer or "").lower()

    # Refusal questions are scored differently
    if q["category"] == "refusal":
        indicators = q.get("refusal_indicators", [])
        forbidden = q.get("must_not_recommend", [])
        s["refusal_signal_hit"] = any(ind.lower() in answer_lc for ind in indicators)
        s["forbidden_drug_mentioned"] = [d for d in forbidden if d.lower() in answer_lc]
        s["passes_refusal"] = s["refusal_signal_hit"] and not s["forbidden_drug_mentioned"]
        return s

    expected = q["expected_source"]
    expected_file = expected["file_name"]
    primary_page = str(expected["primary_page"])
    expected_pages = {str(p) for p in expected["expected_pages"]}

    retrieved_files = [r.file_name for r in retrieved]
    retrieved_pages = [(r.file_name, str(r.page_label) if r.page_label is not None else "") for r in retrieved]

    # Source PDF hit anywhere in retrieved set
    s["source_hit"] = expected_file in retrieved_files
    # Source PDF in top-1
    s["source_hit_top1"] = bool(retrieved) and retrieved[0].file_name == expected_file
    # Primary page hit (correct file AND correct page)
    s["primary_page_hit"] = (expected_file, primary_page) in retrieved_pages
    # Any expected page hit
    s["any_expected_page_hit"] = any(
        f == expected_file and p in expected_pages for f, p in retrieved_pages
    )
    # Rank of first correct (file, page) match
    rank = None
    for i, (f, p) in enumerate(retrieved_pages, start=1):
        if f == expected_file and p in expected_pages:
            rank = i
            break
    s["first_correct_rank"] = rank

    # Key term coverage in generated answer
    key_terms = q.get("key_terms", [])
    if key_terms:
        hits = [t for t in key_terms if t.lower() in answer_lc]
        s["key_terms_total"] = len(key_terms)
        s["key_terms_hit"] = len(hits)
        s["key_terms_missing"] = [t for t in key_terms if t.lower() not in answer_lc]

    # Forbidden term check (e.g., MDD Q1 should not RECOMMEND paroxetine)
    forbidden = q.get("must_not_recommend", [])
    if forbidden:
        s["forbidden_mentioned"] = [t for t in forbidden if t.lower() in answer_lc]

    return s


# ---------- runner ----------

def extract_sources(response) -> list[RetrievedSource]:
    """Pull source nodes from a LlamaIndex Response object."""
    out = []
    for node in getattr(response, "source_nodes", []) or []:
        meta = getattr(node, "metadata", None) or getattr(node.node, "metadata", {}) or {}
        text = getattr(node, "text", None) or getattr(node.node, "get_content", lambda: "")()
        out.append(RetrievedSource(
            file_name=meta.get("file_name", "?"),
            page_label=meta.get("page_label"),
            score=getattr(node, "score", None),
            text_preview=(text or "")[:200].replace("\n", " "),
        ))
    return out


def run(questions_path: Path, tag: str | None) -> dict:
    print(f"Loading questions from {questions_path}")
    data = json.loads(questions_path.read_text())
    questions = data["questions"]
    print(f"  {len(questions)} questions loaded")

    print("Building query engine...")
    qe = load_query_engine()
    print("  ready\n")

    results: list[QuestionResult] = []
    for q in questions:
        print(f"[{q['id']}] {q['question'][:80]}{'...' if len(q['question']) > 80 else ''}")
        t0 = time.time()
        err = None
        answer = ""
        retrieved: list[RetrievedSource] = []
        try:
            resp = qe.query(q["question"])
            answer = getattr(resp, "response", None) or str(resp)
            retrieved = extract_sources(resp)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0
        scores = score_result(q, answer, retrieved) if not err else {}
        results.append(QuestionResult(
            id=q["id"],
            topic=q["topic"],
            category=q["category"],
            question=q["question"],
            expected=q.get("expected_source") or {"out_of_scope": True},
            generated_answer=answer,
            retrieved=retrieved,
            scores=scores,
            elapsed_s=round(elapsed, 2),
            error=err,
        ))
        # one-line per-question summary
        if err:
            print(f"  ERROR: {err}")
        elif q["category"] == "refusal":
            print(f"  refusal_pass={scores.get('passes_refusal')} "
                  f"forbidden_hit={scores.get('forbidden_drug_mentioned')}")
        else:
            print(f"  source_hit={scores.get('source_hit')} "
                  f"primary_page={scores.get('primary_page_hit')} "
                  f"any_page={scores.get('any_expected_page_hit')} "
                  f"first_correct_rank={scores.get('first_correct_rank')}")
        print()

    # ---------- summary table ----------
    print_summary(results)

    # ---------- persist ----------
    config_snapshot = {}
    if cfg:
        SECRET_HINTS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "API")
        for k in dir(cfg):
            if k.startswith("_"):
                continue
            if any(hint in k.upper() for hint in SECRET_HINTS):
                continue  # never serialize anything that smells like a credential
            v = getattr(cfg, k)
            if isinstance(v, (int, float, str, bool)):
                config_snapshot[k] = v

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"run_{timestamp}{'_' + tag if tag else ''}.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps({
        "timestamp": timestamp,
        "tag": tag,
        "questions_path": str(questions_path),
        "config_snapshot": config_snapshot,
        "results": [_serialize(r) for r in results],
    }, indent=2))
    print(f"\nSaved run to {out_path}")
    return {"results": results, "out_path": str(out_path)}


def _serialize(r: QuestionResult) -> dict:
    d = asdict(r)
    d["retrieved"] = [asdict(s) for s in r.retrieved]
    return d


def print_summary(results: list[QuestionResult]) -> None:
    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)

    n = len(results)
    rec_questions = [r for r in results if r.category != "refusal" and not r.error]
    refusal_questions = [r for r in results if r.category == "refusal" and not r.error]

    if rec_questions:
        n_rec = len(rec_questions)
        source_hits = sum(1 for r in rec_questions if r.scores.get("source_hit"))
        primary_hits = sum(1 for r in rec_questions if r.scores.get("primary_page_hit"))
        any_page_hits = sum(1 for r in rec_questions if r.scores.get("any_expected_page_hit"))
        top1 = sum(1 for r in rec_questions if r.scores.get("source_hit_top1"))
        # mean rank of first correct
        ranks = [r.scores.get("first_correct_rank") for r in rec_questions
                 if r.scores.get("first_correct_rank") is not None]
        mean_rank = sum(ranks) / len(ranks) if ranks else float("nan")
        # key term coverage
        kt_hit = sum(r.scores.get("key_terms_hit", 0) for r in rec_questions)
        kt_total = sum(r.scores.get("key_terms_total", 0) for r in rec_questions)

        print(f"Recommendation questions: {n_rec}")
        print(f"  source PDF in retrieved set:    {source_hits}/{n_rec}  ({source_hits/n_rec:.0%})")
        print(f"  source PDF as top-1:            {top1}/{n_rec}  ({top1/n_rec:.0%})")
        print(f"  primary page retrieved:         {primary_hits}/{n_rec}  ({primary_hits/n_rec:.0%})")
        print(f"  any expected page retrieved:    {any_page_hits}/{n_rec}  ({any_page_hits/n_rec:.0%})")
        print(f"  mean rank of first correct hit: {mean_rank:.2f}")
        print(f"  key terms in generated answer:  {kt_hit}/{kt_total}  ({(kt_hit/kt_total) if kt_total else 0:.0%})")

    if refusal_questions:
        n_ref = len(refusal_questions)
        passes = sum(1 for r in refusal_questions if r.scores.get("passes_refusal"))
        print(f"\nRefusal questions: {n_ref}")
        print(f"  correctly declined: {passes}/{n_ref}")
        for r in refusal_questions:
            if not r.scores.get("passes_refusal"):
                print(f"    FAIL [{r.id}]: refusal_signal={r.scores.get('refusal_signal_hit')} "
                      f"forbidden={r.scores.get('forbidden_drug_mentioned')}")

    errors = [r for r in results if r.error]
    if errors:
        print(f"\nErrors: {len(errors)}/{n}")
        for r in errors:
            print(f"  [{r.id}] {r.error}")

    # Per-question one-liner with answer preview
    print("\n" + "-" * 88)
    print("PER-QUESTION DETAIL")
    print("-" * 88)
    for r in results:
        print(f"\n[{r.id}] ({r.topic}/{r.category})  {r.elapsed_s}s")
        print(f"  Q: {r.question}")
        if r.error:
            print(f"  ERROR: {r.error}")
            continue
        ans_preview = textwrap.shorten(r.generated_answer, width=300, placeholder="...")
        print(f"  A: {ans_preview}")
        if r.category == "refusal":
            print(f"  refusal_pass={r.scores.get('passes_refusal')}")
        else:
            exp_file = r.expected.get("file_name", "?")
            exp_page = r.expected.get("primary_page", "?")
            print(f"  expected: {exp_file} p{exp_page}")
            print(f"  retrieved:")
            for i, s in enumerate(r.retrieved[:5], start=1):
                marker = "  "
                if s.file_name == exp_file:
                    if str(s.page_label) == str(exp_page):
                        marker = "**"
                    elif str(s.page_label) in {str(p) for p in r.expected.get("expected_pages", [])}:
                        marker = "* "
                score = f"{s.score:.3f}" if isinstance(s.score, (int, float)) else "—"
                print(f"   {marker}{i}. {s.file_name} p{s.page_label}  score={score}")
            kt_miss = r.scores.get("key_terms_missing") or []
            if kt_miss:
                print(f"  key_terms_missing: {kt_miss}")
            forb = r.scores.get("forbidden_mentioned")
            if forb:
                print(f"  WARNING forbidden_mentioned: {forb}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default=str(Path(__file__).parent / "questions.json"))
    p.add_argument("--tag", default=None, help="Short label to attach to this run's output file")
    args = p.parse_args()
    run(Path(args.questions), args.tag)


if __name__ == "__main__":
    main()
