"""Confidence intervals and paired significance tests for eval runs.

Stdlib only — no scipy/numpy dependency, so it won't fight with llama-index-core.

Usage:
    # Single run: CIs on each metric
    python -m eval.stats eval/results/run_20260502_v2-baseline-10pdfs.json

    # Two runs: paired comparison (the one you actually want)
    python -m eval.stats eval/results/run_A.json eval/results/run_B.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


Z95 = 1.959963985


# ── Interval estimators ───────────────────────────────────────────────

def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the textbook Wald interval (p ± z·sqrt(p(1-p)/n)) because
    Wald has poor coverage when n is small or p is near 0 or 1 — exactly our
    regime (n≈29, p≈0.8). Wilson also never produces bounds outside [0,1],
    and gives a sensible interval when successes == n (Wald gives width zero,
    which is obviously wrong).

    Returns (point_estimate, lower, upper).
    """
    if n == 0:
        return (float("nan"), 0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (p, max(0.0, center - half), min(1.0, center + half))


def bootstrap_mean(values: list[float], n_boot: int = 20000,
                   alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a mean.

    Used for mean-first-correct-rank, where the underlying distribution is
    discrete, bounded, and right-skewed — so the normal approximation behind
    a t-interval is a poor fit. Bootstrap makes no distributional assumption.

    Returns (point_estimate, lower, upper).
    """
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    point = sum(values) / n
    means = []
    for _ in range(n_boot):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (point, lo, hi)


# ── Paired comparison between two runs ────────────────────────────────

def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pairs.

    b = questions that went right -> wrong
    c = questions that went wrong -> right

    Concordant pairs carry no information about which run is better, so they
    drop out entirely. Under H0 (change had no effect), each discordant pair
    is a coin flip: b ~ Binomial(b+c, 0.5).

    The exact binomial form is used rather than the chi-square approximation
    because b+c is typically small (<10) on a 29-question eval set.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_paired_delta(deltas: list[float], n_boot: int = 20000,
                           alpha: float = 0.05, seed: int = 0
                           ) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean paired difference.

    `deltas` should be per-question (run_B_value - run_A_value). Pairing is
    what buys the statistical power here: question difficulty varies enormously
    across the eval set, and pairing removes that variance from the comparison
    entirely.

    If the resulting CI excludes 0, the change moved the metric.
    """
    return bootstrap_mean(deltas, n_boot=n_boot, alpha=alpha, seed=seed)


# ── Run-file plumbing ─────────────────────────────────────────────────

def load_run(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_per_question(run: dict) -> dict[str, dict]:
    """Pull per-question results into {question_id: {metric: value}}.

    Tolerant of a few plausible JSON shapes since run_eval.py's exact output
    schema may drift. If this raises, print the JSON's top-level keys and
    adjust the accessors below.
    """
    # Common shapes: {"results": [...]} or {"questions": [...]} or a bare list
    if isinstance(run, list):
        rows = run
    else:
        rows = run.get("results") or run.get("questions") or run.get("per_question")
    if rows is None:
        raise KeyError(
            f"Couldn't find per-question rows. Top-level keys: {list(run.keys())}"
        )
    out = {}
    for row in rows:
        qid = row.get("id") or row.get("question_id") or row.get("qid")
        out[str(qid)] = row
    return out


def _get(row: dict, *names, default=None):
    """Fetch the first present key among `names`, searching nested score dicts."""
    for n in names:
        if n in row:
            return row[n]
    for container in ("scores", "metrics", "heuristics"):
        sub = row.get(container)
        if isinstance(sub, dict):
            for n in names:
                if n in sub:
                    return sub[n]
    return default


# ── Reports ───────────────────────────────────────────────────────────

def report_single(path: str | Path) -> None:
    run = load_run(path)
    rows = extract_per_question(run)

    print(f"\n=== Single-run intervals: {Path(path).name} ===")
    print(f"    {len(rows)} questions in file\n")

    binary_metrics = [
        ("source_hit_top1", "Top-1 source PDF correct"),
        ("primary_page_hit", "Primary page retrieved"),
        ("any_expected_page_hit", "Any expected page retrieved"),
        ("source_hit", "Correct PDF anywhere in top-K"),
    ]

    print(f"{'metric':34s} {'obs':>9s}  {'point':>7s}  {'95% Wilson CI':>18s}  width")
    print("-" * 82)
    for key, label in binary_metrics:
        vals = [_get(r, key) for r in rows.values()]
        vals = [v for v in vals if isinstance(v, bool) or v in (0, 1)]
        if not vals:
            continue
        n = len(vals)
        x = sum(1 for v in vals if v)
        p, lo, hi = wilson(x, n)
        print(f"{label:34s} {x:4d}/{n:<4d}  {p*100:6.1f}%  "
              f"[{lo*100:5.1f}%, {hi*100:5.1f}%]  {(hi-lo)*100:5.1f}pp")

    # Mean rank — bootstrap, restricted to questions that actually hit
    ranks = []
    for r in rows.values():
        v = _get(r, "first_correct_rank")
        if isinstance(v, (int, float)) and v > 0:
            ranks.append(float(v))
    if ranks:
        m, lo, hi = bootstrap_mean(ranks)
        print("-" * 82)
        print(f"{'Mean first-correct rank':34s} {'n='+str(len(ranks)):>9s}  "
              f"{m:6.2f}   [{lo:5.2f}, {hi:5.2f}]   (bootstrap)")
        print(f"{'  rank distribution':34s} {sorted(int(r) for r in ranks)}")


def report_paired(path_a: str | Path, path_b: str | Path) -> None:
    a_rows = extract_per_question(load_run(path_a))
    b_rows = extract_per_question(load_run(path_b))
    shared = sorted(set(a_rows) & set(b_rows))

    print(f"\n=== Paired comparison ===")
    print(f"    A: {Path(path_a).name}")
    print(f"    B: {Path(path_b).name}")
    print(f"    {len(shared)} questions present in both "
          f"(A has {len(a_rows)}, B has {len(b_rows)})\n")
    if not shared:
        print("    No shared question IDs — nothing to compare.")
        return

    for key, label in [("source_hit_top1", "Top-1 source PDF correct"),
                       ("primary_page_hit", "Primary page retrieved")]:
        both = worse = better = neither = 0
        for qid in shared:
            av, bv = _get(a_rows[qid], key), _get(b_rows[qid], key)
            if av is None or bv is None:
                continue
            if av and bv:
                both += 1
            elif av and not bv:
                worse += 1
            elif not av and bv:
                better += 1
            else:
                neither += 1
        total = both + worse + better + neither
        if total == 0:
            continue
        p = mcnemar_exact(worse, better)
        net = better - worse
        print(f"{label}")
        print(f"   both correct: {both:2d}   both wrong: {neither:2d}   "
              f"A→B improved: {better:2d}   A→B broke: {worse:2d}")
        print(f"   net {net:+d} questions   McNemar exact p = {p:.4f}"
              f"{'  (p<0.05)' if p < 0.05 else '  (not significant)'}\n")

    # Paired rank deltas
    deltas = []
    for qid in shared:
        av = _get(a_rows[qid], "first_correct_rank")
        bv = _get(b_rows[qid], "first_correct_rank")
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)) \
                and av > 0 and bv > 0:
            deltas.append(float(bv) - float(av))
    if deltas:
        m, lo, hi = bootstrap_paired_delta(deltas)
        verdict = ("B better" if hi < 0 else
                   "A better" if lo > 0 else
                   "indistinguishable")
        print(f"Mean paired rank change (B − A), n={len(deltas)}")
        print(f"   {m:+.3f}   95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]   → {verdict}")
        print(f"   (negative = B retrieves the right page at a better rank)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", help="path to a run JSON")
    ap.add_argument("run_b", nargs="?", help="optional second run for paired comparison")
    args = ap.parse_args()

    report_single(args.run_a)
    if args.run_b:
        report_single(args.run_b)
        report_paired(args.run_a, args.run_b)


if __name__ == "__main__":
    main()
