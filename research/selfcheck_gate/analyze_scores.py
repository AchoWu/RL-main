"""Aggregate stats over scores_full.jsonl.

Reports:
  - n_rollouts, correct rate, TOOLONG counts
  - s_final distribution by correct label (both long and short target)
  - separation |mean(s_final|wrong) - mean(s_final|correct)|
  - a rough AUC of "s_final < tau predicts wrong" via score-rank
  - s_k distribution across cut positions (early vs late) by correct label
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def q(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    idx = int(round((len(xs) - 1) * p))
    return xs[idx]


def rank_auc(pos_scores, neg_scores):
    """AUC where NEG (wrong) is the positive class of interest (low s → wrong).

    Equivalent to AUC of a classifier that predicts wrong when s is small.
    """
    if not pos_scores or not neg_scores:
        return None
    # Probability that a random wrong-rollout has LOWER s_final than a random correct one
    wins = 0
    ties = 0
    for w in neg_scores:
        for c in pos_scores:
            if w < c:
                wins += 1
            elif w == c:
                ties += 1
    total = len(pos_scores) * len(neg_scores)
    return (wins + 0.5 * ties) / total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path,
                        default=Path("research/selfcheck_gate/artifacts/scores_full.jsonl"))
    args = parser.parse_args()

    rows: list[dict] = []
    with args.scores.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    n = len(rows)
    correct_rows = [r for r in rows if r["correct"]]
    wrong_rows = [r for r in rows if not r["correct"]]
    n_toolong_final_long = sum(1 for r in rows if r["s_final_long"] is None)
    n_toolong_final_short = sum(1 for r in rows if r["s_final_short"] is None)

    print(f"n_rollouts = {n}")
    print(f"correct    = {len(correct_rows)} ({len(correct_rows)/n:.1%})")
    print(f"wrong      = {len(wrong_rows)} ({len(wrong_rows)/n:.1%})")
    print(f"s_final_long  TOOLONG: {n_toolong_final_long}")
    print(f"s_final_short TOOLONG: {n_toolong_final_short}")
    print()

    # s_final distribution by correct label
    for target in ("long", "short"):
        key = f"s_final_{target}"
        c = [r[key] for r in correct_rows if r[key] is not None]
        w = [r[key] for r in wrong_rows if r[key] is not None]
        if not c or not w:
            continue
        print(f"=== s_final_{target} ===")
        print(f"  correct n={len(c)}: mean={statistics.mean(c):+.3f} "
              f"std={statistics.pstdev(c):.3f} "
              f"[p10={q(c,0.1):+.3f} p50={q(c,0.5):+.3f} p90={q(c,0.9):+.3f}]")
        print(f"  wrong   n={len(w)}: mean={statistics.mean(w):+.3f} "
              f"std={statistics.pstdev(w):.3f} "
              f"[p10={q(w,0.1):+.3f} p50={q(w,0.5):+.3f} p90={q(w,0.9):+.3f}]")
        sep = statistics.mean(c) - statistics.mean(w)
        print(f"  separation (mean_correct - mean_wrong): {sep:+.3f}")
        auc = rank_auc(c, w)
        print(f"  AUC(low s_final → wrong)              : {auc:.3f}")
        print()

    # s_k distribution across cut positions
    print("=== s_k across cut positions (k = paragraph index) ===")
    # Bucket cuts by k range: early (k=1-3), mid (k=4-9), late (k>=10)
    buckets = [("early(k=1-3)", 1, 3), ("mid(k=4-9)", 4, 9), ("late(k>=10)", 10, 999)]
    for target in ("long", "short"):
        print(f"-- target={target} --")
        for name, lo, hi in buckets:
            c_vals, w_vals = [], []
            for r in rows:
                for cut in r["cuts"]:
                    if not (lo <= cut["k"] <= hi):
                        continue
                    v = cut.get(f"s_k_{target}")
                    if v is None:
                        continue
                    (c_vals if r["correct"] else w_vals).append(v)
            if not c_vals or not w_vals:
                continue
            sep = statistics.mean(c_vals) - statistics.mean(w_vals)
            print(f"  {name:12s} correct n={len(c_vals):4d} mean={statistics.mean(c_vals):+.3f}  "
                  f"| wrong n={len(w_vals):4d} mean={statistics.mean(w_vals):+.3f}  "
                  f"sep={sep:+.3f}")
    print()

    # Bottom-10 s_k across all cuts (candidate "clearly dead" segments)
    all_cuts = []
    for r in rows:
        for cut in r["cuts"]:
            v = cut.get("s_k_long")
            if v is None:
                continue
            all_cuts.append({
                "index": r["index"], "k": cut["k"], "correct": r["correct"],
                "gt": r["ground_truth"], "s_k_long": v,
                "s_k_short": cut.get("s_k_short"),
            })
    all_cuts.sort(key=lambda x: x["s_k_long"])
    print("=== Bottom-10 s_k_long ===")
    for c in all_cuts[:10]:
        print(f"  rollout={c['index']:3d} k={c['k']:3d} correct={c['correct']!s:5s} "
              f"gt={c['gt']:>6s} s_k_long={c['s_k_long']:+.3f} s_k_short={c['s_k_short']}")
    print()
    print("=== Top-10 s_k_long (most confident on-track) ===")
    for c in all_cuts[-10:]:
        print(f"  rollout={c['index']:3d} k={c['k']:3d} correct={c['correct']!s:5s} "
              f"gt={c['gt']:>6s} s_k_long={c['s_k_long']:+.3f} s_k_short={c['s_k_short']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
