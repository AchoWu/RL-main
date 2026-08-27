"""Compute pass@k (and avg@n) from AIME eval jsonl written by vllm_opd_aime.py.

Each problem contributes `--num-generation` consecutive rows, so correctness is
graded once per row and then grouped. pass@k uses the unbiased Codex estimator
(1 - C(n-c, k) / C(n, k)) rather than just looking at the first k samples, so a
single 32-sample run gives a low-variance estimate for every k <= 32.
"""

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def _is_correct(row):
    """Grade one response. Mirrors eval_metric() in vllm_opd_aime.py exactly."""
    parsed = parse(
        row["response"],
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    return float(verify(parsed, row["answer"]))


def pass_at_k(n, c, k):
    """Unbiased estimator of the probability that k samples contain a correct one."""
    if n - c < k:
        return 1.0
    # prod_{i=n-c+1}^{n} (1 - k/i) == 1 - C(n-c,k)/C(n,k), computed stably
    p = 1.0
    for i in range(n - c + 1, n + 1):
        p *= 1.0 - k / i
    return 1.0 - p


def score_file(path, num_generation, workers):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if len(rows) % num_generation != 0:
        raise ValueError(
            f"{path}: {len(rows)} rows is not a multiple of num_generation={num_generation}"
        )

    with ProcessPoolExecutor(max_workers=workers) as ex:
        flags = list(ex.map(_is_correct, rows, chunksize=8))

    # correct-count per problem
    counts = [
        int(sum(flags[i : i + num_generation]))
        for i in range(0, len(flags), num_generation)
    ]
    return counts, sum(flags) / len(flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="aime2024_*.jsonl paths (globs ok)")
    ap.add_argument("--num-generation", type=int, default=32)
    ap.add_argument("--ks", default="1,8,16,32")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    paths = sorted({p for pat in args.files for p in glob.glob(pat)})
    if not paths:
        raise SystemExit("no input files matched")

    results = []
    for path in paths:
        tag = os.path.basename(path)
        if tag.startswith("aime2024_"):
            tag = tag[len("aime2024_") :]
        tag = tag.removesuffix(".jsonl").removesuffix("_avg32")

        counts, avg = score_file(path, args.num_generation, args.workers)
        n = args.num_generation
        row = {"tag": tag, "n_problems": len(counts), f"avg@{n}": avg}
        for k in ks:
            if k > n:
                continue
            row[f"pass@{k}"] = sum(pass_at_k(n, c, k) for c in counts) / len(counts)
        results.append(row)
        print(f"[scored] {tag}: {len(counts)} problems", flush=True)

    n = args.num_generation
    cols = [f"avg@{n}"] + [f"pass@{k}" for k in ks if k <= n]
    width = max(len(r["tag"]) for r in results)
    print()
    print("model".ljust(width) + "  " + "  ".join(c.rjust(9) for c in cols))
    print("-" * (width + 2 + 11 * len(cols)))
    for r in results:
        cells = "  ".join(f"{r[c] * 100:8.2f}%" for c in cols)
        print(f"{r['tag'].ljust(width)}  {cells}")


if __name__ == "__main__":
    main()
