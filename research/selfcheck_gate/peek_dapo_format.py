"""Peek at DAPO problem/gt format to inform target-string wording."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts",
        type=Path,
        default=Path("research/selfcheck_gate/artifacts/rollouts.jsonl"),
    )
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    rows: list[dict] = []
    with args.rollouts.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"Total rollouts: {len(rows)}")
    print()

    # Show tail of problem text + gt for a few samples.
    for i, row in enumerate(rows[: args.num_samples]):
        print(f"=== ROW {i} PROBLEM tail (last 300 chars) ===")
        print(row["problem"][-300:])
        print(f"=== ROW {i} GT ===")
        print(repr(row["ground_truth"]))
        print()

    # gt length distribution over all rollouts.
    gts: list[str] = [row["ground_truth"] for row in rows]
    lengths = [len(g) for g in gts]
    print("=== GT length stats (chars) ===")
    print(
        f"min={min(lengths)} max={max(lengths)} "
        f"mean={statistics.mean(lengths):.1f} median={statistics.median(lengths)}"
    )
    print("longest 3:", sorted(gts, key=len, reverse=True)[:3])
    print("shortest 3:", sorted(gts, key=len)[:3])

    # How many problem texts explicitly mention \boxed?
    with_boxed = sum(1 for row in rows if "boxed" in row["problem"].lower())
    print()
    print(f"Problems mentioning 'boxed': {with_boxed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
