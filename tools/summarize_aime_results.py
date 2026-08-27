"""Summarize AIME-2024 eval results written by run_aime_eval_batch.sh."""

import argparse
import glob
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "acc_*.txt"))):
        tag = os.path.basename(path)[len("acc_") : -len(".txt")]
        with open(path) as f:
            text = f.read().strip()
        m = re.search(r"acc:\s*([0-9.]+)", text)
        rows.append((tag, float(m.group(1)) if m else None))

    if not rows:
        print("no results found")
        return

    width = max(len(t) for t, _ in rows)
    print(f"{'checkpoint'.ljust(width)}   AIME-2024 avg@32")
    print("-" * (width + 20))
    for tag, acc in rows:
        print(f"{tag.ljust(width)}   {'FAILED' if acc is None else f'{acc:.4f}  ({acc * 100:.2f}%)'}")


if __name__ == "__main__":
    main()
