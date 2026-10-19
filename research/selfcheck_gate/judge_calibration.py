#!/usr/bin/env python3
"""Calibrate the DeepSeek judge on a small hand-labeled probe set.

Pipeline:
  1. From rollouts.jsonl, auto-pick candidate segments in three buckets:
       - "likely_salvageable": segments from CORRECT-answer rollouts
       - "likely_dead":        segments from WRONG-answer rollouts, taken
                               from the LAST third of the CoT (where drift
                               has had time to accumulate)
       - "backtrack":          segments immediately AFTER phrases like
                               "Wait,", "Actually,", "Let me reconsider"
                               (regardless of final correctness). These
                               are the segments B-probe is most at risk of
                               misjudging — the judge MUST call these
                               salvageable, else the whole gate is toast.
  2. Emit review CSV. You manually correct the `true_label` column.
  3. Re-run this script in --score mode: it queries DeepSeek judge for
     every segment and computes accuracy vs the corrected labels.

Usage:
    # Phase A: build the review file
    uv run python research/selfcheck_gate/judge_calibration.py build \
        --rollouts research/selfcheck_gate/artifacts/rollouts.jsonl \
        --output   research/selfcheck_gate/artifacts/judge_review.jsonl \
        --per-bucket 10

    # Phase B (after you edit the JSONL to fix `true_label` fields)
    uv run python research/selfcheck_gate/judge_calibration.py score \
        --review research/selfcheck_gate/artifacts/judge_review.jsonl \
        --report research/selfcheck_gate/artifacts/judge_calibration.json \
        --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from get_ds_response import get_ds_result  # noqa: E402

BACKTRACK_TRIGGERS = [
    r"\bWait,",
    r"\bActually,",
    r"\bHmm,",
    r"\bLet me reconsider",
    r"\bLet me re-check",
    r"\bLet me recheck",
    r"\bOn second thought",
    r"\bI made a mistake",
]

JUDGE_SYSTEM = (
    "You are a strict mathematical reasoning judge. Answer questions with "
    "exactly one word (YES or NO) and nothing else."
)

JUDGE_TEMPLATE = """Problem:
{problem}

Student's reasoning so far:
{prefix}

Question: Based on the reasoning above, is the student on a path that can
still reach the correct final answer? Consider that the student might
still self-correct in the next few steps, so a temporary mistake that the
student is about to fix should be judged as still salvageable.

Answer with exactly one word: YES or NO."""


def split_paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]


def check_answer(response: str, ground_truth: str) -> bool:
    try:
        from math_verify import LatexExtractionConfig, parse, verify
        from latex2sympy2_extended import NormalizationConfig
    except ImportError:
        return False
    parsed = parse(
        response,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False, malformed_operators=False,
                    basic_latex=True, equations=True, boxed="all", units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    return bool(verify(parsed, ground_truth))


def find_backtrack_positions(paragraphs: list[str]) -> list[int]:
    """Return indices of paragraphs that OPEN with a backtrack trigger."""
    triggers = re.compile("|".join(BACKTRACK_TRIGGERS))
    return [i for i, p in enumerate(paragraphs) if triggers.match(p.lstrip())]


def prefix_before(paragraphs: list[str], k: int) -> str:
    return "\n\n".join(paragraphs[:k]).strip()


def build_review(
    rollouts_path: Path, output_path: Path, per_bucket: int, seed: int
) -> None:
    rng = random.Random(seed)
    rows = [json.loads(line) for line in rollouts_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    correct_rows: list[dict] = []
    wrong_rows: list[dict] = []
    for row in rows:
        ok = check_answer(row["response"], row["ground_truth"])
        (correct_rows if ok else wrong_rows).append({**row, "correct": ok})
    print(f"[build] rollouts: {len(rows)} total, {len(correct_rows)} correct, {len(wrong_rows)} wrong")

    salvageable: list[dict] = []
    dead: list[dict] = []
    backtrack: list[dict] = []

    for row in correct_rows:
        paras = split_paragraphs(row["response"])
        if len(paras) < 4:
            continue
        k = rng.randint(2, max(3, len(paras) // 2))  # middle of a correct trajectory
        salvageable.append({
            "problem": row["problem"],
            "ground_truth": row["ground_truth"],
            "final_correct": True,
            "cut_at_paragraph": k,
            "prefix": prefix_before(paras, k),
            "bucket": "likely_salvageable",
            "true_label": "salvageable",  # you can flip to "dead" if you spot obvious drift
        })

    for row in wrong_rows:
        paras = split_paragraphs(row["response"])
        if len(paras) < 4:
            continue
        # last third — drift has had time to accumulate but not so late that
        # the "reasoning so far" already contains the wrong \boxed answer.
        low = max(2, int(len(paras) * 0.67))
        high = max(low + 1, len(paras) - 1)
        k = rng.randint(low, high)
        dead.append({
            "problem": row["problem"],
            "ground_truth": row["ground_truth"],
            "final_correct": False,
            "cut_at_paragraph": k,
            "prefix": prefix_before(paras, k),
            "bucket": "likely_dead",
            "true_label": "dead",  # flip if the segment looks fine
        })

    for row in rows:
        paras = split_paragraphs(row["response"])
        for k in find_backtrack_positions(paras):
            if k < 2:
                continue
            backtrack.append({
                "problem": row["problem"],
                "ground_truth": row["ground_truth"],
                "final_correct": check_answer(row["response"], row["ground_truth"]),
                "cut_at_paragraph": k,
                "prefix": prefix_before(paras, k),
                "bucket": "backtrack",
                "true_label": "salvageable",  # HYPOTHESIS: teacher wants to fix; flip if actually dead
            })
            break  # one per rollout is enough

    rng.shuffle(salvageable)
    rng.shuffle(dead)
    rng.shuffle(backtrack)
    picked = salvageable[:per_bucket] + dead[:per_bucket] + backtrack[:per_bucket]
    rng.shuffle(picked)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in picked:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    counts = {"salvageable": len(salvageable), "dead": len(dead), "backtrack": len(backtrack)}
    print(f"[build] bucket sizes before capping: {counts}")
    print(f"[build] wrote {len(picked)} candidate segments to {output_path}")
    print("[build] NEXT: open the JSONL and, for each segment, verify the")
    print("        `true_label` field. Values must be 'salvageable' or 'dead'.")


def call_judge(problem: str, prefix: str) -> tuple[str, str]:
    """Return (raw_response, parsed_label ∈ {'salvageable','dead','unparsable'})."""
    text = get_ds_result(
        JUDGE_TEMPLATE.format(problem=problem, prefix=prefix),
        system_prompt=JUDGE_SYSTEM,
        temperature=0.0,
        max_tokens=8,
    )
    up = text.strip().upper()
    if up.startswith("YES"):
        return text, "salvageable"
    if up.startswith("NO"):
        return text, "dead"
    return text, "unparsable"


def score_review(review_path: Path, report_path: Path, concurrency: int) -> None:
    entries = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    for e in entries:
        if e.get("true_label") not in ("salvageable", "dead"):
            raise SystemExit(
                f"Entry missing valid true_label ({e.get('true_label')!r}); "
                "hand-label all entries before scoring."
            )

    def worker(i: int, entry: dict) -> tuple[int, str, str]:
        raw, pred = call_judge(entry["problem"], entry["prefix"])
        return i, raw, pred

    results: dict[int, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(worker, i, e) for i, e in enumerate(entries)]
        for fut in as_completed(futs):
            i, raw, pred = fut.result()
            results[i] = (raw, pred)
            print(f"  {i + 1}/{len(entries)} -> {pred}")

    per_bucket: dict[str, dict[str, int]] = {}
    confusion = {"salvageable": {"salvageable": 0, "dead": 0, "unparsable": 0},
                 "dead":        {"salvageable": 0, "dead": 0, "unparsable": 0}}
    detailed: list[dict] = []

    for i, entry in enumerate(entries):
        raw, pred = results[i]
        true = entry["true_label"]
        bucket = entry["bucket"]
        confusion[true][pred] += 1
        b = per_bucket.setdefault(bucket, {"total": 0, "correct": 0, "wrong": 0, "unparsable": 0})
        b["total"] += 1
        if pred == "unparsable":
            b["unparsable"] += 1
        elif pred == true:
            b["correct"] += 1
        else:
            b["wrong"] += 1
        detailed.append({**entry, "judge_raw": raw, "judge_pred": pred})

    n = len(entries)
    parsable = sum(1 for i in range(n) if results[i][1] != "unparsable")
    correct = sum(1 for i in range(n) if results[i][1] == entries[i]["true_label"])
    summary = {
        "n_total": n,
        "n_parsable": parsable,
        "overall_accuracy_on_parsable": correct / parsable if parsable else 0.0,
        "confusion_true_x_pred": confusion,
        "per_bucket": per_bucket,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"summary": summary, "entries": detailed}, indent=2, ensure_ascii=False))
    print(f"[score] wrote detailed report to {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="produce candidate segments for hand review")
    b.add_argument("--rollouts", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--per-bucket", type=int, default=10)
    b.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("score", help="query judge on hand-labeled review file")
    s.add_argument("--review", type=Path, required=True)
    s.add_argument("--report", type=Path, required=True)
    s.add_argument("--concurrency", type=int, default=16)

    args = parser.parse_args()

    if args.cmd == "build":
        build_review(args.rollouts, args.output, args.per_bucket, args.seed)
    elif args.cmd == "score":
        score_review(args.review, args.report, args.concurrency)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
