#!/usr/bin/env python3
"""Estimate the per-problem contrastive baseline b_i on real problems.

Design choice (per user): baseline is computed per rollout at gate time,
using "problem + chat template (assistant opener, empty content)" as the
neutral anchor. This script is the OFFLINE diagnostic for that choice —
it computes b_i on a real problem sample and reports the distribution.

Read the output like this:
  - std, IQR, and range of b_i tell you how much the anchor moves per
    problem. Since gate-time uses per-rollout b_i (not the mean here),
    these numbers do NOT need to be small — they only need to be smaller
    than the eventual signal from real prefixes (Task 5 checks that).
  - The mean is a rough starting point for the threshold tau if we ever
    fall back to a global constant.

Usage:
    uv run python research/selfcheck_gate/estimate_baseline.py \
        --teacher-model /dev/shm/llms/Skywork-OR1-Math-7B/ \
        --probe-tokens research/selfcheck_gate/artifacts/probe_tokens.json \
        --output research/selfcheck_gate/artifacts/baseline_b.json \
        --num-problems 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repo root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nemo_rl.data.datasets.response_datasets.dapo_math import (  # noqa: E402
    prepare_dapo_math_processed_dataset,
)


SYSTEM_PROMPT = (
    "You are a helpful math assistant. Reason step by step and be concise."
)


def format_anchor(tokenizer, problem: str, probe_template: str) -> str:
    """Chat-template a problem + assistant opener + probe (no student content).

    This matches what gate-time per-rollout baseline computes: the log-ratio
    at the FIRST self-check position, before any student tokens have been
    emitted. b_i is problem-specific but student-independent.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    opener = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return f"{opener}{probe_template}"


@torch.inference_mode()
def score_prefix(model, tokenizer, text: str, pos_id: int, neg_id: int, device) -> float:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    logits = model(ids).logits[0, -1]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return float(log_probs[pos_id] - log_probs[neg_id])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--probe-tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-problems", type=int, default=200)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    probe_meta = json.loads(args.probe_tokens.read_text())
    chosen = probe_meta["chosen"]
    if not chosen.get("both_single_token", False):
        raise RuntimeError(
            "Chosen probe pair is not fully single-token; scoring assumes symmetry."
        )
    pos_id = int(chosen["positive_ids"][0])
    neg_id = int(chosen["negative_ids"][0])
    probe_template = probe_meta["probe_template"]

    print(f"Loading DAPO-Math-17k-Processed (seed={args.dataset_seed}) ...")
    splits = prepare_dapo_math_processed_dataset(seed=args.dataset_seed)
    train_ds = splits["train"]
    problems: list[str] = []
    for row in train_ds:
        msg = row["messages"][0]
        if msg["role"] == "user":
            problems.append(msg["content"])
        if len(problems) >= args.num_problems:
            break

    print(f"Loading teacher {args.teacher_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=dtype, trust_remote_code=True
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    scores: list[float] = []
    samples: list[dict] = []
    for i, problem in enumerate(problems):
        text = format_anchor(tokenizer, problem, probe_template)
        s = score_prefix(model, tokenizer, text, pos_id, neg_id, device)
        scores.append(s)
        if i < 5:
            samples.append({"problem": problem[:180], "log_ratio": s})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(problems)}")

    scores_t = torch.tensor(scores)
    payload = {
        "teacher_model": args.teacher_model,
        "probe_template": probe_template,
        "positive": chosen["positive"],
        "negative": chosen["negative"],
        "positive_id": pos_id,
        "negative_id": neg_id,
        "num_problems": len(scores),
        "baseline_mode": "per_rollout_diagnostic",
        "note": (
            "b is computed per rollout at gate time using the SAME anchor "
            "shape (problem + chat opener + probe). Statistics here just "
            "describe how much b_i moves across problems."
        ),
        "mean": float(scores_t.mean()),
        "std": float(scores_t.std(unbiased=False)),
        "min": float(scores_t.min()),
        "max": float(scores_t.max()),
        "quantiles": {
            "p10": float(scores_t.quantile(0.10)),
            "p25": float(scores_t.quantile(0.25)),
            "p50": float(scores_t.quantile(0.50)),
            "p75": float(scores_t.quantile(0.75)),
            "p90": float(scores_t.quantile(0.90)),
        },
        "iqr": float(scores_t.quantile(0.75) - scores_t.quantile(0.25)),
        "example_problems": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(
        {k: payload[k] for k in ["mean", "std", "min", "max", "quantiles", "iqr"]},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
