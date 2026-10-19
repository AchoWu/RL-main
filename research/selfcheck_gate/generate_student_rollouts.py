#!/usr/bin/env python3
"""Generate student rollouts on DAPO problems for offline gate validation.

Uses the same conventions as vllm_opd_aime.py:
  - R1-Distill has NO system prompt; chat template auto-inserts <think>\\n
  - MATH_QUERY_TEMPLATE: appends "put your final answer within \\boxed{}"
  - temperature=0.6, top_p=0.95 (R1 official defaults)

Output: JSONL, one line per rollout, fields:
    problem, ground_truth, response

Usage:
    uv run python research/selfcheck_gate/generate_student_rollouts.py \
        --student-model /dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/ \
        --num-problems 200 \
        --output research/selfcheck_gate/artifacts/rollouts.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoTokenizer  # noqa: E402

from nemo_rl.data.datasets.response_datasets.dapo_math import (  # noqa: E402
    prepare_dapo_math_processed_dataset,
)

MATH_QUERY_TEMPLATE = (
    "{Question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-problems", type=int, default=200)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    print(f"Loading DAPO-Math-17k-Processed (seed={args.dataset_seed}) ...")
    splits = prepare_dapo_math_processed_dataset(seed=args.dataset_seed)
    train_ds = splits["train"]

    problems: list[str] = []
    ground_truths: list[str] = []
    for row in train_ds:
        msgs = row["messages"]
        # msgs = [{'user': problem}, {'assistant': ground_truth}]
        user_msg = next(m for m in msgs if m["role"] == "user")
        gt_msg = next(m for m in msgs if m["role"] == "assistant")
        problems.append(user_msg["content"])
        ground_truths.append(gt_msg["content"])
        if len(problems) >= args.num_problems:
            break

    print(f"Loaded {len(problems)} problems. Loading vLLM ...")

    # Lazy import to keep argparse cheap
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model, trust_remote_code=True
    )
    llm = LLM(
        model=args.student_model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_model_len,
    )

    prompts_formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=p)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in problems
    ]

    outputs = llm.generate(prompts_formatted, sampling)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for problem, gt, out in zip(problems, ground_truths, outputs):
            row = {
                "problem": problem,
                "ground_truth": gt,
                "response": out.outputs[0].text,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(problems)} rollouts to {args.output}")

    # Quick correctness stats to sanity-check the rollouts.
    try:
        from math_verify import LatexExtractionConfig, parse, verify
        from latex2sympy2_extended import NormalizationConfig

        correct = 0
        for gt, out in zip(ground_truths, outputs):
            parsed = parse(
                out.outputs[0].text,
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
            correct += float(verify(parsed, gt))
        print(f"[sanity] student accuracy: {correct / len(problems):.4f}")
    except ImportError:
        print("[sanity] math_verify not available; skipping accuracy check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
