#!/usr/bin/env python3
"""Score every \\n\\n cut point in every rollout with the answer-anchored gate.

For each rollout:
  b_i          = mean_t log P(target[t] | opener, target[:t])
  s_k          = mean_t log P(target[t] | opener + prefix_k, target[:t]) - b_i
where target = "\\n\\nTherefore, the final answer is $\\boxed{gt}$." and
prefix_k is the first k paragraphs of the student rollout.

We also emit s_final (whole rollout as prefix) as a sanity anchor. Expected:
correct-answer rollouts should have s_final >= 0, wrong-answer << 0.

Usage:
    uv run python research/selfcheck_gate/score_prefixes.py \
        --teacher-model /dev/shm/llms/Skywork-OR1-Math-7B/ \
        --rollouts research/selfcheck_gate/artifacts/rollouts.jsonl \
        --output   research/selfcheck_gate/artifacts/scores.jsonl \
        --num-rollouts 200

Output: JSONL, one line per rollout, fields:
    problem, ground_truth, response, correct (bool),
    b_i, s_final, cuts: [{k, s_k, prefix_char_len, ...}]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scoring import (  # noqa: E402
    build_target_long,
    build_target_short,
    chat_opener,
    mean_target_logprob,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-rollouts", type=int, default=None)
    parser.add_argument("--max-cuts-per-rollout", type=int, default=20,
                        help="Cap on cut points scored per rollout, taken from the "
                             "HEAD of the rollout. Rationale: if a rollout goes off, "
                             "it usually goes off early — dense early coverage is "
                             "more informative than uniform sampling across the tail. "
                             "Also short prefixes are cheap to score.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-context-tokens", type=int, default=16384,
                        help="Skip scoring for any context+target longer than this (teacher max_model_len).")
    args = parser.parse_args()

    rollouts: list[dict] = []
    with args.rollouts.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rollouts.append(json.loads(line))
    if args.num_rollouts:
        rollouts = rollouts[: args.num_rollouts]
    print(f"Loaded {len(rollouts)} rollouts")

    print(f"Loading teacher {args.teacher_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=dtype, trust_remote_code=True
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_f:
        for i, row in enumerate(rollouts):
            problem = row["problem"]
            gt = row["ground_truth"]
            response = row["response"]

            targets = {
                "long": build_target_long(gt),
                "short": build_target_short(gt),
            }
            opener = chat_opener(tokenizer, problem)

            def safe_score(ctx: str, tgt: str) -> float | None:
                n = len(tokenizer(ctx + tgt, add_special_tokens=False).input_ids)
                if n > args.max_context_tokens:
                    return None
                return mean_target_logprob(model, tokenizer, ctx, tgt, device)

            b_i = {name: safe_score(opener, tgt) for name, tgt in targets.items()}

            # s_final: whole rollout as prefix
            full_prefix_ctx = opener + response
            s_final_raw = {name: safe_score(full_prefix_ctx, tgt) for name, tgt in targets.items()}
            s_final = {
                name: (s_final_raw[name] - b_i[name])
                if s_final_raw[name] is not None and b_i[name] is not None
                else None
                for name in targets
            }

            # Per-cut scoring at every \n\n boundary, capped to first N cuts.
            paragraphs = split_paragraphs(response)
            cuts: list[dict] = []
            skipped_cuts = 0
            all_cuts = list(range(1, len(paragraphs)))
            if args.max_cuts_per_rollout is not None:
                cut_range = all_cuts[: args.max_cuts_per_rollout]
            else:
                cut_range = all_cuts
            for k in cut_range:
                prefix_text = "\n\n".join(paragraphs[:k])
                ctx = opener + prefix_text + "\n\n"
                s_k_raw = {name: safe_score(ctx, tgt) for name, tgt in targets.items()}
                if all(v is None for v in s_k_raw.values()):
                    skipped_cuts += 1
                    continue
                cuts.append({
                    "k": k,
                    "prefix_char_len": len(prefix_text),
                    "s_k_raw_long": s_k_raw["long"],
                    "s_k_raw_short": s_k_raw["short"],
                    "s_k_long": (s_k_raw["long"] - b_i["long"])
                        if s_k_raw["long"] is not None and b_i["long"] is not None else None,
                    "s_k_short": (s_k_raw["short"] - b_i["short"])
                        if s_k_raw["short"] is not None and b_i["short"] is not None else None,
                })

            record = {
                "index": i,
                "problem": problem,
                "ground_truth": gt,
                "response": response,
                "correct": check_answer(response, gt),
                "b_i_long": b_i["long"],
                "b_i_short": b_i["short"],
                "s_final_long": s_final["long"],
                "s_final_short": s_final["short"],
                "s_final_raw_long": s_final_raw["long"],
                "s_final_raw_short": s_final_raw["short"],
                "num_paragraphs": len(paragraphs),
                "skipped_cuts_toolong": skipped_cuts,
                "cuts": cuts,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            # Release cached blocks between rollouts to avoid CUDA state drift
            # on long runs with wildly varying sequence lengths.
            torch.cuda.empty_cache()

            if (i + 1) % 5 == 0 or i == 0:
                sfl = f"{s_final['long']:+.3f}" if s_final['long'] is not None else "NULL"
                sfs = f"{s_final['short']:+.3f}" if s_final['short'] is not None else "NULL"
                bl = f"{b_i['long']:.3f}" if b_i['long'] is not None else "NULL"
                bs = f"{b_i['short']:.3f}" if b_i['short'] is not None else "NULL"
                print(
                    f"  {i + 1}/{len(rollouts)}: correct={record['correct']} "
                    f"b_long={bl} b_short={bs} "
                    f"s_final_long={sfl} s_final_short={sfs} "
                    f"n_cuts={len(cuts)} skipped_toolong={skipped_cuts}"
                )

    print(f"Wrote scores to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
