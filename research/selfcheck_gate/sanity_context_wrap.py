"""Sanity check: teacher's mean logprob on target under different context wrappings.

Manufacture 3 contexts for one correct rollout and see how P(target|ctx) moves:
  A: opener only (baseline b_i)
  B: opener + response (whatever the student produced)
  C: opener + response + "</think>\\n\\n"  (force conclude)
  D: opener + response.replace("</think>", "") + "</think>\\n\\n"  (normalize)

Print mean logprobs. Expect D > C > B, and A << D.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scoring import build_target_long, build_target_short, chat_opener, mean_target_logprob


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    args = parser.parse_args()

    # Grab the first CORRECT rollout to keep things clean.
    with args.rollouts.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    device = torch.device("cuda")
    model.to(device).eval()

    # Try the first 3 rollouts to see a small spread.
    for i, row in enumerate(rows[:5]):
        problem = row["problem"]
        gt = row["ground_truth"]
        response = row["response"]

        # Also compute correctness so we can see if s_final aligns with it.
        try:
            from math_verify import LatexExtractionConfig, parse, verify
            from latex2sympy2_extended import NormalizationConfig
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
            correct = bool(verify(parsed, gt))
        except Exception:
            correct = None

        opener = chat_opener(tokenizer, problem)
        tgt_long = build_target_long(gt)
        tgt_short = build_target_short(gt)

        # Variants
        ctx_A = opener
        ctx_B = opener + response
        ctx_C = opener + response + "\n</think>\n\n"
        response_no_think = response.replace("</think>", "")
        ctx_D = opener + response_no_think + "\n</think>\n\n"

        def score(ctx, tgt):
            n_tokens = len(tokenizer(ctx + tgt, add_special_tokens=False).input_ids)
            if n_tokens > 16384:
                return None
            return mean_target_logprob(model, tokenizer, ctx, tgt, device)

        print(f"--- rollout {i} (correct={correct}, gt={gt}, has_end_think={'</think>' in response}) ---")
        for name, ctx in [("A opener", ctx_A), ("B +response", ctx_B)]:
            for tname, tgt in [("long", tgt_long), ("short", tgt_short)]:
                s = score(ctx, tgt)
                s_str = f"{s:+.3f}" if s is not None else "TOOLONG"
                print(f"  {name:14s} {tname}: {s_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
