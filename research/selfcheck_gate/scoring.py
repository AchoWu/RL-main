"""Shared helpers for answer-anchored self-check gate scoring.

Score:
    b_i    = mean_t log P_theta(target[t] | problem + chat_opener + target[:t])
    s_k    = mean_t log P_theta(target[t] | problem + chat_opener + student_prefix_k + target[:t]) - b_i

Positive s_k = student prefix makes the teacher MORE confident it can emit
the correct final answer; deeply negative s_k = student went off a cliff.

target_str is the answer-anchored suffix that ends the CoT and asserts the
correct final answer. Wording follows the DAPO prompt convention
("put your final answer within \\boxed{}"). GT is short (mean 2.3 char, max 5
in the 200-rollout sample), so per-token mean over ~10 target tokens is
stable.
"""

from __future__ import annotations

import torch


SYSTEM_PROMPT = (
    "You are a helpful math assistant. Reason step by step and be concise."
)

MATH_QUERY_TEMPLATE = (
    "{Question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)


def build_target_long(ground_truth: str) -> str:
    """Long target: full CoT-closer sentence + boxed answer.

    Includes "Therefore, the final answer is" boilerplate. The boilerplate
    tokens are gt-independent but their log-prob absorbs the "is the teacher
    ready to conclude?" confound — useful when scoring at mid-CoT cut points
    where the teacher might not naturally emit \\boxed{} yet.
    """
    return f"\n\nTherefore, the final answer is $\\boxed{{{ground_truth}}}$."


def build_target_short(ground_truth: str) -> str:
    """Short target: bare boxed answer.

    Every token is gt-specific, so per-token mean has higher signal density
    than the long target. But because there is no "we are concluding now"
    lead-in, scores at mid-CoT prefixes may confound "wrong reasoning" with
    "teacher not ready to conclude". Best for late-cut scoring.
    """
    return f"$\\boxed{{{ground_truth}}}$"


# Back-compat default: kept for callers that don't pick a variant explicitly.
build_target = build_target_long


def chat_opener(tokenizer, problem: str) -> str:
    """Chat-template problem + assistant opener (no student content yet)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=problem)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def mean_target_logprob(
    model,
    tokenizer,
    context: str,
    target: str,
    device: torch.device,
) -> float:
    """Return per-token mean log P(target | context) under ``model``.

    The context and target are concatenated at the string level; the token
    boundary is derived by re-tokenizing both. We use the incremental-diff
    trick (encode context+target, encode context, diff ids) to avoid the
    "space at start of new segment" BPE surprise.
    """
    context_ids = tokenizer(context, add_special_tokens=False).input_ids
    full_ids = tokenizer(context + target, add_special_tokens=False).input_ids
    if full_ids[: len(context_ids)] != context_ids:
        # Fall back to independent tokenization; may be slightly noisier at
        # the seam but avoids a crash.
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        full_ids = context_ids + target_ids
    target_ids = full_ids[len(context_ids) :]
    if not target_ids:
        return 0.0

    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids).logits[0]  # (T, V)
    # Predict token t from position t-1. Target starts at index len(context_ids).
    start = len(context_ids)
    # position preds for target token j (0-indexed within target) come from
    # logits[start - 1 + j]
    pred_logits = logits[start - 1 : start - 1 + len(target_ids)]
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    tgt = torch.tensor(target_ids, device=device)
    picked = log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
    return float(picked.mean())
