#!/usr/bin/env python3
"""Probe the teacher tokenizer for a single-token positive/negative pair.

Selects the first candidate pair where BOTH phrases encode to exactly one
token id in the teacher tokenizer's vocabulary. Falls back to "first-token
argmax" only if no candidate pair is fully single-token — but reports it
loudly, since the whole scoring assumes single-token symmetry.

Usage:
    uv run python research/selfcheck_gate/probe_tokenizer.py \
        --teacher-model /dev/shm/llms/Skywork-OR1-Math-7B/ \
        --output research/selfcheck_gate/artifacts/probe_tokens.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

# Order = preference. First pair that is single-token on BOTH sides wins.
CANDIDATE_PAIRS: list[tuple[str, str]] = [
    (" on-track", " off-track"),
    (" continue", " restart"),
    (" correct", " wrong"),
    (" good", " bad"),
]

# Preceding context that will actually appear before the probe token at
# inference time. Encoding the candidate WITH this prefix and diffing the
# ids is more faithful than encoding the candidate alone (BPE merges
# behave differently at sequence start).
PROBE_TEMPLATE = "\nSelf-check: from here I should"


def token_ids_for_candidate(tokenizer, candidate: str) -> list[int]:
    """Return the incremental token ids added by appending ``candidate`` to the probe template."""
    base_ids = tokenizer(PROBE_TEMPLATE, add_special_tokens=False).input_ids
    full_ids = tokenizer(PROBE_TEMPLATE + candidate, add_special_tokens=False).input_ids
    if full_ids[: len(base_ids)] != base_ids:
        raise RuntimeError(
            f"Tokenizer is not prefix-stable for candidate {candidate!r}; "
            f"cannot use incremental encoding."
        )
    return full_ids[len(base_ids) :]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)

    report: list[dict] = []
    chosen: dict | None = None
    for positive, negative in CANDIDATE_PAIRS:
        pos_ids = token_ids_for_candidate(tokenizer, positive)
        neg_ids = token_ids_for_candidate(tokenizer, negative)
        entry = {
            "positive": positive,
            "negative": negative,
            "positive_ids": pos_ids,
            "negative_ids": neg_ids,
            "positive_tokens": tokenizer.convert_ids_to_tokens(pos_ids),
            "negative_tokens": tokenizer.convert_ids_to_tokens(neg_ids),
            "both_single_token": len(pos_ids) == 1 and len(neg_ids) == 1,
        }
        report.append(entry)
        if entry["both_single_token"] and chosen is None:
            chosen = entry

    if chosen is None:
        # Fall back: first pair whose FIRST token differs — we'll compare
        # first-token logits only. Loud warning; caller should reconsider.
        fallback = None
        for entry in report:
            if entry["positive_ids"][:1] != entry["negative_ids"][:1]:
                fallback = entry
                break
        if fallback is None:
            raise RuntimeError(
                "No candidate pair has a distinguishable first token; "
                "add more candidates."
            )
        fallback = dict(fallback)
        fallback["fallback_first_token_only"] = True
        chosen = fallback

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "teacher_model": args.teacher_model,
        "probe_template": PROBE_TEMPLATE,
        "candidates": report,
        "chosen": chosen,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(chosen, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
