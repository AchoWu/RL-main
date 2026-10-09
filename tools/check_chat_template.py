#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preflight-check the chat template a distillation run will actually use.

``math_hf_data_processor`` renders every prompt through
``tokenizer.apply_chat_template(..., add_generation_prompt=True)`` using the
*student's* tokenizer (``policy.tokenizer``), and that same tokenizer is handed
to the teacher policy. So the template is a silent, shared dependency of both
rollout and the teacher's logprobs. This script renders a real sample and
checks the things that actually break a run:

1. The student tokenizer has a chat template at all (a base model often does
   not -- ``apply_chat_template`` then raises).
2. The rendered prompt ends in an assistant generation prompt, so the model is
   being asked to answer rather than to continue the user's turn.
3. The student and teacher templates agree. The teacher scores tokens laid out
   by the student's template, so a mismatch feeds the teacher a prompt format
   it was never trained on.
4. EOS is set and is reachable, so rollouts can stop before max_new_tokens.
5. No double-BOS, which ``AllTaskProcessedDataset`` asserts on at runtime.
6. The prompt instruction is not duplicated (relevant when a dataset already
   carries "put your final answer within \\boxed{}" and prompt_file re-adds it).

Exits non-zero if a hard problem is found, so it can gate a training launch.

Usage:
    python tools/check_chat_template.py \
        --student /dev/shm/llms/JustRL-II-base-model/ \
        --teacher /dev/shm/llms/justrl2step100/ \
        --data ultradata_math_val.jsonl \
        --input-key prompt
"""

import argparse
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, help="Student model/tokenizer path.")
    parser.add_argument(
        "--teacher", default=None, help="Teacher model/tokenizer path (optional)."
    )
    parser.add_argument(
        "--data", default=None, help="JSONL whose first record is rendered."
    )
    parser.add_argument(
        "--input-key", default="prompt", help="Field holding the problem text."
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Prompt wrapper applied by data.prompt_file (omit when it is null).",
    )
    return parser.parse_args()


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    args = parse_args()
    from transformers import AutoTokenizer

    problems: list[str] = []
    hard_fail = False

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)

    # --- 1. template presence -------------------------------------------------
    section("1. Student chat template")
    template = tok.chat_template
    if template is None:
        print("[FAIL] tokenizer.chat_template is None.")
        print(
            "       math_hf_data_processor calls apply_chat_template and will "
            "raise ValueError at the first sample."
        )
        print(
            "       Fix: set policy.tokenizer.chat_template (null selects the "
            "passthrough template) or point at a model that has one."
        )
        problems.append("student has no chat template")
        hard_fail = True
    elif isinstance(template, dict):
        print(f"[WARN] Multiple named templates: {sorted(template.keys())}")
        problems.append("student template is a dict")
    else:
        oneline = " ".join(str(template).split())
        print(f"[ok] present, {len(str(template))} chars")
        print(f"     {oneline[:200]}{'...' if len(oneline) > 200 else ''}")

    # --- 2. the rendered prompt ----------------------------------------------
    section("2. Rendered prompt (what the model actually sees)")
    problem = "What is 1+1?"
    if args.data:
        with open(args.data, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    problem = json.loads(line)[args.input_key]
                    break

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            problem = f.read().format(problem)

    rendered = None
    if template is not None:
        try:
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
        except Exception as exc:  # noqa: BLE001 - surface any template error
            print(f"[FAIL] apply_chat_template raised: {type(exc).__name__}: {exc}")
            problems.append("apply_chat_template raised")
            hard_fail = True

    if rendered is not None:
        print("--- begin ---")
        print(rendered)
        print("--- end ---")

        # The generation prompt is the part the processor relies on: without it
        # the model is completing the user's turn instead of answering.
        bare = tok.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        suffix = rendered[len(bare) :] if rendered.startswith(bare) else ""
        if suffix.strip():
            print(f"[ok] generation prompt appended: {suffix!r}")
        else:
            print("[WARN] add_generation_prompt added nothing.")
            print(
                "       The model will continue the user turn rather than "
                "start an assistant turn. Common for base models."
            )
            problems.append("no generation prompt")

        # Instruction duplication: the dataset may already end with its own
        # "reason step by step / \boxed{}" instruction.
        low = rendered.lower()
        for phrase in ("step by step", "step-by-step"):
            if low.count(phrase) > 1:
                print(
                    f"[WARN] {phrase!r} appears {low.count(phrase)}x -- "
                    "prompt_file is likely duplicating the dataset instruction."
                )
                problems.append("duplicated instruction")
                break
        if low.count("boxed") > 1:
            print(f"[WARN] 'boxed' appears {low.count('boxed')}x (same cause).")

    # --- 3. student vs teacher ----------------------------------------------
    section("3. Student vs teacher template")
    if not args.teacher:
        print("[skip] no --teacher given")
    else:
        t_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
        if str(t_tok.chat_template) == str(template):
            print("[ok] identical templates")
        else:
            print("[WARN] templates DIFFER.")
            print(
                "       Only the student's tokenizer renders prompts, and the "
                "teacher scores those tokens, so the teacher sees a prompt "
                "format it was not trained on. Check the teacher's own format."
            )
            problems.append("student/teacher template mismatch")

        # Vocab equality is what NRL_SKIP_DISTILLATION_TOKENIZER_CHECK bypasses;
        # top-k logit distillation needs a shared token->id mapping.
        if t_tok.get_vocab() == tok.get_vocab():
            print("[ok] identical token->id mapping")
        else:
            print(
                "[FAIL] token->id mappings differ: top-k teacher indices do not "
                "refer to the same tokens for the student."
            )
            problems.append("vocab mismatch")
            hard_fail = True
        if t_tok.eos_token_id != tok.eos_token_id:
            print(
                f"[WARN] EOS differs: student {tok.eos_token_id} "
                f"vs teacher {t_tok.eos_token_id}"
            )
            problems.append("eos mismatch")

    # --- 4. EOS reachability -------------------------------------------------
    section("4. Stop condition")
    print(f"eos_token={tok.eos_token!r} id={tok.eos_token_id}")
    print(f"pad_token={tok.pad_token!r} id={tok.pad_token_id}")
    if tok.eos_token_id is None:
        print(
            "[FAIL] no EOS: configure_generation_config sets "
            "stop_token_ids=[eos_token_id], so rollouts cannot stop early."
        )
        problems.append("no eos")
        hard_fail = True
    elif rendered is not None:
        # If the template's assistant turn ends with a token the model never
        # emits, every rollout runs to max_new_tokens -- slow and truncated.
        with_answer = tok.apply_chat_template(
            [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": "42"},
            ],
            tokenize=False,
            add_special_tokens=False,
        )
        turn_end = with_answer[with_answer.rindex("42") + 2 :]
        print(f"assistant turn ends with: {turn_end!r}")
        if tok.eos_token and tok.eos_token in turn_end:
            print("[ok] template's assistant turn ends in EOS -> rollouts can stop")
        else:
            print(
                "[WARN] EOS does not appear at the end of the assistant turn. "
                "Confirm the model emits a stop token, or rollouts will run to "
                "max_new_tokens."
            )
            problems.append("eos not in assistant turn end")

    # --- 5. double BOS -------------------------------------------------------
    section("5. Double BOS (asserted at runtime)")
    if rendered is None or tok.bos_token_id is None:
        print("[skip] no BOS token or nothing rendered")
    else:
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        if len(ids) > 1 and ids[0] == ids[1] == tok.bos_token_id:
            print("[FAIL] double BOS -> assert_no_double_bos will trip.")
            problems.append("double bos")
            hard_fail = True
        else:
            print(f"[ok] first ids {ids[:3]}, bos={tok.bos_token_id}")

    # --- verdict -------------------------------------------------------------
    section("Verdict")
    if not problems:
        print("[ok] no problems found.")
        return 0
    for p in problems:
        print(f" - {p}")
    if hard_fail:
        print("\nHARD FAIL: the run would crash or distill against wrong tokens.")
        return 1
    print("\nWarnings only: inspect the rendered prompt above and decide.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
