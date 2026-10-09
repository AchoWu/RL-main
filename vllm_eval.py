"""Evaluate a math model with vLLM, scored the same way training scores it.

This script is for measuring the before/after of a distillation run, so every
knob that could make the number differ from the training loop's own
``val:accuracy`` is deliberately aligned with it:

* **Prompt**: the dataset's prompt is used verbatim, matching
  ``data.prompt_file=null``. The old hardcoded MATH_QUERY_TEMPLATE asked for a
  different answer format than the one the student was distilled on, which
  understates the student.
* **Verifier**: ``math_metric`` with the same extraction targets as
  ``HFVerifyWorker`` (nemo_rl/environments/math_environment.py), and the gold
  wrapped as ``\\boxed{gt}``. The previous ``parse``/``verify`` pair used
  LatexExtractionConfig only, which scores some responses differently -- notably
  a response that is correct but rambles on into a second problem, where it
  extracts the *later* stray answer and marks a correct response wrong.
* **Stop token**: taken from the chat template's assistant terminator rather
  than ``tokenizer.eos_token``. These models declare eos as '</s>' but actually
  end a turn with '<|im_end|>', so relying on the declared eos lets every
  rollout run to max_tokens and hallucinate extra turns.

``finish_reason`` is recorded per sample, so a misconfigured stop token shows up
as a ``length`` rate instead of silently depressing accuracy.

Usage:
    python vllm_eval.py --model_path <ckpt> --test_file <jsonl> \
        --output_path preds.jsonl --prompt_key prompt --answer_key ground_truth
"""

import argparse
import json
import logging
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def setup_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )


def resolve_stop_token_ids(tokenizer):
    """Return the ids that actually end an assistant turn.

    tokenizer.eos_token is only what the config declares; the chat template's
    terminator is what the model was trained to emit. When they disagree,
    trusting eos means nothing ever stops the generation.
    """
    stop_ids = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)

    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
            tokenize=False,
            add_special_tokens=False,
        )
    except Exception:  # no chat template -- eos is all we have
        return stop_ids

    turn_end = rendered[rendered.rindex("y") + 1 :]
    for token in tokenizer.convert_ids_to_tokens(
        tokenizer(turn_end, add_special_tokens=False)["input_ids"]
    ):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id not in stop_ids:
            if token in tokenizer.all_special_tokens or token.startswith("<|"):
                stop_ids.append(token_id)
                logging.info(
                    f"Chat template ends an assistant turn with {token!r} "
                    f"(id={token_id}); adding it to stop_token_ids."
                )
    return stop_ids


def build_model_and_tokenizer(model_path, tensor_parallel_size=1, max_model_len=None):
    logging.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    llm = LLM(
        model=model_path,
        dtype="auto",
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        **({"max_model_len": max_model_len} if max_model_len else {}),
    )
    return llm, tokenizer


def batch_generate_vllm(prompts, llm, tokenizer, gen_cfg, stop_token_ids):
    """Generate one response per entry of ``prompts``.

    vLLM batches internally and the engine is the bottleneck, so the whole list
    is submitted at once rather than in a Python-level loop -- that lets vLLM
    schedule across all requests instead of serializing chunks.
    """
    sampling_params = SamplingParams(
        temperature=gen_cfg["temperature"],
        top_k=gen_cfg["top_k"],
        top_p=gen_cfg["top_p"],
        max_tokens=gen_cfg["max_new_tokens"],
        stop_token_ids=stop_token_ids,
        seed=gen_cfg.get("seed"),
    )

    # The prompt is used as-is: training runs with data.prompt_file=null, and
    # these datasets already carry their own "reason step by step / \boxed{}"
    # instruction. Wrapping it again would evaluate a format the student was
    # never distilled on.
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        for p in prompts
    ]

    outputs = llm.generate(rendered, sampling_params)
    return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outputs]


def predict(args):
    gen_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
    }

    with open(args.test_file, "r", encoding="utf-8") as f:
        data = [json.loads(row) for row in f if row.strip()]
    if args.limit:
        data = data[: args.limit]

    missing = [k for k in (args.prompt_key, args.answer_key) if k not in data[0]]
    if missing:
        raise KeyError(
            f"{missing} not in {args.test_file}; available keys: {sorted(data[0])}. "
            "Set --prompt_key/--answer_key to match the file."
        )

    n = args.num_generation
    prompts = [d[args.prompt_key] for d in data for _ in range(n)]
    answers = [str(d[args.answer_key]) for d in data for _ in range(n)]
    # Group id ties the n samples of one problem together for pass@n.
    group_ids = [i for i, _ in enumerate(data) for _ in range(n)]

    llm, tokenizer = build_model_and_tokenizer(
        args.model_path, args.tensor_parallel_size, args.max_model_len
    )
    stop_token_ids = resolve_stop_token_ids(tokenizer)
    logging.info(f"stop_token_ids={stop_token_ids}")

    results = batch_generate_vllm(prompts, llm, tokenizer, gen_cfg, stop_token_ids)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for gid, p, a, (r, fr) in zip(group_ids, prompts, answers, results):
            f.write(
                json.dumps(
                    {
                        "group_id": gid,
                        "problem": p,
                        "answer": a,
                        "response": r,
                        "finish_reason": fr,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    logging.info(f"Prediction saved to {args.output_path}")


def eval_metric(file_path):
    """Score predictions exactly as HFVerifyWorker does during training."""
    logging.info(f"Evaluating {file_path}")

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )

    with open(file_path, "r", encoding="utf-8") as f:
        rows = [json.loads(r) for r in f if r.strip()]

    correct = 0.0
    unparsed = 0
    bad_gold = 0
    errored = 0
    per_group = defaultdict(list)
    finish_reasons = defaultdict(int)

    for row in tqdm(rows, desc="Evaluating"):
        finish_reasons[row.get("finish_reason", "unknown")] += 1
        try:
            # Gold is wrapped so the latex extractor has something to match,
            # mirroring HFVerifyWorker's ground_truth_parsable.
            score, _ = verify_func(["\\boxed{" + row["answer"] + "}"], [row["response"]])
            score = float(score)
        except Exception as exc:
            score = 0.0
            # math_verify raises the same ValueError whether the *gold* failed
            # to parse or the prediction did, but they mean opposite things: a
            # bad gold is a dataset/answer-column problem that would silently
            # zero out a correct model, while an unparsed prediction really is
            # a miss. Keep them apart so a broken answer column is visible.
            if "No gold targets" in str(exc):
                bad_gold += 1
            elif isinstance(exc, (ValueError, IndexError, TypeError)):
                unparsed += 1
            else:
                errored += 1
                logging.warning(f"{type(exc).__name__} while scoring: {exc}")
        correct += score
        per_group[row.get("group_id", id(row))].append(score)

    total = len(rows)
    avg_at_n = correct / total if total else float("nan")
    # pass@n: a problem counts if any of its n samples is right. Distinct from
    # avg@n, which the previous version reported while being labelled pass@8.
    pass_at_n = (
        sum(1 for scores in per_group.values() if max(scores) > 0) / len(per_group)
        if per_group
        else float("nan")
    )
    n_per_group = total // len(per_group) if per_group else 0

    logging.info(f"samples={total} problems={len(per_group)} n={n_per_group}")
    logging.info(f"avg@{n_per_group} (mean accuracy) : {avg_at_n:.4f}")
    if n_per_group > 1:
        logging.info(f"pass@{n_per_group}                 : {pass_at_n:.4f}")
    logging.info(f"unparsed={unparsed} bad_gold={bad_gold} errored={errored}")
    if bad_gold:
        logging.warning(
            f"{bad_gold}/{total} ground truths did not parse, and every one of "
            "them scored 0 regardless of the model's answer. Check the "
            "--answer_key column before trusting this accuracy."
        )

    length_rate = finish_reasons.get("length", 0) / total if total else 0.0
    logging.info(f"finish_reason: {dict(finish_reasons)}")
    if length_rate > 0.1:
        logging.warning(
            f"{length_rate:.1%} of samples hit max_tokens instead of stopping. "
            "That usually means the stop token is wrong, and it depresses "
            "accuracy: an over-running response can ramble into a second "
            "problem whose stray answer gets extracted instead of the real one."
        )

    return avg_at_n


if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument(
        "--mode", type=str, default="all", choices=["predict", "eval", "all"]
    )
    # Field names of the test file. Defaults match UltraData-RL-Math-2609.
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--answer_key", type=str, default="ground_truth")
    parser.add_argument("--num_generation", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.mode in ["predict", "all"]:
        predict(args)
    if args.mode in ["eval", "all"]:
        eval_metric(args.output_path)
