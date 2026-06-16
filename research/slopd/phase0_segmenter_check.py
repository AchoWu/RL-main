"""Phase 0.1: Segmenter robustness check on AIME 2024.

Generate CoT rollouts from Qwen3-1.7B (no training, instruct mode) on AIME 2024
problems, run the math-aware segmenter on each rollout, and report:
- Distribution of sentence count per trajectory
- Distribution of sentence length (chars and rough tokens)
- Fraction of pathological sentences (too short / too long)
- Sample inspection: dump 5 trajectories with their segmentations

Pass criteria:
- 90%+ trajectories produce >= 3 sentences
- Pathological sentence rate < 10%
- Median sentence count >= 8 (AIME is long-CoT)

Usage:
    python phase0_segmenter_check.py \\
        --model_path /group/40143/howu/llms/Qwen3-1.7B \\
        --data_path /path/to/aime_2024.jsonl \\
        --out_dir ./phase0_results \\
        --max_problems 30 \\
        --max_new_tokens 4096
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as stats
import sys
from pathlib import Path

# Ensure local segmenter is importable when run from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmenter import segment, SegmentStats  # noqa: E402


PROMPT_TEMPLATE = """Solve the following math problem. Show your reasoning step by step, then give the final answer.

Problem: {problem}

Solution:"""


def load_problems(path: str, max_problems: int) -> list[dict]:
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
            if len(problems) >= max_problems:
                break
    return problems


def generate_rollouts(
    model_path: str,
    problems: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Run vLLM-based generation. Falls back to HF transformers if vLLM unavailable."""
    try:
        from vllm import LLM, SamplingParams
        print(f"[generate] using vLLM, model={model_path}")
        llm = LLM(model=model_path, dtype="bfloat16", trust_remote_code=True,
                  gpu_memory_utilization=0.85, max_model_len=8192)
        sampling = SamplingParams(
            temperature=temperature, top_p=top_p,
            max_tokens=max_new_tokens, n=1,
        )
        prompts = [PROMPT_TEMPLATE.format(problem=p["problem"]) for p in problems]
        outputs = llm.generate(prompts, sampling)
        return [o.outputs[0].text for o in outputs]
    except ImportError:
        print("[generate] vLLM not available, falling back to HF transformers")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[generate] using HF transformers, model={model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rollouts = []
    for i, problem in enumerate(problems):
        prompt = PROMPT_TEMPLATE.format(problem=problem["problem"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        rollouts.append(text)
        print(f"  [{i+1}/{len(problems)}] generated {len(text)} chars")
    return rollouts


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[k])


def report(stats_list: list[SegmentStats], out_dir: Path) -> dict:
    n_traj = len(stats_list)
    sent_counts = [s.n_sentences for s in stats_list]
    all_char_lens = [L for s in stats_list for L in s.sentence_char_lens]
    all_token_lens = [L for s in stats_list for L in s.sentence_token_lens]
    n_pathological = sum(s.too_short + s.too_long for s in stats_list)
    n_total_sents = sum(sent_counts)

    pass_3plus = sum(1 for c in sent_counts if c >= 3) / max(1, n_traj)
    median_sents = stats.median(sent_counts) if sent_counts else 0
    patho_rate = n_pathological / max(1, n_total_sents)

    summary = {
        "n_trajectories": n_traj,
        "n_total_sentences": n_total_sents,
        "sentence_count_per_traj": {
            "mean": stats.mean(sent_counts) if sent_counts else 0,
            "median": median_sents,
            "p10": percentile(sent_counts, 10),
            "p90": percentile(sent_counts, 90),
            "min": min(sent_counts) if sent_counts else 0,
            "max": max(sent_counts) if sent_counts else 0,
        },
        "sentence_char_length": {
            "mean": stats.mean(all_char_lens) if all_char_lens else 0,
            "median": stats.median(all_char_lens) if all_char_lens else 0,
            "p10": percentile(all_char_lens, 10),
            "p90": percentile(all_char_lens, 90),
        },
        "sentence_token_length": {
            "mean": stats.mean(all_token_lens) if all_token_lens else 0,
            "median": stats.median(all_token_lens) if all_token_lens else 0,
            "p10": percentile(all_token_lens, 10),
            "p90": percentile(all_token_lens, 90),
        },
        "pathological": {
            "total": n_pathological,
            "rate": patho_rate,
            "too_short": sum(s.too_short for s in stats_list),
            "too_long": sum(s.too_long for s in stats_list),
        },
        "pass_criteria": {
            "frac_traj_with_3plus_sentences": pass_3plus,
            "median_sentence_count": median_sents,
            "pathological_rate": patho_rate,
            "PASS_3plus_sentences (>= 0.9)": pass_3plus >= 0.9,
            "PASS_median_count (>= 8)": median_sents >= 8,
            "PASS_pathological_rate (< 0.1)": patho_rate < 0.1,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Dump 5 sample trajectories with their segmentations.
    samples_path = out_dir / "samples.txt"
    with open(samples_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(stats_list[:5]):
            f.write(f"=== Trajectory {i+1} ===\n")
            f.write(f"raw text length: {len(s.raw_text)} chars\n")
            f.write(f"n_sentences: {s.n_sentences}\n")
            f.write(f"--- raw ---\n{s.raw_text}\n")
            f.write(f"--- segmented ({s.n_sentences} sentences) ---\n")
            for j, sent in enumerate(s.sentences):
                f.write(f"  [{j+1}] ({len(sent)} chars) {sent}\n")
            f.write("\n\n")

    # Histograms (text-only, no matplotlib dep).
    with open(out_dir / "histograms.txt", "w", encoding="utf-8") as f:
        f.write("Sentence count per trajectory:\n")
        for c in sorted(set(sent_counts)):
            n = sent_counts.count(c)
            f.write(f"  {c:4d}: {'#' * n} ({n})\n")
        f.write("\nSentence char length (binned):\n")
        bins = [(0, 10), (10, 30), (30, 60), (60, 100), (100, 200),
                (200, 400), (400, 600), (600, 10000)]
        for lo, hi in bins:
            n = sum(1 for L in all_char_lens if lo <= L < hi)
            label = f"{lo}-{hi}" if hi < 10000 else f">={lo}"
            f.write(f"  {label:>10}: {'#' * (n // 2)} ({n})\n")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/group/40143/howu/llms/Qwen3-1.7B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", default="./phase0_results")
    ap.add_argument("--max_problems", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--cached_rollouts", default=None,
                    help="Optional path to a JSONL of cached rollouts to skip generation.")
    ap.add_argument("--save_rollouts", default=None,
                    help="Optional path to save generated rollouts as JSONL.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase0] loading problems from {args.data_path}")
    problems = load_problems(args.data_path, args.max_problems)
    print(f"[phase0] loaded {len(problems)} problems")

    if args.cached_rollouts and os.path.exists(args.cached_rollouts):
        print(f"[phase0] loading cached rollouts from {args.cached_rollouts}")
        rollouts = []
        with open(args.cached_rollouts, "r", encoding="utf-8") as f:
            for line in f:
                rollouts.append(json.loads(line)["rollout"])
        rollouts = rollouts[:len(problems)]
    else:
        print(f"[phase0] generating rollouts...")
        rollouts = generate_rollouts(
            args.model_path, problems,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if args.save_rollouts:
            with open(args.save_rollouts, "w", encoding="utf-8") as f:
                for problem, text in zip(problems, rollouts):
                    f.write(json.dumps({"id": problem["id"], "rollout": text}) + "\n")
            print(f"[phase0] saved rollouts to {args.save_rollouts}")

    print(f"[phase0] segmenting {len(rollouts)} rollouts...")
    stats_list = [segment(r) for r in rollouts]

    print(f"[phase0] writing report to {out_dir}")
    summary = report(stats_list, out_dir)

    # Print key results to stdout.
    print("\n" + "=" * 60)
    print("PHASE 0.1 RESULTS")
    print("=" * 60)
    print(json.dumps(summary["pass_criteria"], indent=2))
    print(f"\nFull report: {out_dir / 'summary.json'}")
    print(f"Sample dump: {out_dir / 'samples.txt'}")
    print(f"Histograms:  {out_dir / 'histograms.txt'}")


if __name__ == "__main__":
    main()
