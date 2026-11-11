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
r"""Block-OPD gradient diagnostics at a fixed student checkpoint.

Three stages, each run once per GPU shard (see run_diag.sh):

  generate   sample on-policy rollouts with vLLM from the training prompt distribution
  analyze    per rollout: teacher top-k, the direct / score gradients for every block
             size (exact, single GPU, sketched for batch-level statistics) and nested
             branch sampling at random response positions
  summarize  merge the shards and print the Q1 / Q2 / Q3 tables

Usage (from the repo root):
  python -m tools.bopd_diagnostics.bopd_diag generate --student <hf> --out <dir> \
      --shard 0 --num_shards 8 [hydra overrides for the training config ...]
  python -m tools.bopd_diagnostics.bopd_diag analyze --student <hf> --teacher <hf> \
      --out <dir> --shard 0 --num_shards 8
  python -m tools.bopd_diagnostics.bopd_diag summarize --out <dir>
"""

import argparse
import glob
import json
import math
import os
import time
from collections import defaultdict
from typing import Any

import torch

from tools.bopd_diagnostics import diag_lib
from tools.bopd_diagnostics.diag_lib import (
    GradSketch,
    add_kl_to_go_stats,
    autocast_ctx,
    branch_continuations,
    branch_kl_to_go,
    branch_score_stats,
    direct_logit_grad,
    kl_to_go_and_baseline,
    new_kl_to_go_stats,
    objective_backward,
    student_terms,
    summarize_branches,
    summarize_gradients,
    summarize_kl_to_go,
    teacher_topk,
)


def _shard_range(n: int, shard: int, num_shards: int) -> tuple[int, int]:
    return n * shard // num_shards, n * (shard + 1) // num_shards


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def load_prompts(
    config_path: str, overrides: list[str], num_samples: int, seed: int
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """num_samples random training prompts, tokenized exactly as run_distillation_math does."""
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.processors import math_hf_data_processor
    from nemo_rl.utils.config import load_config, parse_hydra_overrides

    # registered at import time by examples/run_distillation_math.py; the YAML uses it
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
    cfg = load_config(config_path)
    if overrides:
        cfg = parse_hydra_overrides(cfg, overrides)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    tokenizer = get_tokenizer(cfg["policy"]["tokenizer"])
    data_cfg = cfg["data"]
    spec = TaskDataSpec(
        task_name="math",
        prompt_file=data_cfg["prompt_file"],
        system_prompt_file=data_cfg["system_prompt_file"],
    )
    # seed 42 as in run_distillation_math.setup_data (decides the validation holdout)
    data = load_response_dataset(data_cfg, 42)
    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        spec,
        math_hf_data_processor,
        max_seq_length=data_cfg["max_input_seq_length"],
    )
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    prompts = []
    for idx in order.tolist():
        datum = dataset[idx]
        if datum["loss_multiplier"] == 0:  # over-long prompt, masked out in training
            continue
        prompts.append(torch.cat([m["token_ids"] for m in datum["message_log"]]))
        if len(prompts) == num_samples:
            break
    return prompts, cfg


def cmd_generate(args: argparse.Namespace, overrides: list[str]) -> None:
    # importing nemo_rl initializes CUDA in this process, so vLLM's engine
    # subprocess must be spawned rather than forked
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM, SamplingParams

    prompts, cfg = load_prompts(args.config, overrides, args.num_samples, args.seed)
    gen_cfg = cfg["policy"]["generation"]
    tokenizer_name = cfg["policy"]["tokenizer"]["name"]
    lo, hi = _shard_range(len(prompts), args.shard, args.num_shards)
    llm = LLM(
        model=args.student,
        tokenizer=tokenizer_name,
        dtype="bfloat16",
        max_model_len=args.max_total_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    eos_id = llm.get_tokenizer().eos_token_id
    requests, params = [], []
    for sid in range(lo, hi):
        prompt = prompts[sid].tolist()
        requests.append({"prompt_token_ids": prompt})
        params.append(
            SamplingParams(
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                top_k=-1 if gen_cfg["top_k"] is None else gen_cfg["top_k"],
                max_tokens=args.max_total_len - len(prompt),
                # configure_generation_config defaults stop_token_ids to [eos]
                stop_token_ids=gen_cfg["stop_token_ids"] or [eos_id],
                seed=args.seed * 100_003 + sid,
            )
        )
    outputs = llm.generate(requests, params)
    records = []
    for sid, req, out in zip(range(lo, hi), requests, outputs):
        prompt = req["prompt_token_ids"]
        records.append(
            {
                "sample_id": sid,
                "prompt_len": len(prompt),
                "ids": torch.tensor(
                    prompt + list(out.outputs[0].token_ids), dtype=torch.int32
                ),
                "finish_reason": out.outputs[0].finish_reason,
            }
        )
    os.makedirs(os.path.join(args.out, "rollouts"), exist_ok=True)
    path = os.path.join(args.out, "rollouts", f"shard_{args.shard}.pt")
    torch.save({"records": records, "eos_id": eos_id, "args": vars(args)}, path)
    lengths = [len(r["ids"]) - r["prompt_len"] for r in records]
    print(
        f"[shard {args.shard}] {len(records)} rollouts, mean response length "
        f"{sum(lengths) / max(len(lengths), 1):.0f} -> {path}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _variant_specs(block_sizes: list[int], block_modes: list[str]) -> list[tuple]:
    """(name, kl_coef, mode, k, use_baseline); "D" is the direct gradient."""
    specs = [("D", 1.0, None, None, False)]
    for mode in block_modes:
        for k in block_sizes:
            specs.append((f"S:{mode}:{k}", 0.0, mode, k, False))
            specs.append((f"Sbl:{mode}:{k}", 0.0, mode, k, True))
    return specs


def _branch_rollout(
    args,
    student,
    teacher,
    s_cache,
    t_cache,
    h,
    ids,
    prompt_len,
    t_vals,
    t_idx,
    kl,
    ent,
    togo,
    eos_id,
    student_vocab,
    sample_id,
) -> list[dict[str, Any]]:
    device = ids.device
    length = ids.shape[0]
    horizon = max(args.block_sizes) - 1
    n1, n2 = args.branch_n1, args.branch_n2
    cpu_gen = torch.Generator().manual_seed(args.seed * 7_919 + sample_id)
    gpu_gen = torch.Generator(device=device).manual_seed(args.seed * 7_919 + sample_id)
    candidates = torch.arange(prompt_len - 1, length - 1)
    pick = torch.randperm(len(candidates), generator=cpu_gen)[: args.branch_positions]
    head = student.get_output_embeddings()
    out = []
    for i in sorted(candidates[pick].tolist()):
        with torch.no_grad(), autocast_ctx(device):
            logits = head(h[i : i + 1])
        logits = logits.float()[0]
        probs = torch.softmax(logits, dim=-1)
        direct = direct_logit_grad(logits, t_vals[i], t_idx[i])
        first = torch.multinomial(probs, n1, replacement=True, generator=gpu_gen)
        # extra last row replays the real continuation: its KL-to-go must match training's
        real = ids[i + 1 : i + 1 + horizon]
        replay = torch.full((horizon,), eos_id, dtype=torch.long, device=device)
        replay[: real.numel()] = real
        next_tokens = torch.cat([first.repeat_interleave(n2), replay[:1]])
        forced = torch.full(
            (next_tokens.numel(), horizon), -1, dtype=torch.long, device=device
        )
        forced[-1] = replay
        kls, alives = [], []
        for s in range(0, next_tokens.numel(), args.branch_chunk):
            e = min(next_tokens.numel(), s + args.branch_chunk)
            f_kl, alive = branch_continuations(
                student,
                teacher,
                s_cache,
                t_cache,
                i,
                next_tokens[s:e],
                forced[s:e],
                horizon,
                eos_id,
                args.max_total_len,
                args.topk,
                student_vocab,
                gpu_gen,
            )
            kls.append(f_kl)
            alives.append(alive)
        future_kl, alive = torch.cat(kls), torch.cat(alives)
        resp_offset = i - (prompt_len - 1)
        stats = {}
        for key, (r_roll, b_roll) in togo.items():
            mode, k = key.split(":")
            k = int(k)
            r_branch = branch_kl_to_go(future_kl[:-1], alive[:-1], k, mode, resp_offset)
            r_replay = branch_kl_to_go(future_kl[-1:], alive[-1:], k, mode, resp_offset)
            st = branch_score_stats(
                r_branch.view(n1, n2), first, probs, direct, float(b_roll[i])
            )
            st["replay_err"] = abs(float(r_replay[0]) - float(r_roll[i]))
            st["R_rollout"] = float(r_roll[i])
            stats[key] = st
        out.append(
            {
                "sample_id": sample_id,
                "position": i,
                "resp_offset": resp_offset,
                "entropy": float(ent[i]),
                "kl": float(kl[i]),
                "stats": stats,
            }
        )
    return out


def cmd_analyze(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = torch.load(
        os.path.join(args.out, "rollouts", f"shard_{args.shard}.pt"), weights_only=False
    )
    records, eos_id = data["records"], data["eos_id"]

    student = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.float32, attn_implementation=args.attn
    ).to(device)
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if args.fp32:
        diag_lib.BF16_AUTOCAST = False
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.float32 if args.fp32 else torch.bfloat16,
        attn_implementation=args.attn,
    ).to(device)
    teacher.eval().requires_grad_(False)
    student_vocab = student.get_output_embeddings().weight.shape[0]
    params = [p for p in student.parameters() if p.requires_grad]

    specs = _variant_specs(args.block_sizes, args.block_modes)
    variants = [s[0] for s in specs]
    sketch = GradSketch(args.sketch_dim, args.sketch_seed)
    sketches: dict[int, torch.Tensor] = {}
    tokens: dict[int, float] = defaultdict(float)
    samples: list[dict[str, Any]] = []
    togo_keys = [f"{m}:{k}" for m in args.block_modes for k in args.block_sizes]
    togo_stats = {key: new_kl_to_go_stats() for key in togo_keys}
    kl_stats = torch.zeros(3, dtype=torch.float64)  # n, sum KL, sum KL^2
    branches: list[dict[str, Any]] = []

    start = time.time()
    for n_done, rec in enumerate(records):
        ids = rec["ids"].long().to(device)
        length, prompt_len = ids.shape[0], rec["prompt_len"]
        if length - prompt_len < 1:
            continue
        mask = torch.zeros(length - 1, device=device)
        mask[prompt_len - 1 :] = 1
        do_branch = not args.skip_branch and n_done < args.branch_rollouts

        t_vals, t_idx, t_cache = teacher_topk(
            teacher, ids, args.topk, args.chunk, do_branch, student_vocab
        )
        student.eval()
        kl, _, ent, h, s_cache = student_terms(
            student, ids, t_vals, t_idx, args.chunk, do_branch
        )
        sel = mask > 0
        kl_stats += torch.tensor(
            [float(sel.sum()), float(kl[sel].sum()), float((kl[sel] ** 2).sum())],
            dtype=torch.float64,
        )
        togo = {}
        for key in togo_keys:
            mode, k = key.split(":")
            r, b = kl_to_go_and_baseline(kl, mask, int(k), mode)
            togo[key] = (r, b)
            add_kl_to_go_stats(togo_stats[key], r, b, mask)

        batch = rec["sample_id"] // args.batch_size
        if not args.skip_grads:
            student.train()
            if batch not in sketches:
                sketches[batch] = torch.zeros(
                    len(specs), args.sketch_dim, device=device
                )
            grad_direct = None
            n2, dot_direct = [], []
            for vi, (_, kl_coef, mode, k, use_bl) in enumerate(specs):
                weight = None
                if mode is not None:
                    r, b = togo[f"{mode}:{k}"]
                    weight = r - b if use_bl else r
                objective_backward(
                    student, ids, mask, t_vals, t_idx, kl_coef, weight, args.chunk
                )
                grads = [p.grad for p in params]
                if vi == 0:
                    grad_direct = [
                        None if g is None else g.detach().clone() for g in grads
                    ]
                sq, dot = 0.0, 0.0
                for g, gd in zip(grads, grad_direct):
                    if g is None:
                        continue
                    g64 = g.double()
                    sq += float((g64 * g64).sum())
                    if gd is not None:
                        dot += float((g64 * gd.double()).sum())
                    del g64
                n2.append(sq)
                dot_direct.append(dot)
                sketch.add_(sketches[batch][vi], grads)
                for p in params:
                    p.grad = None
            del grad_direct
            samples.append(
                {
                    "sample_id": rec["sample_id"],
                    "batch": batch,
                    "n2": n2,
                    "dotD": dot_direct,
                }
            )
            tokens[batch] += float(mask.sum())

        if do_branch:
            student.eval()
            branches += _branch_rollout(
                args,
                student,
                teacher,
                s_cache,
                t_cache,
                h,
                ids,
                prompt_len,
                t_vals,
                t_idx,
                kl,
                ent,
                togo,
                eos_id,
                student_vocab,
                rec["sample_id"],
            )
        del t_cache, s_cache, h
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[shard {args.shard}] {n_done + 1}/{len(records)} len={length} "
            f"elapsed={time.time() - start:.0f}s",
            flush=True,
        )

    os.makedirs(os.path.join(args.out, "analysis"), exist_ok=True)
    path = os.path.join(args.out, "analysis", f"shard_{args.shard}.pt")
    torch.save(
        {
            "args": vars(args),
            "variants": variants,
            "sketches": {b: s.cpu() for b, s in sketches.items()},
            "tokens": dict(tokens),
            "samples": samples,
            "kl_to_go": togo_stats,
            "kl": kl_stats,
            "branches": branches,
        },
        path,
    )
    print(f"[shard {args.shard}] done -> {path}", flush=True)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.3g}"
    return str(x)


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _render(summary: dict[str, Any]) -> str:
    out = [f"# Block-OPD gradient diagnostics\n\nstudent: `{summary['student']}`\n"]
    kl = summary["per_token_kl"]
    out.append(
        f"per-token KL over {kl['positions']:.0f} response positions: "
        f"mean {_fmt(kl['mean'])}, var {_fmt(kl['var'])}\n"
    )
    grads = summary.get("gradients")
    if grads:
        out.append(
            f"## Q1 score vs direct gradient (parameter space, {grads['num_batches']} "
            "batches)\n\n'batch' = per-batch gradient G (as one training step sees it, "
            "averaged over batches); 'E' = expectation E[G] estimated from cross-batch "
            "products (noise removed; needs >= 2 batches). alpha / perp: G_S = alpha "
            "G_D + perp.\n"
        )
        rows = []
        for name, m in grads["q1"].items():
            rows.append(
                [
                    name,
                    m["ratio_batch"],
                    m["cos_batch"],
                    m["alpha_batch"],
                    m["perp_ratio_batch"],
                    m["ratio_expect"],
                    m["cos_expect"],
                    m["signal_fraction"],
                    m["direct_norm_batch"],
                    m["total_norm_batch"],
                    m["total_cos_batch"],
                    m["sample_ratio_median"],
                    m["sample_cos_median"],
                ]
            )
        out.append(
            _table(
                [
                    "score term",
                    "|G_S|/|G_D|",
                    "cos(G_D,G_S)",
                    "alpha",
                    "|perp|/|G_D|",
                    "|E G_S|/|E G_D|",
                    "cos(E G_D,E G_S)",
                    "signal frac |G_S|^2",
                    "|G_D|",
                    "|G_D+G_S|",
                    "cos(G_D+G_S,G_D)",
                    "sample median ratio",
                    "sample median cos",
                ],
                rows,
            )
        )
    out.append(
        "\n\n## Q2 does y_{i+1} change the future KL? (nested branches)\n\n"
        "cond var = E Var(R_i | y_<=i) = between + within; between = Var over "
        "y_{i+1} of E[R_i | y_<=i+1] (the part the score term can use); within = "
        "noise from y_{i+2..}. |s*|/|d|, cos: expected score gradient vs the direct "
        "gradient in logit space at the branched positions.\n"
    )
    rows = []
    for key, m in summary["branches"].items():
        if not m:
            continue
        rows.append(
            [
                key,
                m["positions"],
                m["mean_R"],
                summary["kl_to_go"][key]["var_R"],
                m["cond_var"],
                m["between"],
                m["within"],
                m["between_share"],
                m["signal_ratio"],
                m["signal_cos"],
                m["replay_mean_rel_err"],
                m["replay_max_rel_err"],
            ]
        )
    out.append(
        _table(
            [
                "block",
                "positions",
                "E[R]",
                "Var(R) all",
                "cond var",
                "between",
                "within",
                "between share",
                "|s*|/|d|",
                "cos(s*,d)",
                "replay err mean",
                "replay err max",
            ],
            rows,
        )
    )
    rows = []
    for key, m in summary["branches"].items():
        for b in m.get("entropy_bins", []):
            rows.append(
                [
                    key,
                    f"{b['entropy_lo']:.2f}-{b['entropy_hi']:.2f}",
                    b["positions"],
                    b["between"],
                    b["within"],
                    b["signal_ratio"],
                    b["snr_none"],
                ]
            )
    if rows:
        out.append("\n\nby student entropy at position i:\n")
        out.append(
            _table(
                [
                    "block",
                    "entropy",
                    "positions",
                    "between",
                    "within",
                    "|s*|/|d|",
                    "SNR (no baseline)",
                ],
                rows,
            )
        )
    out.append(
        "\n\n## Q3 does the baseline reduce variance?\n\nKL-to-go over all response "
        "positions (the score weight is R - b):\n"
    )
    rows = [
        [
            key,
            m["var_R"],
            m["var_R_minus_b"],
            m["var_ratio"],
            m["second_moment_R"],
            m["second_moment_R_minus_b"],
            m["second_moment_ratio"],
        ]
        for key, m in summary["kl_to_go"].items()
        if m
    ]
    out.append(
        _table(
            ["block", "Var(R)", "Var(R-b)", "ratio", "E R^2", "E (R-b)^2", "ratio"],
            rows,
        )
    )
    out.append(
        "\n\nper-position score-gradient noise in logit space, tr Var of "
        "(e_y - q)(R - b); oracle b = E[R | y_<=i]:\n"
    )
    rows = []
    for key, m in summary["branches"].items():
        if m:
            rows.append(
                [
                    key,
                    m["noise_none"],
                    m["noise_hist"],
                    m["noise_oracle"],
                    m["noise_hist_over_none"],
                    m["noise_oracle_over_none"],
                    m["snr_none"],
                    m["snr_hist"],
                    m["snr_oracle"],
                ]
            )
    out.append(
        _table(
            [
                "block",
                "none",
                "history b",
                "oracle b",
                "hist/none",
                "oracle/none",
                "SNR none",
                "SNR hist",
                "SNR oracle",
            ],
            rows,
        )
    )
    if grads:
        out.append(
            "\n\nparameter space, per batch gradient G: signal = |E G|^2, noise = "
            "tr Var(G) (within: from per-sample gradients; cross: from batch spread):\n"
        )
        rows = [
            [
                name,
                m["batch_norm"],
                m["signal_sq"],
                m["noise_within"],
                m["noise_cross"],
                m["snr_within"],
                m["snr_cross"],
                m["cross_batch_cos"],
            ]
            for name, m in grads["q3"].items()
        ]
        out.append(
            _table(
                [
                    "gradient",
                    "|G|",
                    "|E G|^2",
                    "noise within",
                    "noise cross",
                    "SNR within",
                    "SNR cross",
                    "cross-batch cos",
                ],
                rows,
            )
        )
    return "\n".join(out) + "\n"


def cmd_summarize(args: argparse.Namespace) -> None:
    paths = sorted(glob.glob(os.path.join(args.out, "analysis", "shard_*.pt")))
    assert paths, f"no analysis shards under {args.out}/analysis"
    shards = [torch.load(p, weights_only=False) for p in paths]
    variants = shards[0]["variants"]
    sketches: dict[int, torch.Tensor] = {}
    tokens: dict[int, float] = defaultdict(float)
    samples, branches = [], []
    togo = {k: torch.zeros_like(v) for k, v in shards[0]["kl_to_go"].items()}
    kl = torch.zeros(3, dtype=torch.float64)
    for sh in shards:
        assert sh["variants"] == variants, "shards analyzed with different settings"
        for b, s in sh["sketches"].items():
            sketches[b] = sketches[b] + s if b in sketches else s.clone()
        for b, t in sh["tokens"].items():
            tokens[b] += t
        samples += sh["samples"]
        branches += sh["branches"]
        for k, v in sh["kl_to_go"].items():
            togo[k] += v
        kl += sh["kl"]
    n, s, s2 = kl.tolist()
    summary: dict[str, Any] = {
        "student": shards[0]["args"]["student"],
        "num_samples": len(samples),
        "per_token_kl": {"positions": n, "mean": s / n, "var": s2 / n - (s / n) ** 2},
        "kl_to_go": {k: summarize_kl_to_go(v) for k, v in togo.items()},
        "branches": {k: summarize_branches(branches, k) for k in togo},
    }
    if sketches:
        summary["gradients"] = summarize_gradients(variants, sketches, tokens, samples)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    text = _render(summary)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", required=True, help="output directory")
        p.add_argument("--shard", type=int, default=0)
        p.add_argument("--num_shards", type=int, default=1)
        p.add_argument("--seed", type=int, default=1234)
        p.add_argument("--max_total_len", type=int, default=16384)

    g = sub.add_parser("generate", help="sample rollouts (hydra overrides go last)")
    common(g)
    g.add_argument("--student", required=True, help="HF checkpoint of the student")
    g.add_argument("--config", default="examples/configs/distillation_math.yaml")
    g.add_argument("--num_samples", type=int, default=512)
    g.add_argument("--gpu_memory_utilization", type=float, default=0.85)

    a = sub.add_parser("analyze", help="gradients and branches for one shard")
    common(a)
    a.add_argument("--student", required=True)
    a.add_argument("--teacher", required=True)
    a.add_argument("--block_sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    a.add_argument("--block_modes", nargs="+", default=["sliding"])
    a.add_argument("--topk", type=int, default=64, help="distillation.topk_logits_k")
    a.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="rollouts per training step; sample_id // batch_size is the batch",
    )
    a.add_argument("--sketch_dim", type=int, default=1 << 23)
    a.add_argument("--sketch_seed", type=int, default=0)
    a.add_argument(
        "--chunk", type=int, default=2048, help="positions per LM-head chunk"
    )
    a.add_argument("--attn", default="sdpa")
    a.add_argument("--device", default="cuda")
    a.add_argument(
        "--fp32",
        action="store_true",
        help="no bf16 autocast, fp32 teacher: checks that replay errors are bf16 rounding",
    )
    a.add_argument("--skip_grads", action="store_true")
    a.add_argument("--skip_branch", action="store_true")
    a.add_argument("--branch_rollouts", type=int, default=8, help="per shard")
    a.add_argument("--branch_positions", type=int, default=16, help="per rollout")
    a.add_argument("--branch_n1", type=int, default=8, help="sampled y_{i+1}")
    a.add_argument("--branch_n2", type=int, default=4, help="continuations per y_{i+1}")
    a.add_argument("--branch_chunk", type=int, default=8, help="branches per forward")

    s = sub.add_parser("summarize", help="merge shards, write summary.{json,md}")
    s.add_argument("--out", required=True)

    args, overrides = parser.parse_known_args()
    if args.cmd == "generate":
        cmd_generate(args, overrides)
    else:
        if overrides:
            parser.error(f"unrecognized arguments: {overrides}")
        if args.cmd == "analyze":
            cmd_analyze(args)
        else:
            cmd_summarize(args)


if __name__ == "__main__":
    main()
