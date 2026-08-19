#!/usr/bin/env python3
"""Generate traces, extract frozen activations, and fit correctness probes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research.hidden_state_probe.core import (
    append_jsonl,
    binary_metrics,
    grouped_problem_split,
    prefix_checkpoints,
    read_jsonl,
    resolve_layer_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["generate", "extract", "train"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model", default="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
    )
    parser.add_argument("--dataset", default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--prompt-file", type=Path, default=Path("examples/prompts/cot.txt"))
    parser.add_argument("--num-problems", type=int, default=1000)
    parser.add_argument("--trajectories-per-problem", type=int, default=2)
    parser.add_argument("--validation-holdout-size", type=int, default=1000)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-generation-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-checkpoints", type=int, default=8)
    parser.add_argument("--min-prefix-tokens", type=int, default=256)
    parser.add_argument("--min-remaining-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--layer-fractions", default="0.5,0.75,0.9,1.0")
    parser.add_argument("--extraction-chunk-size", type=int, default=16)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--probe-learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--probe-l2", type=float, default=1.0e-4)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--probe-patience", type=int, default=30)
    return parser.parse_args()


def trajectory_path(args: argparse.Namespace) -> Path:
    return args.output_dir / (
        f"trajectories.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl"
    )


def all_trajectories(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    pattern = f"trajectories.shard-*-of-{args.num_shards:02d}.jsonl"
    for path in sorted(args.output_dir.glob(pattern)):
        records.extend(read_jsonl(path))
    return records


def load_tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def load_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.dataset_split)
    shuffled = dataset.shuffle(seed=args.validation_seed)
    if args.validation_holdout_size + args.num_problems > len(shuffled):
        raise ValueError("holdout and requested problems exceed dataset size")
    training = shuffled.select(range(args.validation_holdout_size, len(shuffled)))
    selected = training.shuffle(seed=args.seed).select(range(args.num_problems))
    return [
        {
            "problem_id": index,
            "problem": str(selected[index]["problem"]),
            "ground_truth": str(selected[index]["answer"]),
        }
        for index in range(len(selected))
    ]


def render_prompt(problem: str, tokenizer: Any, template: str) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": template.format(problem)}],
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    return list(token_ids)


def verify_math(ground_truth: str, response: str) -> float:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    if not hasattr(verify_math, "metric"):
        verify_math.metric = math_metric(  # type: ignore[attr-defined]
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
    try:
        score, _ = verify_math.metric(  # type: ignore[attr-defined]
            [f"\\boxed{{{ground_truth}}}"], [response]
        )
        return float(score)
    except (Exception, TimeoutException):
        return 0.0


def batched(items: list[Any], size: int) -> list[list[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def generate(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    path = trajectory_path(args)
    completed = defaultdict(set)
    for record in read_jsonl(path):
        completed[int(record["problem_id"])].add(int(record["trajectory_index"]))

    tokenizer = load_tokenizer(args.model)
    template = args.prompt_file.read_text(encoding="utf-8").strip()
    examples = [
        example
        for example in load_examples(args)
        if example["problem_id"] % args.num_shards == args.shard_index
        and len(completed[example["problem_id"]]) < args.trajectories_per_problem
    ]
    if not examples:
        print("generate already complete")
        return

    for example in examples:
        example["prompt_token_ids"] = render_prompt(
            example["problem"], tokenizer, template
        )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    params = SamplingParams(
        n=args.trajectories_per_problem,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_generation_tokens,
        seed=args.seed + args.shard_index,
    )
    for batch in batched(examples, args.generation_batch_size):
        outputs = llm.generate(
            [{"prompt_token_ids": item["prompt_token_ids"]} for item in batch],
            params,
            use_tqdm=True,
        )
        records = []
        for example, request_output in zip(batch, outputs):
            for trajectory_index, output in enumerate(request_output.outputs):
                if trajectory_index in completed[example["problem_id"]]:
                    continue
                response_ids = list(output.token_ids)
                prefix_positions, probe_positions = prefix_checkpoints(
                    response_ids,
                    tokenizer,
                    response_text=output.text,
                    max_checkpoints=args.max_checkpoints,
                    min_prefix_tokens=args.min_prefix_tokens,
                    min_remaining_tokens=args.min_remaining_tokens,
                )
                records.append(
                    {
                        **example,
                        "trajectory_id": f"p{example['problem_id']}-r{trajectory_index}",
                        "trajectory_index": trajectory_index,
                        "response": output.text,
                        "response_token_ids": response_ids,
                        "response_tokens": len(response_ids),
                        "finish_reason": output.finish_reason,
                        "correct": verify_math(example["ground_truth"], output.text),
                        "checkpoint_prefix_positions": prefix_positions,
                        "checkpoint_probe_positions": probe_positions,
                    }
                )
        append_jsonl(path, records)


def transformer_layers(model: Any) -> Any:
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise TypeError("Could not find transformer layers on the loaded model")


def load_completed_chunks(manifest_path: Path) -> tuple[set[str], list[Path]]:
    completed: set[str] = set()
    chunks = []
    for record in read_jsonl(manifest_path):
        completed.update(record["trajectory_ids"])
        chunks.append(Path(record["chunk_file"]))
    return completed, chunks


def save_extraction_chunk(
    args: argparse.Namespace,
    chunk_index: int,
    rows: list[dict[str, Any]],
    layer_indices: list[int],
    manifest_path: Path,
) -> None:
    chunk_path = args.output_dir / (
        f"hidden.shard-{args.shard_index:02d}.chunk-{chunk_index:05d}.npz"
    )
    embeddings = np.concatenate([row["embeddings"] for row in rows], axis=0)
    np.savez(
        chunk_path,
        embeddings=embeddings.astype(np.float16),
        labels=np.concatenate([row["labels"] for row in rows]),
        problem_ids=np.concatenate([row["problem_ids"] for row in rows]),
        trajectory_ids=np.concatenate([row["trajectory_ids"] for row in rows]),
        prefix_tokens=np.concatenate([row["prefix_tokens"] for row in rows]),
        response_tokens=np.concatenate([row["response_tokens"] for row in rows]),
        checkpoint_indices=np.concatenate([row["checkpoint_indices"] for row in rows]),
        layer_indices=np.asarray(layer_indices, dtype=np.int32),
    )
    append_jsonl(
        manifest_path,
        [
            {
                "chunk_file": str(chunk_path),
                "trajectory_ids": [row["trajectory_id"] for row in rows],
            }
        ],
    )


def extract(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModel

    manifest_path = args.output_dir / f"extraction-manifest.shard-{args.shard_index:02d}.jsonl"
    completed, prior_chunks = load_completed_chunks(manifest_path)
    trajectories = [
        record
        for record in read_jsonl(trajectory_path(args))
        if record["trajectory_id"] not in completed
    ]
    if not trajectories:
        print("extract already complete")
        return

    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()
    layers = transformer_layers(model)
    fractions = [float(value) for value in args.layer_fractions.split(",")]
    layer_indices = resolve_layer_indices(len(layers), fractions)
    captures: dict[int, torch.Tensor] = {}
    current_positions: list[int] = []
    hooks = []

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            positions = torch.tensor(current_positions, device=hidden.device)
            captures[layer_index] = hidden[0, positions].detach()

        return hook

    for layer_index in layer_indices:
        hooks.append(layers[layer_index].register_forward_hook(make_hook(layer_index)))

    chunk_rows = []
    chunk_index = len(prior_chunks)
    try:
        for record_index, record in enumerate(trajectories, start=1):
            prompt_length = len(record["prompt_token_ids"])
            current_positions = [
                prompt_length - 1
                if position == 0
                else prompt_length + int(position) - 1
                for position in record["checkpoint_probe_positions"]
            ]
            input_ids = record["prompt_token_ids"] + record["response_token_ids"]
            captures.clear()
            with torch.inference_mode():
                model(
                    input_ids=torch.tensor([input_ids], device="cuda"),
                    use_cache=False,
                    return_dict=True,
                )
            embeddings = torch.stack(
                [captures[layer_index] for layer_index in layer_indices], dim=1
            ).float().cpu().numpy()
            count = len(record["checkpoint_prefix_positions"])
            chunk_rows.append(
                {
                    "trajectory_id": record["trajectory_id"],
                    "embeddings": embeddings,
                    "labels": np.full(count, int(record["correct"] > 0.5), dtype=np.int8),
                    "problem_ids": np.full(count, record["problem_id"], dtype=np.int32),
                    "trajectory_ids": np.asarray([record["trajectory_id"]] * count),
                    "prefix_tokens": np.asarray(
                        record["checkpoint_prefix_positions"], dtype=np.int32
                    ),
                    "response_tokens": np.full(count, record["response_tokens"], dtype=np.int32),
                    "checkpoint_indices": np.arange(count, dtype=np.int16),
                }
            )
            if len(chunk_rows) >= args.extraction_chunk_size:
                save_extraction_chunk(
                    args, chunk_index, chunk_rows, layer_indices, manifest_path
                )
                chunk_rows = []
                chunk_index += 1
            print(
                f"extracted {record_index}/{len(trajectories)} "
                f"trajectory={record['trajectory_id']} checkpoints={count}",
                flush=True,
            )
        if chunk_rows:
            save_extraction_chunk(
                args, chunk_index, chunk_rows, layer_indices, manifest_path
            )
    finally:
        for hook in hooks:
            hook.remove()


def load_probe_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    layer_indices: np.ndarray | None = None
    for shard_index in range(args.num_shards):
        manifest = args.output_dir / f"extraction-manifest.shard-{shard_index:02d}.jsonl"
        _, chunk_paths = load_completed_chunks(manifest)
        for path in chunk_paths:
            with np.load(path) as data:
                if layer_indices is None:
                    layer_indices = data["layer_indices"].copy()
                elif not np.array_equal(layer_indices, data["layer_indices"]):
                    raise ValueError("layer indices differ across extraction chunks")
                for key in (
                    "embeddings",
                    "labels",
                    "problem_ids",
                    "trajectory_ids",
                    "prefix_tokens",
                    "response_tokens",
                    "checkpoint_indices",
                ):
                    arrays[key].append(data[key].copy())
    if layer_indices is None:
        raise RuntimeError("no extraction chunks found")
    result = {key: np.concatenate(values) for key, values in arrays.items()}
    result["layer_indices"] = layer_indices
    return result


def fit_logistic_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, int]:
    import torch
    import torch.nn.functional as functional

    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1.0e-5] = 1.0
    train_x = (train_x.astype(np.float32) - mean) / std
    validation_x = (validation_x.astype(np.float32) - mean) / std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.from_numpy(train_x).to(device)
    y_train = torch.from_numpy(train_y.astype(np.float32)).to(device)
    x_validation = torch.from_numpy(validation_x).to(device)
    y_validation = torch.from_numpy(validation_y.astype(np.float32)).to(device)
    weight = torch.zeros(train_x.shape[1], device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=args.probe_learning_rate)
    best_loss = float("inf")
    best_weight = None
    best_bias = None
    best_epoch = 0
    stale_epochs = 0
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for epoch in range(args.probe_epochs):
        order = torch.randperm(len(x_train), generator=generator, device=device)
        for batch_indices in order.split(2048):
            logits = x_train[batch_indices] @ weight + bias
            loss = functional.binary_cross_entropy_with_logits(
                logits, y_train[batch_indices]
            ) + 0.5 * args.probe_l2 * weight.square().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            validation_loss = functional.binary_cross_entropy_with_logits(
                x_validation @ weight + bias, y_validation
            ).item()
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_weight = weight.detach().cpu().numpy().copy()
            best_bias = float(bias.detach().cpu())
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.probe_patience:
                break
    assert best_weight is not None and best_bias is not None
    return best_weight, best_bias, mean, std, best_epoch


def sigmoid(logits: np.ndarray) -> np.ndarray:
    positive = logits >= 0
    result = np.empty_like(logits, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def temporal_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    trajectory_ids: np.ndarray,
    checkpoint_indices: np.ndarray,
) -> dict[str, Any]:
    by_trajectory: dict[str, list[int]] = defaultdict(list)
    for row, trajectory_id in enumerate(trajectory_ids):
        by_trajectory[str(trajectory_id)].append(row)
    largest_drops: dict[int, list[float]] = {0: [], 1: []}
    for rows in by_trajectory.values():
        rows.sort(key=lambda row: checkpoint_indices[row])
        trajectory_scores = scores[rows]
        if len(trajectory_scores) < 2:
            continue
        label = int(labels[rows[0]])
        largest_drops[label].append(float(np.max(trajectory_scores[:-1] - trajectory_scores[1:])))
    return {
        "mean_largest_adjacent_drop_correct": float(np.mean(largest_drops[1]))
        if largest_drops[1]
        else None,
        "mean_largest_adjacent_drop_incorrect": float(np.mean(largest_drops[0]))
        if largest_drops[0]
        else None,
        "incorrect_drop_ge_0.2_rate": float(
            np.mean(np.asarray(largest_drops[0]) >= 0.2)
        )
        if largest_drops[0]
        else None,
        "correct_drop_ge_0.2_rate": float(np.mean(np.asarray(largest_drops[1]) >= 0.2))
        if largest_drops[1]
        else None,
    }


def within_problem_concordance(
    labels: np.ndarray,
    scores: np.ndarray,
    problem_ids: np.ndarray,
    checkpoint_indices: np.ndarray,
) -> dict[str, Any]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row, (problem_id, checkpoint_index) in enumerate(
        zip(problem_ids, checkpoint_indices)
    ):
        groups[(int(problem_id), int(checkpoint_index))].append(row)
    concordance = 0.0
    pairs = 0
    mixed_problems = set()
    for (problem_id, _), rows in groups.items():
        positive_scores = scores[rows][labels[rows] == 1]
        negative_scores = scores[rows][labels[rows] == 0]
        if not len(positive_scores) or not len(negative_scores):
            continue
        mixed_problems.add(problem_id)
        for positive_score in positive_scores:
            concordance += float(np.sum(positive_score > negative_scores))
            concordance += 0.5 * float(np.sum(positive_score == negative_scores))
            pairs += len(negative_scores)
    return {
        "concordance": concordance / pairs if pairs else None,
        "num_pairs": pairs,
        "num_mixed_problems": len(mixed_problems),
    }


def train(args: argparse.Namespace) -> None:
    import torch

    data = load_probe_dataset(args)
    splits = grouped_problem_split(data["problem_ids"], seed=args.seed)
    layer_results = []
    fitted = []
    for layer_axis, layer_index in enumerate(data["layer_indices"]):
        masks = {split: splits == split for split in ("train", "validation", "test")}
        weight, bias, mean, std, epochs = fit_logistic_probe(
            data["embeddings"][masks["train"], layer_axis],
            data["labels"][masks["train"]],
            data["embeddings"][masks["validation"], layer_axis],
            data["labels"][masks["validation"]],
            args,
        )
        scores = sigmoid(
            ((data["embeddings"][:, layer_axis].astype(np.float32) - mean) / std)
            @ weight
            + bias
        )
        metrics = {
            split: binary_metrics(data["labels"][mask], scores[mask])
            for split, mask in masks.items()
        }
        layer_results.append(
            {
                "layer_index": int(layer_index),
                "epochs": epochs,
                "metrics": metrics,
            }
        )
        fitted.append((weight, bias, mean, std, scores))
        print(
            f"layer={layer_index} validation_auc={metrics['validation']['roc_auc']} "
            f"test_auc={metrics['test']['roc_auc']}",
            flush=True,
        )

    best_axis = max(
        range(len(layer_results)),
        key=lambda axis: layer_results[axis]["metrics"]["validation"]["roc_auc"]
        if layer_results[axis]["metrics"]["validation"]["roc_auc"] is not None
        else -1.0,
    )
    weight, bias, mean, std, scores = fitted[best_axis]
    best_layer = int(data["layer_indices"][best_axis])
    test_mask = splits == "test"
    checkpoint_metrics = {}
    curve = {}
    for checkpoint_index in np.unique(data["checkpoint_indices"]):
        mask = test_mask & (data["checkpoint_indices"] == checkpoint_index)
        checkpoint_metrics[str(int(checkpoint_index))] = binary_metrics(
            data["labels"][mask], scores[mask]
        )
        curve[str(int(checkpoint_index))] = {
            "correct_mean_score": float(scores[mask & (data["labels"] == 1)].mean())
            if np.any(mask & (data["labels"] == 1))
            else None,
            "incorrect_mean_score": float(scores[mask & (data["labels"] == 0)].mean())
            if np.any(mask & (data["labels"] == 0))
            else None,
        }

    prefix_fractions = data["prefix_tokens"] / np.maximum(1, data["response_tokens"])
    progress_metrics = {}
    for name, low, high in (
        ("empty_prefix", 0.0, 0.0),
        ("0_to_25pct", 0.0, 0.25),
        ("25_to_50pct", 0.25, 0.5),
        ("50_to_75pct", 0.5, 0.75),
        ("75_to_100pct", 0.75, 1.01),
    ):
        if name == "empty_prefix":
            mask = test_mask & (data["prefix_tokens"] == 0)
        else:
            mask = test_mask & (prefix_fractions > low) & (prefix_fractions <= high)
        progress_metrics[name] = binary_metrics(data["labels"][mask], scores[mask])

    trajectory_first_rows = np.unique(data["trajectory_ids"], return_index=True)[1]
    summary = {
        "num_rows": int(len(data["labels"])),
        "num_problems": int(len(np.unique(data["problem_ids"]))),
        "num_trajectories": int(len(np.unique(data["trajectory_ids"]))),
        "trajectory_accuracy": float(data["labels"][trajectory_first_rows].mean()),
        "best_layer": best_layer,
        "layer_results": layer_results,
        "test_metrics_by_checkpoint": checkpoint_metrics,
        "test_metrics_by_progress": progress_metrics,
        "test_score_curve": curve,
        "test_within_problem_concordance": within_problem_concordance(
            data["labels"][test_mask],
            scores[test_mask],
            data["problem_ids"][test_mask],
            data["checkpoint_indices"][test_mask],
        ),
        "test_temporal_metrics": temporal_summary(
            data["labels"][test_mask],
            scores[test_mask],
            data["trajectory_ids"][test_mask],
            data["checkpoint_indices"][test_mask],
        ),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "probe_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "layer_index": best_layer,
            "weight": weight,
            "bias": bias,
            "feature_mean": mean,
            "feature_std": std,
        },
        args.output_dir / "best_logistic_probe.pt",
    )
    with (args.output_dir / "probe_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "problem_id",
                "trajectory_id",
                "split",
                "label",
                "checkpoint_index",
                "prefix_tokens",
                "response_tokens",
                "prefix_fraction",
                "score",
            ]
        )
        for row in range(len(scores)):
            writer.writerow(
                [
                    int(data["problem_ids"][row]),
                    data["trajectory_ids"][row],
                    splits[row],
                    int(data["labels"][row]),
                    int(data["checkpoint_indices"][row]),
                    int(data["prefix_tokens"][row]),
                    int(data["response_tokens"][row]),
                    float(data["prefix_tokens"][row] / max(1, data["response_tokens"][row])),
                    float(scores[row]),
                ]
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    {"generate": generate, "extract": extract, "train": train}[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
