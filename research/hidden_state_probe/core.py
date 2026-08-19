"""Pure helpers for hidden-state correctness probing."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_evenly_spaced(items: Sequence[Any], limit: int) -> list[Any]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in dict.fromkeys(indices)]


def find_answer_start_token(response: str, tokenizer: Any) -> int | None:
    markers = ("\\boxed{", "<answer>", "Final Answer", "final answer")
    positions = [response.find(marker) for marker in markers]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return None
    return len(
        tokenizer(response[: min(positions)], add_special_tokens=False)["input_ids"]
    )


def prefix_checkpoints(
    token_ids: Sequence[int],
    tokenizer: Any,
    *,
    response_text: str,
    max_checkpoints: int,
    min_prefix_tokens: int,
    min_remaining_tokens: int,
) -> tuple[list[int], list[int]]:
    """Return context-prefix and last-content-token positions.

    Positions are response-relative token counts. Zero denotes the empty response
    prefix, whose probe representation is the final prompt token.
    """
    if max_checkpoints <= 1:
        return [0], [0]

    answer_start = find_answer_start_token(response_text, tokenizer)
    upper = len(token_ids) - min_remaining_tokens
    if answer_start is not None:
        upper = min(upper, answer_start)
    if upper < min_prefix_tokens:
        return [0], [0]

    paragraph_boundaries: list[tuple[int, int]] = []
    line_boundaries: list[tuple[int, int]] = []
    last_content_at_position: list[int] = [0]
    trailing_newlines = 0
    last_content_position = 0
    for position, token_id in enumerate(token_ids, start=1):
        piece = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        saw_newline = False
        for char in piece:
            if char == "\n":
                saw_newline = True
                trailing_newlines += 1
            elif not char.isspace():
                trailing_newlines = 0
                last_content_position = position
        last_content_at_position.append(last_content_position)
        if not min_prefix_tokens <= position <= upper or last_content_position == 0:
            continue
        if trailing_newlines >= 2:
            paragraph_boundaries.append((position, last_content_position))
            trailing_newlines = 0
        elif saw_newline:
            line_boundaries.append((position, last_content_position))

    known = {prefix for prefix, _ in paragraph_boundaries}
    line_boundaries = [item for item in line_boundaries if item[0] not in known]
    count = max_checkpoints - 1
    targets = [
        round(
            min_prefix_tokens
            + index * (upper - min_prefix_tokens) / max(1, count - 1)
        )
        for index in range(count)
    ]
    selected = []
    unused_paragraphs = set(paragraph_boundaries)
    unused_lines = set(line_boundaries)
    for target in targets:
        choice = None
        for pool in (unused_paragraphs, unused_lines):
            nearest = min(pool, key=lambda item: abs(item[0] - target)) if pool else None
            if nearest is not None and abs(nearest[0] - target) <= 256:
                choice = nearest
                pool.remove(nearest)
                break
        if choice is None:
            probe_position = last_content_at_position[target]
            choice = (target, probe_position or target)
        selected.append(choice)
    selected = sorted(set(selected))
    return [0] + [item[0] for item in selected], [0] + [item[1] for item in selected]


def grouped_problem_split(
    problem_ids: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> np.ndarray:
    """Assign rows to train/validation/test without splitting a problem."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")
    unique = np.unique(problem_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    train_end = round(len(unique) * train_fraction)
    validation_end = train_end + round(len(unique) * validation_fraction)
    mapping = {
        int(problem_id): split
        for split, values in (
            ("train", unique[:train_end]),
            ("validation", unique[train_end:validation_end]),
            ("test", unique[validation_end:]),
        )
        for problem_id in values
    }
    return np.asarray([mapping[int(problem_id)] for problem_id in problem_ids])


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    precision = np.cumsum(ordered_labels) / np.arange(1, len(labels) + 1)
    return float(precision[ordered_labels == 1].sum() / positives)


def expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, bins: int = 10
) -> float:
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (scores >= low) & (scores < high if index < bins - 1 else scores <= high)
        if mask.any():
            error += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(scores[mask].mean())
            )
    return error


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    predictions = scores >= 0.5
    eps = 1.0e-7
    clipped = np.clip(scores, eps, 1.0 - eps)
    return {
        "num_rows": int(len(labels)),
        "positive_rate": float(labels.mean()) if len(labels) else None,
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "accuracy_at_0.5": float((predictions == labels).mean()) if len(labels) else None,
        "log_loss": float(
            -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean()
        )
        if len(labels)
        else None,
        "brier_score": float(np.square(scores - labels).mean()) if len(labels) else None,
        "ece_10_bin": expected_calibration_error(labels, scores) if len(labels) else None,
    }


def resolve_layer_indices(num_layers: int, fractions: Sequence[float]) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    indices = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("layer fractions must be in (0, 1]")
        indices.append(min(num_layers - 1, max(0, math.ceil(fraction * num_layers) - 1)))
    return list(dict.fromkeys(indices))
