import numpy as np

from research.hidden_state_probe.core import (
    average_precision,
    binary_metrics,
    grouped_problem_split,
    prefix_checkpoints,
    resolve_layer_indices,
    roc_auc,
)
from research.hidden_state_probe.run_hidden_state_probe import within_problem_concordance


class FakeTokenizer:
    pieces = {
        1: "work",
        2: "\n",
        3: "\n",
        4: "more",
        5: "\n\n",
        6: "\\boxed{",
        7: "1}",
        8: "tail",
    }

    def decode(self, token_ids, **kwargs):
        return "".join(self.pieces[token_id] for token_id in token_ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}


def test_prefix_checkpoints_exclude_answer_and_track_content_token():
    prefix_positions, probe_positions = prefix_checkpoints(
        [1, 2, 3, 4, 5, 6, 7, 8],
        FakeTokenizer(),
        response_text="work\n\nmore\n\n\\boxed{1}tail",
        max_checkpoints=3,
        min_prefix_tokens=2,
        min_remaining_tokens=1,
    )
    assert prefix_positions == [0, 3, 5]
    assert probe_positions == [0, 1, 4]


def test_grouped_split_never_splits_problem():
    problem_ids = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    splits = grouped_problem_split(problem_ids, seed=42)
    for problem_id in np.unique(problem_ids):
        assert len(set(splits[problem_ids == problem_id])) == 1


def test_binary_metrics_on_perfect_ranking():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(labels, scores) == 1.0
    assert average_precision(labels, scores) == 1.0
    metrics = binary_metrics(labels, scores)
    assert metrics["accuracy_at_0.5"] == 1.0
    assert metrics["brier_score"] < 0.05


def test_layer_fraction_resolution():
    assert resolve_layer_indices(28, [0.5, 0.75, 0.9, 1.0]) == [13, 20, 25, 27]


def test_within_problem_concordance_controls_problem_difficulty():
    result = within_problem_concordance(
        labels=np.asarray([1, 0, 1, 0]),
        scores=np.asarray([0.8, 0.2, 0.4, 0.6]),
        problem_ids=np.asarray([10, 10, 11, 11]),
        checkpoint_indices=np.asarray([0, 0, 0, 0]),
    )
    assert result["concordance"] == 0.5
    assert result["num_pairs"] == 2
    assert result["num_mixed_problems"] == 2
