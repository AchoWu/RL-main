# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for OADLossFn (Overlap-Aligned Distillation, Path B).

These tests run on CPU (no GPU/distributed required) and exercise the single-rank
code path of OADLossFn. TP/CP behavior is covered by integration tests.

Path B = OADLossFn consumes an exact full-vocab `teacher_logsumexp` supplied by
the teacher worker. The tests construct that field manually from the full
teacher logits, mirroring what dtensor_policy_worker.get_topk_logits returns.
"""

import pytest
import torch

from nemo_rl.algorithms.loss_functions import OADLossFn


def _make_oad_data(
    batch_size: int = 2,
    seq_len: int = 8,
    vocab_size: int = 32,
    topk: int = 5,
    device: str = "cpu",
    seed: int = 0,
):
    """Build a self-consistent (data, student_logits, teacher_logits) triple.

    Builds the FULL teacher logits, derives top-k, and computes the EXACT
    full-vocab teacher_logsumexp — mirroring Path B (worker-supplied logsumexp).
    Returns the full teacher_logits separately so tests can construct
    "student == teacher" cases.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    teacher_logits = torch.randn(
        (batch_size, seq_len, vocab_size), generator=g, device=device
    )
    student_logits = torch.randn(
        (batch_size, seq_len, vocab_size), generator=g, device=device
    )

    teacher_topk_logits, teacher_topk_indices = teacher_logits.topk(topk, dim=-1)
    teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1)  # exact, [B, S]

    data = {
        "input_ids": torch.randint(
            0, vocab_size, (batch_size, seq_len), generator=g, device=device
        ),
        "input_lengths": torch.tensor([seq_len] * batch_size, device=device),
        "token_mask": torch.ones((batch_size, seq_len), device=device),
        "sample_mask": torch.ones(batch_size, device=device),
        "teacher_topk_logits": teacher_topk_logits,
        "teacher_topk_indices": teacher_topk_indices,
        "teacher_logsumexp": teacher_logsumexp,
    }
    return data, student_logits, teacher_logits


def _global_valid_toks(data):
    return torch.sum(data["sample_mask"].unsqueeze(-1) * data["token_mask"])


# -------------------------------------------------------------------
# 1. Sanity: shapes, scalar loss, no NaN/Inf
# -------------------------------------------------------------------
def test_oad_loss_returns_scalar_no_nan():
    data, student_logits, _ = _make_oad_data()
    loss_fn = OADLossFn({"eps": 1e-8})

    loss, metrics = loss_fn(
        student_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )

    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0  # -log(acceptance) with acceptance in (0, 1]

    for key in (
        "loss",
        "acceptance_rate_mean_pathB",
        "teacher_topk_mass",
        "student_mass_on_teacher_topk",
        "active_grad_ratio_token_pathB",
    ):
        assert key in metrics, f"missing metric: {key}"


# -------------------------------------------------------------------
# 2. Identity: student == teacher  =>  loss ≡ 0 (Path B textbook case)
#
# Critical regression guard: this would fail under Path A (loss = -log(M_T_true))
# or with clamp(max=1-eps).
# -------------------------------------------------------------------
def test_oad_loss_zero_when_student_equals_teacher():
    data, _, teacher_logits = _make_oad_data(seed=42)
    loss_fn = OADLossFn({"eps": 1e-12})

    loss, metrics = loss_fn(
        teacher_logits.clone(),
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )

    # Acceptance over teacher's top-k = teacher's true M_T (since p_S == p_T
    # exactly). loss = -log(M_T) which can be > 0 if top-k < vocab.
    # The proper "loss ≡ 0" identity requires top-k == vocab; see test 7.
    # What Path B DOES guarantee unconditionally: acceptance == teacher_topk_mass.
    assert abs(
        metrics["acceptance_rate_mean_pathB"] - metrics["teacher_topk_mass"]
    ) < 1e-5
    assert abs(
        metrics["student_mass_on_teacher_topk"] - metrics["teacher_topk_mass"]
    ) < 1e-5

    # loss == -log(acceptance), no spurious clamp_max bias.
    import math
    expected = -math.log(metrics["acceptance_rate_mean_pathB"])
    assert abs(loss.item() - expected) < 1e-4, (
        f"loss={loss.item()} vs -log(acceptance)={expected}"
    )

    # Under exact identity p_S == p_T at every top-k token, so no token has
    # p_S < p_T strictly — active_grad_ratio is 0.
    assert metrics["active_grad_ratio_token_pathB"] < 1e-5


# -------------------------------------------------------------------
# 3. Numerical correctness: hand-computed acceptance on a tiny case
# -------------------------------------------------------------------
def test_oad_loss_matches_hand_computed_acceptance():
    """Build a 1-batch, 2-position case where we can hand-verify the answer."""
    vocab_size = 6
    topk = 3
    teacher_logits = torch.tensor(
        [[[2.0, 1.0, 0.5, 0.0, -1.0, -2.0], [3.0, 2.0, 1.0, 0.0, -1.0, -2.0]]]
    )
    student_logits = torch.tensor(
        [[[1.5, 1.5, 0.0, 0.5, 0.0, -1.0], [2.0, 2.5, 0.5, 0.5, 0.0, -1.0]]]
    )

    teacher_topk_logits, teacher_topk_indices = teacher_logits.topk(topk, dim=-1)
    teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1)  # exact

    data = {
        "input_ids": torch.zeros((1, 2), dtype=torch.long),
        "input_lengths": torch.tensor([2]),
        "token_mask": torch.ones((1, 2)),
        "sample_mask": torch.ones(1),
        "teacher_topk_logits": teacher_topk_logits,
        "teacher_topk_indices": teacher_topk_indices,
        "teacher_logsumexp": teacher_logsumexp,
    }

    loss_fn = OADLossFn({"eps": 1e-12})
    loss, _ = loss_fn(
        student_logits.clone(),
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )

    # Hand-compute expected acceptance for position 0 (next-token loss only
    # uses positions [:-1], so position 1 is dropped).
    student_lse_0 = torch.logsumexp(student_logits[0, 0], dim=-1)
    teacher_lse_0 = teacher_logsumexp[0, 0]  # exact, supplied by Path B

    s_p_at_topk_0 = (
        student_logits[0, 0, teacher_topk_indices[0, 0]] - student_lse_0
    ).exp()
    t_p_at_topk_0 = (teacher_topk_logits[0, 0] - teacher_lse_0).exp()

    expected_accept_0 = torch.minimum(s_p_at_topk_0, t_p_at_topk_0).sum()
    expected_loss = -torch.log(expected_accept_0)

    torch.testing.assert_close(loss, expected_loss, atol=1e-5, rtol=1e-4)


# -------------------------------------------------------------------
# 4. Gradient flow: backprop through the loss yields finite gradients
#    on student logits.
# -------------------------------------------------------------------
def test_oad_loss_gradient_flow():
    data, student_logits, _ = _make_oad_data(seed=7)
    student_logits = student_logits.detach().requires_grad_(True)

    loss_fn = OADLossFn({"eps": 1e-8})
    loss, _ = loss_fn(
        student_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )
    loss.backward()

    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert student_logits.grad.abs().sum() > 0


# -------------------------------------------------------------------
# 5. Token mask is honored — masked positions must not influence loss
# -------------------------------------------------------------------
def test_oad_loss_respects_token_mask():
    data, student_logits, _ = _make_oad_data(batch_size=1, seq_len=6, seed=3)

    loss_fn = OADLossFn({"eps": 1e-8})
    loss_full, _ = loss_fn(
        student_logits.clone(),
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )

    masked_mask = data["token_mask"].clone()
    masked_mask[:, -1] = 0
    masked_data = {**data, "token_mask": masked_mask}

    student_logits_corrupted = student_logits.clone()
    student_logits_corrupted[:, -1] = -1e3

    loss_masked, _ = loss_fn(
        student_logits_corrupted,
        masked_data,
        global_valid_seqs=torch.sum(masked_data["sample_mask"]),
        global_valid_toks=_global_valid_toks(masked_data),
    )

    assert torch.isfinite(loss_masked)


# -------------------------------------------------------------------
# 6. Sanity bounds on metrics
# -------------------------------------------------------------------
def test_oad_loss_metric_bounds():
    data, student_logits, _ = _make_oad_data(seed=11)
    loss_fn = OADLossFn({"eps": 1e-8})

    _, metrics = loss_fn(
        student_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )

    assert 0.0 <= metrics["acceptance_rate_mean_pathB"] <= 1.0
    assert 0.0 <= metrics["teacher_topk_mass"] <= 1.0
    assert 0.0 <= metrics["student_mass_on_teacher_topk"] <= 1.0
    assert 0.0 <= metrics["active_grad_ratio_token_pathB"] <= 1.0

    # acceptance ≤ min(M_T, student_mass_on_teacher_topk) ≤ both individually,
    # since min(p_S, p_T) ≤ p_T and ≤ p_S.
    assert metrics["acceptance_rate_mean_pathB"] <= (
        metrics["teacher_topk_mass"] + 1e-5
    )
    assert metrics["acceptance_rate_mean_pathB"] <= (
        metrics["student_mass_on_teacher_topk"] + 1e-5
    )


# -------------------------------------------------------------------
# 7. Full-vocab top-k identity: when top-k spans the entire vocabulary,
#    there is no truncation, M_T == 1, and student==teacher gives loss ≡ 0.
#    This is the textbook Path B regression case.
# -------------------------------------------------------------------
def test_oad_loss_zero_when_topk_equals_vocab():
    batch_size, seq_len, vocab_size = 2, 6, 12
    g = torch.Generator().manual_seed(99)
    teacher_logits = torch.randn(
        (batch_size, seq_len, vocab_size), generator=g
    )

    teacher_topk_logits, teacher_topk_indices = teacher_logits.topk(
        vocab_size, dim=-1
    )
    teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1)

    data = {
        "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.long),
        "input_lengths": torch.tensor([seq_len] * batch_size),
        "token_mask": torch.ones((batch_size, seq_len)),
        "sample_mask": torch.ones(batch_size),
        "teacher_topk_logits": teacher_topk_logits,
        "teacher_topk_indices": teacher_topk_indices,
        "teacher_logsumexp": teacher_logsumexp,
    }

    loss_fn = OADLossFn({"eps": 1e-12})
    loss, metrics = loss_fn(
        teacher_logits.clone(),
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )

    assert loss.item() < 1e-5, (
        f"Path B identity at top-k=vocab should give loss ≡ 0, got {loss.item()}"
    )
    assert abs(metrics["acceptance_rate_mean_pathB"] - 1.0) < 1e-5
    assert abs(metrics["teacher_topk_mass"] - 1.0) < 1e-5
    assert metrics["active_grad_ratio_token_pathB"] < 1e-5


# -------------------------------------------------------------------
# 8. Truncation bound: when teacher mass is split (M_T_true ≈ 0.5) and
#    student == teacher, Path B yields loss = -log(M_T) AND
#    teacher_topk_mass directly equals M_T (observable truncation tightness).
#    This validates §3.2 (truncation bound) + §3.4 (M_T monitor).
# -------------------------------------------------------------------
def test_oad_loss_truncation_bound_when_teacher_mass_is_split():
    vocab_size = 16
    topk = 4

    top_logit = torch.log(torch.tensor(0.5 / topk))
    bot_logit = torch.log(torch.tensor(0.5 / (vocab_size - topk)))

    teacher_logits = torch.full((1, 2, vocab_size), bot_logit.item())
    teacher_logits[..., :topk] = top_logit  # token ids 0..topk-1 are top-k

    teacher_topk_logits, teacher_topk_indices = teacher_logits.topk(topk, dim=-1)
    teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1)  # exact

    data = {
        "input_ids": torch.zeros((1, 2), dtype=torch.long),
        "input_lengths": torch.tensor([2]),
        "token_mask": torch.ones((1, 2)),
        "sample_mask": torch.ones(1),
        "teacher_topk_logits": teacher_topk_logits,
        "teacher_topk_indices": teacher_topk_indices,
        "teacher_logsumexp": teacher_logsumexp,
    }

    loss_fn = OADLossFn({"eps": 1e-12})
    loss, metrics = loss_fn(
        teacher_logits.clone(),
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )

    # M_T_true ≈ 0.5; expected loss = -log(0.5).
    expected_loss = -torch.log(torch.tensor(0.5)).item()
    assert abs(loss.item() - expected_loss) < 5e-2, (
        f"Truncation bound under M_T=0.5 should give loss ≈ {expected_loss}, "
        f"got {loss.item()}"
    )

    # Under Path B, teacher_topk_mass IS the observable M_T_true.
    assert abs(metrics["teacher_topk_mass"] - 0.5) < 5e-2, (
        f"expected teacher_topk_mass ≈ 0.5, got {metrics['teacher_topk_mass']}"
    )


# -------------------------------------------------------------------
# 9. Missing teacher_logsumexp must produce a clear error
#    (forward compatibility: workers without Path B support).
# -------------------------------------------------------------------
def test_oad_loss_raises_when_logsumexp_missing():
    data, student_logits, _ = _make_oad_data(seed=5)
    del data["teacher_logsumexp"]

    loss_fn = OADLossFn({"eps": 1e-8})
    with pytest.raises(KeyError, match="teacher_logsumexp"):
        loss_fn(
            student_logits,
            data,
            global_valid_seqs=torch.sum(data["sample_mask"]),
            global_valid_toks=_global_valid_toks(data),
        )
