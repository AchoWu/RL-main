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
"""CPU tests for tools/bopd_diagnostics/diag_lib.py."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from nemo_rl.algorithms.loss_functions import DistillationLossFn


def _load_lib():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "bopd_diagnostics"
        / "diag_lib.py"
    )
    spec = importlib.util.spec_from_file_location("bopd_diag_lib", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lib = _load_lib()

PROMPT, RESPONSE, PAD = 3, 9, 2


def _toy_batch(vocab=7, k=4, seed=0):
    """One sequence: PROMPT prompt tokens, RESPONSE response tokens, PAD padding."""
    gen = torch.Generator().manual_seed(seed)
    seq = PROMPT + RESPONSE + PAD
    ids = torch.randint(1, vocab, (1, seq), generator=gen)
    ids[0, PROMPT + RESPONSE :] = 0
    token_mask = torch.zeros(1, seq)
    token_mask[0, PROMPT : PROMPT + RESPONSE] = 1
    teacher = torch.randn(1, seq, vocab, generator=gen)
    t_vals, t_idx = teacher.topk(k, dim=-1)
    logits = torch.randn(1, seq, vocab, generator=gen)
    return ids, token_mask, t_vals, t_idx, logits


@pytest.mark.parametrize("block_mode", ["fixed", "sliding"])
@pytest.mark.parametrize("block_size", [1, 2, 4])
@pytest.mark.parametrize("baseline", [False, True])
def test_split_objective_matches_distillation_loss(block_mode, block_size, baseline):
    """direct + score objectives / token count reproduce DistillationLossFn's gradient."""
    ids, token_mask, t_vals, t_idx, logits = _toy_batch()
    n_tok = token_mask[:, 1:].sum()

    ref_logits = logits.clone().requires_grad_(True)
    loss_fn = DistillationLossFn(
        {
            "kl_type": "reverse",
            "mixed_kl_weight": 0.0,
            "zero_outside_topk": False,
            "block_size": block_size,
            "block_mode": block_mode,
            "block_baseline": baseline,
            "score_loss_weight": 1.0,
        }
    )
    data = {
        "input_ids": ids,
        "input_lengths": torch.tensor([PROMPT + RESPONSE]),
        "token_mask": token_mask,
        "sample_mask": torch.ones(1),
        "teacher_topk_logits": t_vals,
        "teacher_topk_indices": t_idx,
    }
    loss, _ = loss_fn(ref_logits, data, torch.tensor(1.0), n_tok)
    loss.backward()

    z = logits[0, :-1].clone().requires_grad_(True)
    mask = token_mask[0, 1:]
    kl = lib.topk_reverse_kl(z, t_vals[0, :-1], t_idx[0, :-1])
    logq = lib.sampled_logprob(z, ids[0, 1:])
    objective = (mask * kl).sum()
    if block_size > 1:
        r, b = lib.kl_to_go_and_baseline(kl.detach(), mask, block_size, block_mode)
        objective = objective + (mask * logq * (r - b if baseline else r)).sum()
    (objective / n_tok).backward()

    torch.testing.assert_close(z.grad, ref_logits.grad[0, :-1], rtol=1e-5, atol=1e-6)
    assert ref_logits.grad[0, -1].abs().max() == 0


@pytest.mark.parametrize("block_mode", ["fixed", "sliding"])
@pytest.mark.parametrize("block_size", [2, 3, 5])
def test_branch_kl_to_go_matches_training_kl_to_go(block_mode, block_size):
    """Along the real continuation, the branch KL-to-go equals _block_kl_to_go."""
    torch.manual_seed(0)
    _, token_mask, _, _, _ = _toy_batch()
    mask = token_mask[0, 1:]
    kl = torch.rand(mask.shape[0])
    r, _ = lib.kl_to_go_and_baseline(kl, mask, block_size, block_mode)
    horizon = 4
    padded_kl = torch.cat([kl, torch.zeros(horizon)])
    padded_mask = torch.cat([mask, torch.zeros(horizon)])
    for i in mask.nonzero().flatten().tolist():
        branch = lib.branch_kl_to_go(
            padded_kl[None, i + 1 : i + 1 + horizon],
            padded_mask[None, i + 1 : i + 1 + horizon],
            block_size,
            block_mode,
            resp_offset=i - (PROMPT - 1),
        )
        assert branch[0].item() == pytest.approx(r[i].item(), abs=1e-6)


def test_sketch_is_linear_and_preserves_inner_products():
    torch.manual_seed(0)
    sketch = lib.GradSketch(dim=4096, seed=3)
    shapes = [(300, 400), (7,), (50, 60)]
    a = [torch.randn(s) for s in shapes]
    b = [torch.randn(s) + 0.5 * x for s, x in zip(shapes, a)]

    def sk(vs):
        out = torch.zeros(4096)
        sketch.add_(out, vs)
        return out

    sa, sb = sk(a), sk(b)
    torch.testing.assert_close(sk([x + y for x, y in zip(a, b)]), sa + sb)
    exact = sum(float((x * y).sum()) for x, y in zip(a, b))
    norm = math.sqrt(
        sum(float((x * x).sum()) for x in a) * sum(float((y * y).sum()) for y in b)
    )
    # std of the estimate is about norm / sqrt(dim)
    assert abs(float(sa @ sb) - exact) < 5 * norm / math.sqrt(4096)
    # a single parameter shorter than dim is only sign-flipped and shifted: exact
    small = lib.GradSketch(dim=4096, seed=3)
    x, y = torch.randn(1000), torch.randn(1000)
    sx, sy = torch.zeros(4096), torch.zeros(4096)
    small.add_(sx, [x])
    small.add_(sy, [y])
    assert float(sx @ sy) == pytest.approx(float(x @ y), rel=1e-5)


def test_branch_score_stats_recovers_known_values():
    """Synthetic position: R = mu(y_{i+1}) + Gaussian noise, all quantities known."""
    gen = torch.Generator().manual_seed(0)
    q = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
    mu = torch.tensor([1.0, 3.0, 0.5, 6.0], dtype=torch.float64)
    sigma, b_hist = 0.7, 1.5
    d = torch.tensor([0.3, -0.2, 0.1, 0.05], dtype=torch.float64)
    n1, n2 = 200_000, 3
    y = torch.multinomial(q, n1, replacement=True, generator=gen)
    r = mu[y][:, None] + sigma * torch.randn(n1, n2, generator=gen, dtype=torch.float64)
    st = lib.branch_score_stats(r, y, q, d, b_hist)

    eye = torch.eye(4, dtype=torch.float64)
    s_star = (q[:, None] * (eye - q) * mu[:, None]).sum(0)
    k_diag = 1 - 2 * q + (q * q).sum()
    value = (q * mu).sum()
    expect = {
        "mean_R": float(value),
        "between": float((q * (mu - value) ** 2).sum()),
        "within": sigma**2,
        "signal2": float(s_star @ s_star),
        "cross": float(s_star @ d),
        "m2_none": float((q * (mu**2 + sigma**2) * k_diag).sum()),
        "m2_hist": float((q * ((mu - b_hist) ** 2 + sigma**2) * k_diag).sum()),
        "m2_oracle": float((q * ((mu - value) ** 2 + sigma**2) * k_diag).sum()),
    }
    for key, val in expect.items():
        assert st[key] == pytest.approx(val, rel=0.03, abs=0.01), key
    assert st["m2_oracle"] < st["m2_hist"] < st["m2_none"]


def test_summarize_gradients_matches_exact():
    """Sketch dim above the parameter count makes every inner product exact."""
    torch.manual_seed(0)
    variants = ["D", "S:sliding:2", "Sbl:sliding:2"]
    dim, n_param = 1024, 300
    sketch = lib.GradSketch(dim=dim, seed=1)
    batches = {0: 5, 1: 4, 2: 6}
    grads, sketches, tokens, samples = {}, {}, {}, []
    for b, n in batches.items():
        g = torch.randn(n, 3, n_param, dtype=torch.float64)
        g[:, 1] += 0.5 * g[:, 0] + 0.3  # correlated with D plus a shared mean
        grads[b] = g
        tokens[b] = float(torch.randint(50, 100, ()))
        sketches[b] = torch.zeros(3, dim)
        for s in range(n):
            for v in range(3):
                sketch.add_(sketches[b][v], [g[s, v].float()])
            samples.append(
                {
                    "batch": b,
                    "n2": [float(g[s, v] @ g[s, v]) for v in range(3)],
                    "dotD": [float(g[s, v] @ g[s, 0]) for v in range(3)],
                }
            )
    out = lib.summarize_gradients(variants, sketches, tokens, samples)

    big = {b: grads[b].sum(0) / tokens[b] for b in batches}  # [3, n_param]
    cos = lambda x, y: float(x @ y / (x.norm() * y.norm()))  # noqa: E731
    exp_ratio = sum(float(big[b][1].norm() / big[b][0].norm()) for b in batches) / 3
    exp_cos = sum(cos(big[b][0], big[b][1]) for b in batches) / 3
    q1 = out["q1"]["S:sliding:2"]
    assert q1["ratio_batch"] == pytest.approx(exp_ratio, rel=1e-4)
    assert q1["cos_batch"] == pytest.approx(exp_cos, rel=1e-4)
    total = {b: big[b][0] + big[b][1] for b in batches}
    exp_tcos = sum(cos(total[b], big[b][0]) for b in batches) / 3
    assert q1["total_cos_batch"] == pytest.approx(exp_tcos, rel=1e-4)

    pairs = [(b, c) for b in batches for c in batches if b != c]
    sig = sum(float(big[b][1] @ big[c][1]) for b, c in pairs) / len(pairs)
    q3 = out["q3"]["S:sliding:2"]
    assert q3["signal_sq"] == pytest.approx(sig, rel=1e-4)
    within = []
    for b, n in batches.items():
        g = grads[b][:, 1]
        var = ((g - g.mean(0)) ** 2).sum(1).sum() / (n - 1)
        within.append(float(n * var / tokens[b] ** 2))
    assert q3["noise_within"] == pytest.approx(sum(within) / 3, rel=1e-4)
    exp_cross_cos = sum(cos(big[b][1], big[c][1]) for b, c in pairs) / len(pairs)
    assert q3["cross_batch_cos"] == pytest.approx(exp_cross_cos, rel=1e-4)
    # D + S per-sample norms come from n2 / dotD
    gt = {b: grads[b][:, 0] + grads[b][:, 1] for b in batches}
    within_t = []
    for b, n in batches.items():
        var = ((gt[b] - gt[b].mean(0)) ** 2).sum(1).sum() / (n - 1)
        within_t.append(float(n * var / tokens[b] ** 2))
    assert out["q3"]["D+S:sliding:2"]["noise_within"] == pytest.approx(
        sum(within_t) / 3, rel=1e-4
    )


def _tiny_lm(seed, vocab=32):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return Qwen2ForCausalLM(cfg).eval()


def test_objective_backward_matches_full_logits():
    """Chunked, checkpointed objective gives the gradient of the unchunked one."""
    student = _tiny_lm(0)
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(1, 32, (23,), generator=gen)
    mask = torch.zeros(22)
    mask[4:] = 1
    t_vals, t_idx = torch.randn(22, 32, generator=gen).topk(6, dim=-1)
    weight = torch.rand(22, generator=gen)
    params = list(student.parameters())

    student.train()
    lib.objective_backward(student, ids, mask, t_vals, t_idx, 1.0, weight, chunk=5)
    chunked = [p.grad.clone() for p in params]
    student.zero_grad(set_to_none=True)

    logits = student(input_ids=ids[None]).logits[0, :-1].float()
    kl = lib.topk_reverse_kl(logits, t_vals, t_idx)
    logq = lib.sampled_logprob(logits, ids[1:])
    (mask * (kl + weight * logq)).sum().backward()
    for a, p in zip(chunked, params):
        torch.testing.assert_close(a, p.grad, rtol=1e-4, atol=1e-6)


def test_branch_replay_matches_rollout_kl_to_go():
    """Replaying the real continuation through the KV-cache branch path gives training's R."""
    student, teacher = _tiny_lm(0), _tiny_lm(1)
    eos, prompt_len, max_len = 2, 5, 64
    gen = torch.Generator().manual_seed(2)
    ids = torch.randint(3, 32, (20,), generator=gen)
    ids[-1] = eos
    mask = torch.zeros(19)
    mask[prompt_len - 1 :] = 1
    t_vals, t_idx, t_cache = lib.teacher_topk(teacher, ids, 8, 7, True, 32)
    kl, _, _, _, s_cache = lib.student_terms(student, ids, t_vals, t_idx, 7, True)
    horizon = 4
    sample_gen = torch.Generator().manual_seed(3)
    for i in [prompt_len - 1, 9, 15, 17]:
        real = ids[i + 1 : i + 1 + horizon]
        replay = torch.full((horizon,), eos, dtype=torch.long)
        replay[: real.numel()] = real
        nxt = torch.cat([torch.tensor([7, 9]), replay[:1]])
        forced = torch.full((3, horizon), -1, dtype=torch.long)
        forced[-1] = replay
        future_kl, alive = lib.branch_continuations(
            student,
            teacher,
            s_cache,
            t_cache,
            i,
            nxt,
            forced,
            horizon,
            eos,
            max_len,
            8,
            32,
            sample_gen,
        )
        assert future_kl.shape == (3, horizon) and torch.isfinite(future_kl).all()
        # the replay row is alive exactly where the training mask is
        padded = torch.cat([mask, torch.zeros(horizon)])
        torch.testing.assert_close(alive[-1].float(), padded[i + 1 : i + 1 + horizon])
        for mode in ["fixed", "sliding"]:
            for k in [2, 3, 5]:
                r, _ = lib.kl_to_go_and_baseline(kl, mask, k, mode)
                rb = lib.branch_kl_to_go(
                    future_kl[-1:], alive[-1:], k, mode, i - (prompt_len - 1)
                )
                assert rb[0].item() == pytest.approx(r[i].item(), rel=1e-4, abs=1e-6)
