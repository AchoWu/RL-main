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
"""Math for the Block-OPD gradient diagnostics (docs/design-docs/opd.md, 3.1.1).

Index convention, as in DistillationLossFn: entry i of a per-position tensor belongs
to the student/teacher distribution at sequence position i, the one that predicts
token i + 1, and mask[i] = token_mask[i + 1]. A sequence of length L therefore has
L - 1 positions.

The per-sample block loss gradient splits into
  direct: grad sum_i mask_i KL_i
  score:  grad sum_i mask_i log q(y_{i+1}) * (R_i - b_i)
with R_i the KL-to-go (_block_kl_to_go) and b_i the baseline (_block_kl_baseline,
or 0). Training divides both by the batch's response-token count.
"""

import contextlib
import copy
import math
from typing import Any, Optional

import torch

from nemo_rl.algorithms.loss_functions import _block_kl_baseline, _block_kl_to_go

# ---------------------------------------------------------------------------
# Per-position terms (same formulas as DistillationLossFn, zero_outside_topk false)
# ---------------------------------------------------------------------------


def topk_reverse_kl(
    student_logits: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
) -> torch.Tensor:
    """Reverse KL on the teacher top-k support, both sides renormalized over it.

    Args:
        student_logits: [..., V] float32 full-vocab student logits.
        teacher_topk_logits: [..., k] teacher logits of its top-k tokens.
        teacher_topk_indices: [..., k] their token ids.

    Returns:
        [...] per-position KL.
    """
    s = torch.log_softmax(student_logits.gather(-1, teacher_topk_indices), dim=-1)
    t = torch.log_softmax(teacher_topk_logits.to(s.dtype), dim=-1)
    return (s.exp() * (s - t)).sum(-1)


def sampled_logprob(
    student_logits: torch.Tensor, next_ids: torch.Tensor
) -> torch.Tensor:
    """Full-vocab log q of the sampled tokens, the distribution the score term differentiates."""
    picked = student_logits.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(student_logits, dim=-1)


def kl_to_go_and_baseline(
    per_token_kl: torch.Tensor, mask: torch.Tensor, block_size: int, block_mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """(R, b) of one sequence ([L - 1] each) from the training-loss helpers."""
    kl_to_go = _block_kl_to_go(per_token_kl[None], mask[None], block_size, block_mode)
    baseline = _block_kl_baseline(kl_to_go, mask[None], block_size, block_mode)
    return kl_to_go[0], baseline[0]


def branch_kl_to_go(
    future_kl: torch.Tensor,
    alive: torch.Tensor,
    block_size: int,
    block_mode: str,
    resp_offset: int,
) -> torch.Tensor:
    """KL-to-go R_i of position i along branches that continue after it.

    Matches _block_kl_to_go for a single-turn response, where the positions after i
    are consecutive response positions.

    Args:
        future_kl: [N, J] KL at positions i + 1 .. i + J along each branch.
        alive: [N, J] response mask of those positions.
        block_size: k; needs J >= k - 1.
        block_mode: "fixed" or "sliding".
        resp_offset: index of position i among the response positions, which fixes
            where its block ends in "fixed" mode.

    Returns:
        [N] KL-to-go.
    """
    horizon = future_kl.shape[-1]
    assert horizon >= block_size - 1, "branch horizon shorter than the block"
    j = torch.arange(1, horizon + 1, device=future_kl.device, dtype=future_kl.dtype)
    if block_mode == "fixed":
        weight = (j <= block_size - 1 - resp_offset % block_size).to(future_kl.dtype)
    else:
        weight = ((block_size - j) / block_size).clamp(min=0)
    return (future_kl * alive.to(future_kl.dtype) * weight).sum(-1)


def direct_logit_grad(
    student_logits: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
) -> torch.Tensor:
    """D KL_i / d z_i for the full-vocab student logits z_i ([V] -> [V])."""
    with torch.enable_grad():
        z = student_logits.detach().float().requires_grad_(True)
        kl = topk_reverse_kl(z, teacher_topk_logits, teacher_topk_indices)
        (grad,) = torch.autograd.grad(kl, z)
    return grad


# ---------------------------------------------------------------------------
# Q2 / Q3: score-term statistics at one position from nested branches
# ---------------------------------------------------------------------------


def branch_score_stats(
    kl_to_go: torch.Tensor,
    first_tokens: torch.Tensor,
    probs: torch.Tensor,
    direct_grad: torch.Tensor,
    hist_baseline: float,
) -> dict[str, float]:
    """Logit-space statistics of the score term at one position i.

    The score term's single-sample gradient w.r.t. the logits z_i is
    g = (e_y - q) (R - b), y = y_{i+1} ~ q. Its mean is
    s* = E_y[(e_y - q) mu(y)], mu(y) = E[R | y_<=i, y_{i+1} = y].

    Args:
        kl_to_go: [N1, N2] R along N2 continuations of each of N1 sampled y_{i+1}.
        first_tokens: [N1] the sampled y_{i+1}.
        probs: [V] student distribution q at position i.
        direct_grad: [V] d KL_i / d z_i.
        hist_baseline: the training baseline b_i at this position.

    Returns:
        Unbiased (over the branch sampling) per-position estimates:
          mean_R, R2:   E[R], E[R^2]                       (given y_<=i)
          between:      Var_{y_{i+1}}(mu(y_{i+1}))           (what y_{i+1} changes)
          within:       E_{y_{i+1}} Var(R | y_<=i+1)         (later tokens' noise)
          signal2:      |s*|^2
          cross:        <s*, d>, d the direct gradient
          d2:           |d|^2
          m2_none / m2_hist / m2_oracle: E|g|^2 with b = 0, b = b_i and b = E[R | y_<=i]
    """
    r = kl_to_go.double()
    q = probs.double()
    d = direct_grad.double()
    n1, n2 = r.shape
    assert n1 >= 2, "need at least two sampled next tokens"
    m = r.mean(dim=1)
    within = r.var(dim=1).mean() if n2 > 1 else r.new_zeros(())
    between = m.var() - within / n2

    q2 = (q * q).sum()
    # |e_y - q|^2
    kdiag = 1 - 2 * q[first_tokens] + q2
    # independent continuations make m_a m_b (a != b) unbiased for mu(y_a) mu(y_b), so
    # sum_{a != b} m_a m_b <e_a - q, e_b - q> / (N1 (N1 - 1)) is unbiased for |s*|^2
    u = torch.zeros_like(q).index_add_(0, first_tokens, m) - q * m.sum()
    signal2 = ((u * u).sum() - (m * m * kdiag).sum()) / (n1 * (n1 - 1))
    cross = (m * (d[first_tokens] - (q * d).sum())).mean()

    def second_moment(baseline: torch.Tensor | float) -> torch.Tensor:
        return ((r - baseline) ** 2 * kdiag[:, None]).mean()

    # leave-one-out estimate of E[R | y_<=i] is independent of (y_a, R_a.); its own
    # variance Var(m) / (N1 - 1) adds E[K] times that much, which is removed.
    loo = ((m.sum() - m) / (n1 - 1))[:, None]
    m2_oracle = second_moment(loo) - (between + within / n2) / (n1 - 1) * kdiag.mean()
    out = {
        "mean_R": r.mean(),
        "R2": (r * r).mean(),
        "between": between,
        "within": within,
        "signal2": signal2,
        "cross": cross,
        "d2": (d * d).sum(),
        "m2_none": second_moment(0.0),
        "m2_hist": second_moment(float(hist_baseline)),
        "m2_oracle": m2_oracle,
    }
    return {k: float(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Parameter-space sketch
# ---------------------------------------------------------------------------


class GradSketch:
    """Random-sign fold sketch of a parameter-space vector.

    Coordinate c of parameter p is multiplied by a fixed random sign and added to
    bucket (c + offset_p) mod dim. <S a, S b> is unbiased for <a, b> with standard
    deviation about |a| |b| / sqrt(dim). The sketch is linear and depends only on
    (seed, parameter order, dim), so sketches from different processes add up.
    """

    def __init__(self, dim: int, seed: int):
        self.dim = dim
        self.seed = seed

    def add_(self, out: torch.Tensor, grads: list[Optional[torch.Tensor]]) -> None:
        """Out += S(grads), out: [dim] float32."""
        for idx, grad in enumerate(grads):
            if grad is None:
                continue
            gen = torch.Generator(device=grad.device)
            gen.manual_seed(self.seed * 1_000_003 + idx)
            n = grad.numel()
            signs = torch.randint(
                0, 2, (n,), generator=gen, device=grad.device, dtype=torch.int8
            )
            v = grad.reshape(-1).float() * (signs.float() * 2 - 1)
            del signs
            v = torch.nn.functional.pad(v, (0, (-n) % self.dim))
            fold = v.view(-1, self.dim).sum(dim=0)
            offset = (idx * 2_654_435_761 + self.seed) % self.dim
            out += torch.roll(fold, offset).to(out.device)


# ---------------------------------------------------------------------------
# Model passes (HF causal LMs; one unpadded sequence at a time)
# ---------------------------------------------------------------------------


# analyze --fp32 turns this off to separate bf16 rounding from real effects
BF16_AUTOCAST = True


def autocast_ctx(device: torch.device):
    """bf16 autocast on GPU (training runs bf16 compute on fp32 master weights)."""
    if device.type == "cuda" and BF16_AUTOCAST:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _chunks(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(n, s + size)


def _hidden(model, ids: torch.Tensor, use_cache: bool):
    out = model.get_decoder()(input_ids=ids[None], use_cache=use_cache)
    return out.last_hidden_state[0], (out.past_key_values if use_cache else None)


def remap_teacher_topk(
    vals: torch.Tensor, idx: torch.Tensor, student_vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """distillation._remap_teacher_topk_outside_student_vocab_ on [..., k] tensors."""
    from nemo_rl.algorithms.distillation import (
        _remap_teacher_topk_outside_student_vocab_,
    )

    data = {"teacher_topk_logits": vals, "teacher_topk_indices": idx}
    _remap_teacher_topk_outside_student_vocab_(data, student_vocab_size)
    return data["teacher_topk_logits"], data["teacher_topk_indices"]


@torch.no_grad()
def teacher_topk(
    teacher,
    ids: torch.Tensor,
    k: int,
    chunk: int,
    use_cache: bool,
    student_vocab_size: int,
):
    """Teacher top-k logits at positions 0..L-2 ([L-1, k] each) and optionally its KV cache."""
    with autocast_ctx(ids.device):
        h, cache = _hidden(teacher, ids, use_cache)
    head = teacher.get_output_embeddings()
    vals, idx = [], []
    for s, e in _chunks(ids.shape[0] - 1, chunk):
        with autocast_ctx(ids.device):
            logits = head(h[s:e])
        v, i = logits.float().topk(k, dim=-1)
        vals.append(v)
        idx.append(i)
    vals, idx = remap_teacher_topk(torch.cat(vals), torch.cat(idx), student_vocab_size)
    return vals, idx, cache


@torch.no_grad()
def student_terms(
    student,
    ids: torch.Tensor,
    t_vals: torch.Tensor,
    t_idx: torch.Tensor,
    chunk: int,
    use_cache: bool,
):
    """Per-position KL_i, log q(y_{i+1}) and student entropy ([L-1] each), plus h and cache."""
    with autocast_ctx(ids.device):
        h, cache = _hidden(student, ids, use_cache)
    head = student.get_output_embeddings()
    kl, logq, ent = [], [], []
    for s, e in _chunks(ids.shape[0] - 1, chunk):
        with autocast_ctx(ids.device):
            logits = head(h[s:e])
        logits = logits.float()
        kl.append(topk_reverse_kl(logits, t_vals[s:e], t_idx[s:e]))
        logq.append(sampled_logprob(logits, ids[s + 1 : e + 1]))
        logp = torch.log_softmax(logits, dim=-1)
        ent.append(-(logp.exp() * logp).sum(-1))
    return torch.cat(kl), torch.cat(logq), torch.cat(ent), h, cache


def _chunk_objective(head, h, next_ids, t_vals, t_idx, mask, score_weight, kl_coef):
    with autocast_ctx(h.device):
        logits = head(h)
    logits = logits.float()
    total = logits.new_zeros(())
    if kl_coef:
        total = total + kl_coef * (mask * topk_reverse_kl(logits, t_vals, t_idx)).sum()
    if score_weight is not None:
        total = total + (mask * score_weight * sampled_logprob(logits, next_ids)).sum()
    return total


def objective_backward(
    student,
    ids: torch.Tensor,
    mask: torch.Tensor,
    t_vals: torch.Tensor,
    t_idx: torch.Tensor,
    kl_coef: float,
    score_weight: Optional[torch.Tensor],
    chunk: int,
) -> None:
    """Accumulate the gradient of the direct / score objective into .grad.

    The objective is sum_i mask_i (kl_coef * KL_i + score_weight_i * log q(y_{i+1})).
    score_weight is treated as a constant. The LM head runs in checkpointed chunks so
    the [L, V] logits are never held for the whole sequence.
    """
    with autocast_ctx(ids.device):
        h, _ = _hidden(student, ids, False)
    head = student.get_output_embeddings()
    total = h.new_zeros((), dtype=torch.float32)
    for s, e in _chunks(ids.shape[0] - 1, chunk):
        if not mask[s:e].any():
            continue
        total = total + torch.utils.checkpoint.checkpoint(
            _chunk_objective,
            head,
            h[s:e],
            ids[s + 1 : e + 1],
            t_vals[s:e],
            t_idx[s:e],
            mask[s:e],
            None if score_weight is None else score_weight[s:e],
            kl_coef,
            use_reentrant=False,
        )
    total.backward()


def _branch_cache(cache, prefix_len: int, repeats: int):
    branch = copy.deepcopy(cache)
    branch.crop(prefix_len)
    branch.batch_repeat_interleave(repeats)
    return branch


@torch.no_grad()
def branch_continuations(
    student,
    teacher,
    student_cache,
    teacher_cache,
    position: int,
    next_tokens: torch.Tensor,
    forced: Optional[torch.Tensor],
    horizon: int,
    eos_id: int,
    max_len: int,
    topk: int,
    student_vocab_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continue the prefix y_<=position along len(next_tokens) branches.

    Branch n takes y_{i+1} = next_tokens[n] (i = position), then samples from the
    student up to y_{i+horizon}, or follows forced[n] (a row of -1 = sample).

    Returns:
        future_kl: [N, horizon] KL at positions i + 1 .. i + horizon.
        alive: [N, horizon] response mask of those positions (0 after EOS or at max_len).
    """
    n = next_tokens.shape[0]
    device = next_tokens.device
    prefix_len = position + 1
    cache = _branch_cache(student_cache, prefix_len, n)
    tokens, student_logits = [next_tokens], []
    current = next_tokens
    for step in range(1, horizon + 1):
        with autocast_ctx(device):
            out = student(
                input_ids=current[:, None], past_key_values=cache, use_cache=True
            )
        logits = out.logits[:, -1].float()
        student_logits.append(logits)
        if step < horizon:
            current = torch.multinomial(
                torch.softmax(logits, dim=-1), 1, generator=generator
            ).squeeze(-1)
            if forced is not None:
                follow = forced[:, step] >= 0
                current = torch.where(follow, forced[:, step], current)
            tokens.append(current)
    del cache
    tokens = torch.stack(tokens, dim=1)  # [N, horizon]: y_{i+1} .. y_{i+horizon}
    student_logits = torch.stack(student_logits, dim=1)  # [N, horizon, V]

    cache = _branch_cache(teacher_cache, prefix_len, n)
    with autocast_ctx(device):
        t_logits = teacher(
            input_ids=tokens, past_key_values=cache, use_cache=True
        ).logits
    del cache
    t_vals, t_idx = t_logits.float().topk(topk, dim=-1)
    t_vals, t_idx = remap_teacher_topk(t_vals, t_idx, student_vocab_size)
    future_kl = topk_reverse_kl(student_logits, t_vals, t_idx)

    # position i + t is a response position iff y_{i+1..i+t} has no EOS and the
    # token it predicts, at i + t + 1, fits in max_len
    steps = torch.arange(1, horizon + 1, device=device)
    alive = (torch.cumsum((tokens == eos_id).long(), dim=1) == 0) & (
        position + steps + 1 < max_len
    )
    return future_kl, alive


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _offdiag_mean(mat: torch.Tensor) -> float:
    n = mat.shape[0]
    if n < 2:
        return float("nan")
    return float((mat.sum() - mat.diagonal().sum()) / (n * (n - 1)))


def _safe_sqrt(x: float) -> float:
    return math.sqrt(x) if x > 0 else float("nan")


def summarize_gradients(
    variants: list[str],
    sketches: dict[int, torch.Tensor],
    tokens: dict[int, float],
    samples: list[dict[str, Any]],
    col_chunk: int = 1 << 20,
) -> dict[str, Any]:
    """Q1 / Q3 parameter-space metrics.

    Args:
        variants: base gradient names; variants[0] must be "D" (direct), the others
            score terms "S:<mode>:<k>" (no baseline) / "Sbl:<mode>:<k>" (baseline).
        sketches: batch id -> [n_var, dim] sketch of the per-sample gradient sum.
        tokens: batch id -> response-token count (training's normalizer).
        samples: per-sample {"batch", "n2": [n_var], "dotD": [n_var]} exact squared
            norms and inner products with the sample's direct gradient.

    Returns:
        {"q1": {score variant: metrics}, "q3": {gradient: metrics}, "num_batches": n}.
    """
    assert variants[0] == "D"
    batches = sorted(sketches)
    nb, nv = len(batches), len(variants)
    # outputs: every base gradient plus D + each score term (the full block-OPD gradient)
    names = list(variants) + ["D+" + v for v in variants[1:]]
    coef = torch.zeros(len(names), nv, dtype=torch.float64)
    for o in range(nv):
        coef[o, o] = 1
    for o, v in enumerate(variants[1:], start=nv):
        coef[o, 0] = 1
        coef[o, variants.index(v)] = 1

    rows = torch.cat([sketches[b].float() / tokens[b] for b in batches])  # [nb*nv, dim]
    gram = torch.zeros(nb * nv, nb * nv, dtype=torch.float64)
    for s, e in _chunks(rows.shape[1], col_chunk):
        block = rows[:, s:e].double()
        gram += block @ block.T
    gram = gram.view(nb, nv, nb, nv)
    # go[b, o, c, p] = <G_o^b, G_p^c>, G^b the token-normalized batch gradient
    go = torch.einsum("ov,bvcw,pw->bocp", coef, gram, coef)

    def per_sample_n2(o: int, rec: dict[str, Any]) -> float:
        if o < nv:
            return float(rec["n2"][o])
        x = variants.index(names[o][2:])
        return float(rec["n2"][0] + rec["n2"][x] + 2 * rec["dotD"][x])

    q3 = {}
    for o, name in enumerate(names):
        diag = torch.stack([go[b, o, b, o] for b in range(nb)])
        within = []
        for bi, b in enumerate(batches):
            recs = [r for r in samples if r["batch"] == b]
            n = len(recs)
            if n < 2:
                continue
            sum_sq = sum(per_sample_n2(o, r) for r in recs)
            norm_sum_sq = float(diag[bi]) * tokens[b] ** 2
            var_sample = (sum_sq - norm_sum_sq / n) / (n - 1)
            within.append(n * var_sample / tokens[b] ** 2)
        mean_sq = float(diag.mean())
        signal = _offdiag_mean(go[:, o, :, o])
        cos = go[:, o, :, o] / torch.sqrt(torch.outer(diag, diag))
        noise_within = sum(within) / len(within) if within else float("nan")
        q3[name] = {
            "batch_norm": float(diag.sqrt().mean()),
            "signal_sq": signal,
            "noise_within": noise_within,
            "noise_cross": mean_sq - signal,
            "snr_within": signal / noise_within if within else float("nan"),
            "snr_cross": signal / (mean_sq - signal),
            "cross_batch_cos": _offdiag_mean(cos),
        }

    q1 = {}
    for x_name in variants[1:]:
        x = names.index(x_name)
        t = names.index("D+" + x_name)
        n_d = torch.stack([go[b, 0, b, 0] for b in range(nb)]).sqrt()
        n_x = torch.stack([go[b, x, b, x] for b in range(nb)]).sqrt()
        n_t = torch.stack([go[b, t, b, t] for b in range(nb)]).sqrt()
        dot = torch.stack([go[b, 0, b, x] for b in range(nb)])
        dot_t = torch.stack([go[b, 0, b, t] for b in range(nb)])
        perp = (n_x**2 - dot**2 / n_d**2).clamp(min=0).sqrt()
        sig_d = _offdiag_mean(go[:, 0, :, 0])
        sig_x = _offdiag_mean(go[:, x, :, x])
        cross = _offdiag_mean(go[:, 0, :, x])
        mean_sq_x = float((n_x**2).mean())
        ratios = [math.sqrt(r["n2"][x] / r["n2"][0]) for r in samples if r["n2"][0] > 0]
        coss = [
            float(r["dotD"][x]) / math.sqrt(r["n2"][0] * r["n2"][x])
            for r in samples
            if r["n2"][0] > 0 and r["n2"][x] > 0
        ]
        q1[x_name] = {
            "ratio_batch": float((n_x / n_d).mean()),
            "cos_batch": float((dot / (n_d * n_x)).mean()),
            "alpha_batch": float((dot / n_d**2).mean()),
            "perp_ratio_batch": float((perp / n_d).mean()),
            "total_ratio_batch": float((n_t / n_d).mean()),
            "total_cos_batch": float((dot_t / (n_d * n_t)).mean()),
            "total_norm_batch": float(n_t.mean()),
            "direct_norm_batch": float(n_d.mean()),
            "ratio_expect": _safe_sqrt(sig_x / sig_d) if sig_d > 0 else float("nan"),
            "cos_expect": (
                cross / math.sqrt(sig_d * sig_x)
                if sig_d > 0 and sig_x > 0
                else float("nan")
            ),
            "alpha_expect": cross / sig_d if sig_d > 0 else float("nan"),
            "signal_fraction": sig_x / mean_sq_x if mean_sq_x > 0 else float("nan"),
            "sample_ratio_median": _median(ratios),
            "sample_cos_median": _median(coss),
        }
    return {"num_batches": nb, "q1": q1, "q3": q3}


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(torch.tensor(values, dtype=torch.float64).median())


def new_kl_to_go_stats() -> torch.Tensor:
    """Running sums [n, sum R, sum R^2, sum (R-b), sum (R-b)^2] over response positions."""
    return torch.zeros(5, dtype=torch.float64)


def add_kl_to_go_stats(
    acc: torch.Tensor,
    kl_to_go: torch.Tensor,
    baseline: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    sel = mask > 0
    r = kl_to_go[sel].double().cpu()
    rb = (kl_to_go - baseline)[sel].double().cpu()
    acc += torch.stack(
        [
            r.new_tensor(float(r.numel())),
            r.sum(),
            (r * r).sum(),
            rb.sum(),
            (rb * rb).sum(),
        ]
    )


def summarize_kl_to_go(acc: torch.Tensor) -> dict[str, float]:
    n, s, s2, sb, sb2 = acc.tolist()
    if n == 0:
        return {}
    mean, mean_b = s / n, sb / n
    var, var_b = s2 / n - mean**2, sb2 / n - mean_b**2
    return {
        "positions": n,
        "mean_R": mean,
        "var_R": var,
        "second_moment_R": s2 / n,
        "mean_R_minus_b": mean_b,
        "var_R_minus_b": var_b,
        "second_moment_R_minus_b": sb2 / n,
        "var_ratio": var_b / var if var > 0 else float("nan"),
        "second_moment_ratio": sb2 / s2 if s2 > 0 else float("nan"),
    }


def summarize_branches(
    records: list[dict[str, Any]], key: str, num_entropy_bins: int = 3
) -> dict[str, Any]:
    """Q2 / Q3 logit-space metrics pooled over branched positions for one (mode, k).

    Pooling sums numerators and denominators over positions, so the signal ratio and
    cosine are those of the per-position logit gradients concatenated over positions.
    """
    recs = [r for r in records if key in r["stats"]]
    if not recs:
        return {}

    def pooled(rs: list[dict[str, Any]]) -> dict[str, float]:
        st = [r["stats"][key] for r in rs]
        tot = {f: sum(s[f] for s in st) for f in st[0]}
        n = len(st)
        signal2, d2 = tot["signal2"], tot["d2"]
        cond = (tot["between"] + tot["within"]) / n
        out = {
            "positions": n,
            "mean_R": tot["mean_R"] / n,
            "cond_var": cond,
            "between": tot["between"] / n,
            "within": tot["within"] / n,
            "between_share": tot["between"] / n / cond if cond > 0 else float("nan"),
            "signal_ratio": _safe_sqrt(signal2 / d2) if d2 > 0 else float("nan"),
            "signal_cos": (
                tot["cross"] / math.sqrt(signal2 * d2)
                if signal2 > 0 and d2 > 0
                else float("nan")
            ),
        }
        for b in ["none", "hist", "oracle"]:
            noise = tot[f"m2_{b}"] - signal2
            out[f"noise_{b}"] = noise / n
            # signal2 <= 0: no signal detectable above the sampling noise
            out[f"snr_{b}"] = (
                signal2 / noise if noise > 0 and signal2 > 0 else float("nan")
            )
        for b in ["hist", "oracle"]:
            out[f"noise_{b}_over_none"] = (
                out[f"noise_{b}"] / out["noise_none"]
                if out["noise_none"] > 0
                else float("nan")
            )
        scale = sum(abs(s["R_rollout"]) for s in st) / n
        out["replay_max_rel_err"] = max(s["replay_err"] for s in st) / max(scale, 1e-12)
        out["replay_mean_rel_err"] = (
            sum(s["replay_err"] for s in st) / n / max(scale, 1e-12)
        )
        return out

    summary = pooled(recs)
    ent = torch.tensor([r["entropy"] for r in recs])
    order = ent.argsort()
    bins = []
    for chunk in order.chunk(num_entropy_bins):
        rs = [recs[int(i)] for i in chunk]
        if rs:
            p = pooled(rs)
            bins.append(
                {
                    "entropy_lo": float(ent[chunk].min()),
                    "entropy_hi": float(ent[chunk].max()),
                    "positions": p["positions"],
                    "between": p["between"],
                    "within": p["within"],
                    "signal_ratio": p["signal_ratio"],
                    "snr_none": p["snr_none"],
                }
            )
    summary["entropy_bins"] = bins
    return summary
