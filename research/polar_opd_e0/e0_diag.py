"""PolarOPD E0: zero-training diagnostic of the order x spectrum decomposition of OPD.

Three stages, each runnable on its own GPU shard:

  gen     student samples responses to DeepScaleR train prompts with vLLM
          (same cot.txt prompt and T=1 sampling as examples/configs/distillation_math.yaml)
          and grades them with math_verify (same call as HFVerifyWorker).
  score   student and teacher re-score every response with HF transformers; per response
          position we compute the polar decomposition on the full vocab and on the teacher
          top-K window that DistillationLossFn actually trains on.
  report  merges all shards and prints / dumps the summary.

Notation (per position h, s = student, t = teacher, y = sampled token):
  K1      = log t(y) - log s(y)                          single-sample OPD advantage
  A_ord   = log sigma_s(r_t(y)) - log s(y)               PolarOPD lambda=0 advantage
  KL      = KL(s || t)                                   dense reverse-KL OPD loss
  D_ord   = KL(s || s^pi)    s^pi = student spectrum in teacher order
  D_spec  = KL(sigma_s || sigma_t) = KL(s || placebo)    placebo = teacher spectrum in student order
  KL_T    = KL(s || t^(1/T) / Z)                         teacher tempered by T (monotone -> order unchanged)

The question E0 answers: how much of OPD's signal is order (teacher knowledge in the PolarOPD sense)
vs spectrum, and how much of the spectral part is a per-context miscalibration that no global
teacher temperature removes (the only regime where PolarOPD can beat tempered OPD).
"""

import argparse
import glob
import json
import os
import random
import time

import numpy as np

ENTROPY_BIN_EDGES = [0.1, 0.5, 1.5]  # student entropy (nats); 4 bins
TEMPS = np.exp(np.linspace(np.log(1 / 8), np.log(8), 25))  # includes T=1 at index 12


# ----------------------------------------------------------------------------- gen
def load_problems(n_problems, seed):
    from datasets import load_dataset

    ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    ds = ds.shuffle(seed=seed)
    return [
        dict(pidx=i, problem=ds[i]["problem"], gold=str(ds[i]["answer"]))
        for i in range(min(n_problems, len(ds)))
    ]


def make_grader():
    import logging

    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    logging.getLogger("math_verify").setLevel(logging.CRITICAL)
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )

    def grade(response, gold):
        try:
            score, _ = verify_func(["\\boxed{" + gold + "}"], [response])
            return float(score) > 0.5
        except (
            Exception,
            TimeoutException,
        ):  # TimeoutException subclasses BaseException
            return False

    return grade


def cmd_gen(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.student)
    prompt_tmpl = open(args.prompt_file, encoding="utf-8").read()
    problems = load_problems(args.n_problems, args.seed)[args.shard :: args.num_shards]
    prompt_ids = []
    for p in problems:
        msg = tok.apply_chat_template(
            [{"role": "user", "content": prompt_tmpl.format(p["problem"])}],
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        prompt_ids.append(tok(msg, add_special_tokens=False)["input_ids"])

    max_prompt = max(len(x) for x in prompt_ids)
    llm = LLM(
        model=args.student,
        dtype="bfloat16",
        max_model_len=max_prompt + args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed + args.shard,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(
        n=args.n_samples,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_new_tokens,
        seed=args.seed + args.shard,
    )
    t0 = time.time()
    outs = llm.generate(
        [{"prompt_token_ids": x} for x in prompt_ids], sp, use_tqdm=True
    )
    print(f"[gen shard {args.shard}] generated in {time.time() - t0:.0f}s", flush=True)

    grade = make_grader()
    path = os.path.join(args.out_dir, f"gen_shard{args.shard}.jsonl")
    n_corr = n_tot = 0
    with open(path, "w", encoding="utf-8") as f:
        for p, pid, out in zip(problems, prompt_ids, outs):
            for j, o in enumerate(out.outputs):
                correct = grade(o.text, p["gold"])
                n_corr += correct
                n_tot += 1
                rec = dict(
                    pidx=p["pidx"],
                    sidx=j,
                    gold=p["gold"],
                    prompt_ids=pid,
                    response_ids=list(o.token_ids),
                    correct=correct,
                    truncated=o.finish_reason == "length",
                )
                f.write(json.dumps(rec) + "\n")
    print(
        f"[gen shard {args.shard}] {n_tot} responses, acc={n_corr / max(n_tot, 1):.3f} -> {path}"
    )


# ----------------------------------------------------------------------------- score
def answer_mask(tok, response_ids):
    r"""Boolean mask over response tokens overlapping the content of the last \boxed{...}."""
    pieces = tok.batch_decode([[i] for i in response_ids])
    text = "".join(pieces)
    start = text.rfind("\\boxed{")
    mask = np.zeros(len(response_ids), dtype=bool)
    if start < 0:
        return mask
    a = start + len("\\boxed{")
    depth, b = 1, None
    for k in range(a, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            b = k
            break
    if b is None or b == a:
        return mask
    cum = 0
    for i, pc in enumerate(pieces):
        if cum < b and cum + len(pc) > a:
            mask[i] = True
        cum += len(pc)
    return mask


def _grad(s, ls, lq):
    """Logit gradient of KL(s || sg(q)) w.r.t. student logits (the dense OPD update direction)."""
    a = ls - lq
    return s * (a - (s * a).sum(-1, keepdim=True))


def _cos(g1, g2):
    return (g1 * g2).sum(-1) / (g1.norm(dim=-1) * g2.norm(dim=-1) + 1e-30)


def polar_stats(ls, lt, y, K, temps):
    """All per-position statistics for one chunk. ls/lt: [P, V] fp32 log-probs, y: [P] sampled ids."""
    import torch

    P = ls.shape[0]
    ar = torch.arange(P, device=ls.device)
    s = ls.exp()
    lsy, lty = ls[ar, y], lt[ar, y]
    rs = (ls > lsy[:, None]).sum(-1)
    rt = (lt > lty[:, None]).sum(-1)
    sig_s, o_s = ls.sort(-1, descending=True)
    sig_t, o_t = lt.sort(-1, descending=True)
    out = dict(
        K1=lty - lsy,
        Aord=sig_s.gather(-1, rt[:, None])[:, 0] - lsy,
        rs=rs,
        rt=rt,
        H_s=-(s * ls).sum(-1),
        H_t=-(lt.exp() * lt).sum(-1),
        argmax_agree=o_s[:, 0] == o_t[:, 0],
        y_in_tK=rt < K,
        tmax_in_sK=(ls > ls.gather(-1, o_t[:, :1])).sum(-1) < K,
    )

    # full-vocab dense decomposition
    l_ord = torch.empty_like(ls).scatter_(
        -1, o_t, sig_s
    )  # student spectrum, teacher order
    l_pl = torch.empty_like(ls).scatter_(
        -1, o_s, sig_t
    )  # teacher spectrum, student order
    del sig_t
    out["KL"] = (s * (ls - lt)).sum(-1)
    out["Dord"] = (s * (ls - l_ord)).sum(-1)
    out["Dspec"] = (s * (ls - l_pl)).sum(-1)
    g_opd = _grad(s, ls, lt)
    out["cos_pl"] = _cos(g_opd, _grad(s, ls, l_pl))
    out["cos_ord"] = _cos(g_opd, _grad(s, ls, l_ord))
    del l_ord, l_pl, g_opd, sig_s, o_s
    kl_T = torch.stack(
        [(s * (ls - torch.log_softmax(lt / T, -1))).sum(-1) for T in temps], -1
    )

    # teacher top-K window, renormalised on both sides (DistillationLossFn, zero_outside_topk=false)
    idx = o_t[:, :K]
    sK = torch.log_softmax(ls.gather(-1, idx), -1)
    ltK_raw = lt.gather(-1, idx)
    tK = torch.log_softmax(
        ltK_raw, -1
    )  # already sorted: teacher order is the identity here
    pK = sK.exp()
    sig_sK, o_sK = sK.sort(-1, descending=True)
    l_plK = torch.empty_like(sK).scatter_(-1, o_sK, tK)
    out["KLk"] = (pK * (sK - tK)).sum(-1)
    out["Dordk"] = (pK * (sK - sig_sK)).sum(-1)
    out["Dspeck"] = (pK * (sK - l_plK)).sum(-1)
    g_opdK = _grad(pK, sK, tK)
    out["cos_plk"] = _cos(g_opdK, _grad(pK, sK, l_plK))
    out["cos_ordk"] = _cos(g_opdK, _grad(pK, sK, sig_sK))
    kl_Tk = torch.stack(
        [(pK * (sK - torch.log_softmax(ltK_raw / T, -1))).sum(-1) for T in temps], -1
    )
    return out, kl_T, kl_Tk


def cmd_score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(args.device)
    recs = []
    for path in sorted(glob.glob(os.path.join(args.out_dir, "gen_shard*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            recs.extend(json.loads(line) for line in f)
    recs.sort(key=lambda r: (r["pidx"], r["sidx"]))
    recs = [r for r in recs if len(r["response_ids"]) > 0][
        args.shard :: args.num_shards
    ]
    print(f"[score shard {args.shard}] {len(recs)} responses", flush=True)

    tok = AutoTokenizer.from_pretrained(args.student)
    tok_t = AutoTokenizer.from_pretrained(args.teacher)
    if tok.get_vocab() != tok_t.get_vocab():
        raise SystemExit(
            "student and teacher tokenizers differ; E0 needs a shared vocab"
        )
    V = len(tok)

    def load(path):
        for attn in ("flash_attention_2", "sdpa"):
            try:
                m = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype=torch.bfloat16, attn_implementation=attn
                )
                break
            except (ImportError, ValueError):
                continue
        m = m.to(dev).eval()
        # fp32 head: bf16 logits tie a lot in the tail, which corrupts the rank statistics
        W = m.get_output_embeddings().weight[:V].float()
        return m.get_decoder(), W

    dec_s, W_s = load(args.student)
    dec_t, W_t = load(args.teacher)
    temps = torch.tensor(TEMPS, device=dev, dtype=torch.float32)
    edges = torch.tensor(ENTROPY_BIN_EDGES, device=dev)
    nb = len(ENTROPY_BIN_EDGES) + 1

    pos = {}  # per-position arrays, concatenated over responses
    per_seq = dict(n=[], correct=[], truncated=[], pidx=[], sidx=[])
    ans_masks = []
    klT_bins = np.zeros((nb, len(TEMPS)))
    klTk_bins = np.zeros((nb, len(TEMPS)))
    t0 = time.time()
    for i, r in enumerate(recs):
        ids = torch.tensor([r["prompt_ids"] + r["response_ids"]], device=dev)
        Lp, Lr = len(r["prompt_ids"]), len(r["response_ids"])
        y_all = ids[0, Lp:]
        with torch.no_grad():
            h_s = dec_s(input_ids=ids).last_hidden_state[0, Lp - 1 : Lp - 1 + Lr]
            h_t = dec_t(input_ids=ids).last_hidden_state[0, Lp - 1 : Lp - 1 + Lr]
            chunks = []
            for c in range(0, Lr, args.chunk):
                sl = slice(c, c + args.chunk)
                ls = torch.log_softmax(h_s[sl].float() @ W_s.T, -1)
                lt = torch.log_softmax(h_t[sl].float() @ W_t.T, -1)
                out, kl_T, kl_Tk = polar_stats(ls, lt, y_all[sl], args.topk, temps)
                del ls, lt
                b = torch.bucketize(out["H_s"], edges, right=True)  # == np.digitize
                for k in range(nb):
                    m = b == k
                    if m.any():
                        klT_bins[k] += kl_T[m].sum(0).double().cpu().numpy()
                        klTk_bins[k] += kl_Tk[m].sum(0).double().cpu().numpy()
                mn, am = kl_T.min(-1)
                mnk, amk = kl_Tk.min(-1)
                out.update(
                    KLctxT=mn,
                    logTstar=temps.log()[am],
                    KLctxTk=mnk,
                    logTstark=temps.log()[amk],
                )
                chunks.append({k: v.cpu() for k, v in out.items()})
        for k in chunks[0]:
            v = torch.cat([c[k] for c in chunks]).numpy()
            v = (
                v.astype(np.float32)
                if v.dtype.kind == "f"
                else v.astype(np.int32 if v.dtype.kind in "iu" else bool)
            )
            pos.setdefault(k, []).append(v)
        for k, v in (
            ("n", Lr),
            ("correct", r["correct"]),
            ("truncated", r["truncated"]),
            ("pidx", r["pidx"]),
            ("sidx", r["sidx"]),
        ):
            per_seq[k].append(v)
        ans_masks.append(answer_mask(tok, r["response_ids"]))
        if (i + 1) % 10 == 0 or i + 1 == len(recs):
            print(
                f"[score shard {args.shard}] {i + 1}/{len(recs)}  {time.time() - t0:.0f}s",
                flush=True,
            )

    np.savez(
        os.path.join(args.out_dir, f"score_shard{args.shard}.npz"),
        **{f"pos_{k}": np.concatenate(v) for k, v in pos.items()},
        **{f"seq_{k}": np.array(v) for k, v in per_seq.items()},
        ans_mask=np.concatenate(ans_masks),
        klT_bins=klT_bins,
        klTk_bins=klTk_bins,
        temps=TEMPS,
        topk=args.topk,
    )


# ----------------------------------------------------------------------------- report
def auc(score, label):
    """Mann-Whitney AUC of score for label==True (ties get average rank)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or (~label).all():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_s = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calib_block(KL, KLctxT, klT_sum, n_tok, logTstar):
    """How much of the reverse KL a global vs a per-position teacher temperature can remove."""
    j = int(np.argmin(klT_sum))
    raw = float(KL.mean())
    glob_ = float(klT_sum[j] / n_tok)
    ctx = float(KLctxT.mean())
    w = KL / max(KL.sum(), 1e-30)
    mu = float((w * logTstar).sum())
    return dict(
        KL=raw,
        best_global_T=float(TEMPS[j]),
        KL_globalT=glob_,
        KL_ctxT=ctx,
        frac_removed_by_globalT=1 - glob_ / raw,
        frac_removed_only_by_ctxT=(glob_ - ctx) / raw,
        KLweighted_std_logTstar=float(np.sqrt((w * (logTstar - mu) ** 2).sum())),
    )


def cmd_report(args):
    files = sorted(glob.glob(os.path.join(args.out_dir, "score_shard*.npz")))
    zs = [np.load(f) for f in files]
    cat = lambda k: np.concatenate([z[k] for z in zs])  # noqa: E731
    P = {k[4:]: cat(k) for k in zs[0].files if k.startswith("pos_")}
    S = {k[4:]: cat(k) for k in zs[0].files if k.startswith("seq_")}
    ans = cat("ans_mask")
    klT_bins = sum(z["klT_bins"] for z in zs)
    klTk_bins = sum(z["klTk_bins"] for z in zs)
    K = int(zs[0]["topk"])
    n_tok = len(P["K1"])
    seq_id = np.repeat(np.arange(len(S["n"])), S["n"])
    tok_correct = S["correct"][seq_id].astype(bool)
    R = dict(
        n_responses=int(len(S["n"])),
        n_tokens=int(n_tok),
        accuracy=float(S["correct"].mean()),
        truncated=float(S["truncated"].mean()),
        mean_len=float(S["n"].mean()),
    )

    # 1) order vs spectrum share
    Ao, K1 = P["Aord"], P["K1"]
    nz = Ao != 0
    R["order_share"] = dict(
        frac_rank_agree_sampled=float((P["rs"] == P["rt"]).mean()),
        frac_argmax_agree=float(P["argmax_agree"].mean()),
        Dord_over_KL_full=float(P["Dord"].sum() / P["KL"].sum()),
        Dspec_over_KL_full=float(P["Dspec"].sum() / P["KL"].sum()),
        C_over_KL_full=float((P["KL"] - P["Dord"] - P["Dspec"]).sum() / P["KL"].sum()),
        Dord_over_KL_topk=float(P["Dordk"].sum() / P["KLk"].sum()),
        Dspec_over_KL_topk=float(P["Dspeck"].sum() / P["KLk"].sum()),
        C_over_KL_topk=float(
            (P["KLk"] - P["Dordk"] - P["Dspeck"]).sum() / P["KLk"].sum()
        ),
        var_K1=float(K1.var()),
        var_Aord=float(Ao.var()),
        corr_K1_Aord=float(np.corrcoef(K1, Ao)[0, 1]),
        sign_disagree_given_Aord_nonzero=float(
            (np.sign(K1[nz]) != np.sign(Ao[nz])).mean()
        )
        if nz.any()
        else float("nan"),
    )

    # 2) calibration: global vs per-context teacher temperature
    R["calibration_full"] = calib_block(
        P["KL"], P["KLctxT"], klT_bins.sum(0), n_tok, P["logTstar"]
    )
    R["calibration_topk"] = calib_block(
        P["KLk"], P["KLctxTk"], klTk_bins.sum(0), n_tok, P["logTstark"]
    )

    # 3) gradient geometry: is the dense OPD update mostly the placebo (spectrum-only) update?
    w, wk = P["KL"] / P["KL"].sum(), P["KLk"] / P["KLk"].sum()
    R["gradient"] = dict(
        cos_opd_placebo_full=float(np.nanmean(P["cos_pl"])),
        cos_opd_order_full=float(np.nanmean(P["cos_ord"])),
        cos_opd_placebo_full_KLw=float(np.nansum(w * P["cos_pl"])),
        cos_opd_order_full_KLw=float(np.nansum(w * P["cos_ord"])),
        cos_opd_placebo_topk_KLw=float(np.nansum(wk * P["cos_plk"])),
        cos_opd_order_topk_KLw=float(np.nansum(wk * P["cos_ordk"])),
    )

    # 4) sign noise on answer tokens (+ magnitude-weighted, since near-zero advantages flip sign for free)
    def sign_noise(A):
        a, c = A[ans], tok_correct[ans]
        bad = np.where(c, a < 0, a > 0)
        return dict(
            frac_bad=float(bad.mean()),
            frac_zero=float((a == 0).mean()),
            mag_weighted_bad=float(
                (np.abs(a) * bad).sum() / max(np.abs(a).sum(), 1e-30)
            ),
        )

    R["answer_tokens"] = dict(n=int(ans.sum()), K1=sign_noise(K1), Aord=sign_noise(Ao))

    # 5) does the signal know which responses are correct?
    def seq_mean(x, m=None):
        m = np.ones_like(x, bool) if m is None else m
        num = np.bincount(seq_id, weights=np.where(m, x, 0), minlength=len(S["n"]))
        den = np.bincount(seq_id, weights=m.astype(float), minlength=len(S["n"]))
        return num / np.maximum(den, 1), den > 0

    lab = S["correct"].astype(bool)
    R["correctness_auc"] = {}
    for name, x in (
        ("K1", K1),
        ("Aord", Ao),
        ("negKL", -P["KL"]),
        ("negDord", -P["Dord"]),
        ("negDspec", -P["Dspec"]),
    ):
        v, _ = seq_mean(x)
        va, has = seq_mean(x, ans)
        R["correctness_auc"][f"{name}_seqmean"] = auc(v, lab)
        R["correctness_auc"][f"{name}_answer"] = auc(va[has], lab[has])

    # 6) support barrier
    R["support"] = dict(
        frac_y_outside_teacher_topK=float((~P["y_in_tK"]).mean()),
        frac_teacher_argmax_outside_student_topK=float((~P["tmax_in_sK"]).mean()),
        K=K,
    )

    # 7) by student entropy
    b = np.digitize(P["H_s"], ENTROPY_BIN_EDGES)
    names = ["H<0.1", "0.1-0.5", "0.5-1.5", ">=1.5"]
    R["by_entropy"] = {}
    for k, nm in enumerate(names):
        m = b == k
        if not m.any():
            continue
        kl = P["KL"][m].sum()
        glob_ = klT_bins[k].min()
        R["by_entropy"][nm] = dict(
            token_share=float(m.mean()),
            KL_share=float(kl / P["KL"].sum()),
            rank_agree=float((P["rs"][m] == P["rt"][m]).mean()),
            Dord_over_KL=float(P["Dord"][m].sum() / kl),
            best_global_T=float(TEMPS[int(np.argmin(klT_bins[k]))]),
            frac_removed_by_globalT=float(1 - glob_ / kl),
            frac_removed_only_by_ctxT=float((glob_ - P["KLctxT"][m].sum()) / kl),
        )

    path = os.path.join(args.out_dir, "e0_report.json")
    json.dump(R, open(path, "w"), indent=1)
    print(json.dumps(R, indent=1))
    print(f"-> {path}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "score", "report"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--student")
    ap.add_argument("--teacher")
    ap.add_argument("--prompt_file", default="examples/prompts/cot.txt")
    ap.add_argument("--n_problems", type=int, default=256)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--topk", type=int, default=64, help="= distillation.topk_logits_k")
    ap.add_argument(
        "--chunk", type=int, default=512, help="positions per full-vocab chunk"
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    {"gen": cmd_gen, "score": cmd_score, "report": cmd_report}[args.stage](args)


if __name__ == "__main__":
    main()
