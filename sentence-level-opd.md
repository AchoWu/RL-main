# SLOPD: Sentence-Level On-Policy Distillation

> Design note for a sentence-granularity variant of On-Policy Distillation (OPD) in NeMo-RL.
> Status: design phase, not yet implemented.

## 1. Motivation

### 1.1 Token is implementation, not reasoning

Current OPD (and all major distillation variants — vanilla KD, GKD, TIP, TIDE) operates at **token level**: for each token position, align teacher and student distributions over the vocabulary.

But token is the wrong granularity for reasoning distillation:

- A reasoning trajectory is a sequence of **reasoning steps** (set up equation → expand → simplify → conclude). Each step is roughly a sentence.
- Token-level KL aligns surface symbols (which `×` symbol, which whitespace pattern), not reasoning operations.
- Two semantically equivalent sentences ("12 × (10+3)" vs "(10+3) × 12") have very different token-level KL but should be treated as the same reasoning act.

### 1.2 Token-level OPD has structural problems we cannot fix at token level

| Problem | Why token-level cannot fix it |
|---------|-------------------------------|
| **Top-k projection bias** | KL on top-64 tokens loses tail mass; tail signal is not recoverable at token granularity without sending full vocab |
| **OOD state corruption** | After student errors, downstream states are OOD for teacher; teacher's per-token logits are unreliable but token-level OPD treats them as ground truth |
| **Multiple correct paths** | Token-level KL forces matching one specific surface form, suppressing valid alternatives |
| **Surface-form noise** | Token-level KL spends gradient on stylistic differences (article choice, tokenization variants) instead of reasoning structure |

### 1.3 The hypothesis

**Reasoning distillation should operate at sentence granularity.** Specifically:

- The unit of supervision is a sentence, not a token.
- The "distribution" over which we align teacher and student is a small set of candidate sentences, not the full vocabulary.
- Multiple candidate sentences are sampled from the teacher (plus the student's own sentence) to form a candidate pool; teacher and student both score this pool; KL is computed on the pool.

This is the **direct sentence-level analog of token-level OPD**: replace "vocabulary" with "candidate pool", replace "next-token logits" with "sentence-level log-likelihood".

## 2. Method: SLOPD

### 2.1 High-level protocol

For each prompt:

```
1. Student rollout to produce trajectory τ
2. Segment τ into sentences [s_1, s_2, ..., s_n]
3. For each sentence s_i:
   a. prefix_i = prompt + s_1 + ... + s_{i-1}
   b. Teacher samples K-1 candidate continuations from prefix_i
   c. Candidate pool C_i = {teacher candidates} ∪ {s_i}, |C_i| = K
   d. Teacher and student each compute log-likelihood of every candidate in C_i
   e. Length-normalize, softmax, compute KL
4. Total loss = Σ_i KL_i
```

The student's actual sentence `s_i` is always included in the pool — this preserves the on-policy nature: when student happens to produce a sentence the teacher also likes, the gradient reinforces it.

### 2.2 Sentence log-likelihood

For a candidate sentence $c$ tokenized as $(t_1, t_2, \ldots, t_L)$ following prefix $\text{pre}$:

$$
\log P_M(c \mid \text{pre}) = \sum_{j=1}^{L} \log P_M(t_j \mid \text{pre}, t_{<j})
$$

This is the standard sequence log-likelihood from any autoregressive LM.

### 2.3 Length normalization

Raw sequence log-likelihoods are dominated by length (longer candidate → more negative). To compare candidates of different lengths fairly, normalize per token:

$$
\overline{\log P}_M(c) = \frac{1}{L} \log P_M(c \mid \text{pre})
$$

This represents the model's **average per-token confidence** on the candidate, length-invariant.

### 2.4 Pool-level KL

Define pool-level distributions via softmax over normalized log-likelihoods:

$$
\tilde p_T^{(k)} = \frac{\exp(\overline{\log P}_T(c_k))}{\sum_{k'=1}^{K} \exp(\overline{\log P}_T(c_{k'}))}
$$

$$
\tilde q_S^{(k)} = \frac{\exp(\overline{\log P}_S(c_k))}{\sum_{k'=1}^{K} \exp(\overline{\log P}_S(c_{k'}))}
$$

Per-sentence loss is forward KL:

$$
\boxed{\quad \mathcal{L}_i = \mathrm{KL}(\tilde p_T \,\|\, \tilde q_S) = \sum_{k=1}^{K} \tilde p_T^{(k)} \log\frac{\tilde p_T^{(k)}}{\tilde q_S^{(k)}} \quad}
$$

Total loss:

$$
\mathcal{L}^{\text{SLOPD}} = \mathbb{E}_{\tau \sim q_{\theta_{\text{old}}}}\!\Bigg[\sum_{i=1}^{n} \mathcal{L}_i\Bigg]
$$

### 2.5 Worked example

**Prompt**: "Compute 12 × 13."

**Student rollout**: `"Let me think step by step.\nI'll compute it directly: 156.\nSo the answer is 156."`

Segment: $s_1, s_2, s_3$.

For $i=2$, $\text{prefix}_2 = $ "Compute 12 × 13.\nLet me think step by step.\n", actual $s_2 = $ "I'll compute it directly: 156."

Teacher samples K-1=3 continuations:
- $c_1$: "12 × 13 = 156." (6 tokens)
- $c_2$: "Using distributive: 12 × (10+3)." (8 tokens)
- $c_3$: "First, 12 × 10 = 120." (9 tokens)
- $c_4 = s_2$: "I'll compute it directly: 156." (7 tokens)

Compute logprobs (length-normalized):

| Candidate | $\overline{\log P}_T$ | $\overline{\log P}_S$ |
|-----------|----------------------|----------------------|
| $c_1$ | -0.533 | -1.250 |
| $c_2$ | -1.063 | -0.750 |
| $c_3$ | -1.122 | -0.578 |
| $c_4$ | -0.686 | -0.429 |

Softmax to get pool distributions:

| Candidate | $\tilde p_T$ | $\tilde q_S$ |
|-----------|------------|------------|
| $c_1$ | 0.333 | 0.145 |
| $c_2$ | 0.196 | 0.240 |
| $c_3$ | 0.185 | 0.285 |
| $c_4$ | 0.286 | 0.330 |

KL($\tilde p_T \| \tilde q_S$) ≈ 0.116. This is $\mathcal{L}_2$.

Interpretation: teacher prefers $c_1$ (direct answer); student over-allocates probability to $c_3$ (step-by-step expansion). Gradient pushes student toward teacher's preference shape.

## 3. Why this is better than token-level OPD

### 3.1 Granularity matches reasoning structure

Sentences are the natural unit of reasoning steps. Aligning at this level transfers reasoning operators rather than surface symbols.

### 3.2 Naturally handles equivalent paraphrases

If teacher samples both "12 × (10+3)" and "(10+3) × 12" into the candidate pool, both receive similar $\tilde p_T$ mass. Student does not need to commit to one surface form.

### 3.3 No top-k vocabulary projection problem

The distribution is over K (≈ 4-8) sentences, not over a 150k-token vocabulary. There is no "top-k cutoff" issue and no out-of-top-k token to handle. The candidate pool itself defines the support.

### 3.4 Teacher compute reduces, not increases

| Method | Teacher forwards per trajectory |
|--------|-------------------------------|
| Token-level OPD | $N$ (one per token, $N \approx 1000$ for long CoT) |
| SLOPD | $M \cdot K$ where $M \approx N/30$ (one per sentence × K candidates) |
| With $K=5$, $M=N/30$ | $N/6$ (≈ 6× cheaper than token-level) |

Even though teacher now needs to *generate* candidate sentences (not just forward), the total token count touched by teacher is smaller because we skip teacher forwards on student-only intermediate tokens.

### 3.5 OOD-state robustness

Teacher generates its candidates from $\text{prefix}_i$. Even if prefix is mildly OOD due to earlier student errors, teacher's *generative* output on a complete-sentence prefix is far more reliable than its *next-token logit* on a mid-sentence partially-corrupted state.

## 4. Theoretical positioning

### 4.1 What SLOPD is not

- **Not** a true sentence-space KL — sentence space is unbounded; we cannot define a normalized distribution over it without a support set.
- **Not** an unbiased estimator of any closed-form objective — the candidate pool is sampling-dependent.

### 4.2 What SLOPD is

A **sample-based KL estimator on a teacher-induced support**. The K-element pool is a stochastic proxy for "the set of sentences teacher would say here". KL on this proxy approximates teacher-vs-student divergence over teacher-likely continuations.

This is structurally analogous to PPO's KL penalty (which is also sample-based, not closed-form). The estimator has variance, but practical experience with PPO suggests $K=4$-$8$ samples give stable training signal.

### 4.3 Connection to token-level OPD

Token-level OPD with top-k projection is a **degenerate special case** of SLOPD where:
- Each "sentence" is a single token
- "Candidate pool" is teacher's top-k tokens
- Length normalization is trivial (length=1)

SLOPD generalizes this in two directions: longer units, and pool defined by teacher *generation* rather than *vocabulary top-k*.

## 5. Implementation

### 5.1 Data flow

```
vLLM rollout              → τ as token sequence
sentence segmenter        → [s_1, ..., s_n] (token spans)
per-sentence loop:
  prefix construction     → prefix_i (token sequence)
  teacher sentence sampling → K-1 candidates from teacher policy on prefix_i
  pool construction       → C_i = teacher_candidates ∪ {s_i}
  teacher logprob compute → log_p_T[1..K] via teacher forward on prefix+candidate
  student logprob compute → log_q_S[1..K] via student forward on prefix+candidate
  length normalize        → log_p_T_norm, log_q_S_norm
  softmax                 → tilde_p_T, tilde_q_S over K
  KL                      → L_i
sum over sentences        → total loss
```

### 5.2 Step 4 detail: candidate logprob computation

```python
def compute_candidate_logprob(model, prefix_ids, cand_ids):
    """
    prefix_ids: [P] token ids for prefix
    cand_ids:   [L] token ids for candidate
    Returns:    scalar = log P_model(cand | prefix)
    """
    full_ids = torch.cat([prefix_ids, cand_ids])      # [P+L]
    logits = model(full_ids.unsqueeze(0)).logits[0]   # [P+L, V]
    
    # Position P-1 predicts cand_ids[0]
    # Position P+L-2 predicts cand_ids[L-1]
    relevant_logits = logits[P-1 : P+L-1]             # [L, V]
    log_probs = F.log_softmax(relevant_logits, dim=-1)
    token_lp = log_probs.gather(-1, cand_ids.unsqueeze(-1)).squeeze(-1)  # [L]
    return token_lp.sum()                              # scalar
```

Teacher forward is `torch.no_grad()`. Student forward retains grad.

### 5.3 Step 5 detail: KL on pool

```python
def pool_kl(log_p_T_raw, log_q_S_raw, lengths):
    """
    log_p_T_raw, log_q_S_raw: [K] raw sequence logprobs
    lengths:                  [K] candidate lengths
    """
    # Length normalize
    log_p_T_norm = log_p_T_raw / lengths
    log_q_S_norm = log_q_S_raw / lengths
    
    # Softmax in log-space (numerically stable)
    log_p_T = log_p_T_norm - torch.logsumexp(log_p_T_norm, dim=0)
    log_q_S = log_q_S_norm - torch.logsumexp(log_q_S_norm, dim=0)
    
    p_T = log_p_T.exp()
    
    # Forward KL
    kl = (p_T * (log_p_T - log_q_S)).sum()
    return kl
```

### 5.4 Sentence segmenter

Primary segmentation: punctuation-based (`.`, `?`, `!`, `\n`) with LaTeX/code protection.

For math/CoT (primary target):
- Split on `.\n`, `\n`, `?`, `!`
- Protect `\d+\.\d+` (decimals) and LaTeX environments
- Treat reasoning markers (`Therefore`, `Step \d+:`, `So`) as boundary hints

Fallback: `nltk.punkt` with custom abbreviation list.

Pathological cases (very long sentence > 200 tokens, very short sentence < 5 tokens) are merged with neighbors.

### 5.5 Teacher candidate sampling

```python
def sample_teacher_candidates(teacher, prefix_ids, K, max_sent_len=80):
    """
    Sample K-1 diverse next-sentence continuations from teacher.
    """
    candidates = []
    # Oversample for dedup
    for _ in range(2 * (K - 1)):
        gen_ids = teacher.generate(
            prefix_ids,
            max_new_tokens=max_sent_len,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            stop_sequences=[".", "?", "!", "\n"]
        )
        candidates.append(gen_ids)
    
    # Dedup by embedding similarity (or simple string match for first pass)
    candidates = dedup(candidates, threshold=0.95)
    
    # Take top K-1 by teacher's own likelihood
    candidates = sorted(candidates, key=lambda c: teacher_score(c), reverse=True)
    return candidates[:K-1]
```

### 5.6 Batching

For each prefix, K candidates form a natural batch:

```python
# Pad to max length, batch forward
max_len = max(len(c) for c in candidate_ids_list)
padded = pad_to(candidate_ids_list, max_len)
attention_mask = make_mask(candidate_ids_list, max_len)

# Single forward call instead of K calls
prefix_repeated = prefix_ids.unsqueeze(0).expand(K, -1)
inputs = torch.cat([prefix_repeated, padded], dim=1)  # [K, P+max_len]
mask = torch.cat([torch.ones_like(prefix_repeated), attention_mask], dim=1)

logits = student(inputs, attention_mask=mask).logits  # [K, P+max_len, V]
# Extract per-candidate logprob using offsets
```

### 5.7 Optimization: shared prefix forward

The K candidates share the same prefix. Naive batching recomputes prefix attention K times. Optimization: forward the prefix once, cache KV, then forward each candidate continuation reusing the cached KV.

vLLM and most inference engines support this via prefix caching. For training, a simple reuse via `past_key_values` works.

This optimization is Phase 2; Phase 1 uses naive batching.

## 6. Configuration schema

`examples/configs/distillation_math.yaml` additions:

```yaml
loss_fn:
    distillation_type: "sentence"           # "token" (current) | "sentence" (SLOPD)
    
    # Sentence-level specific
    sentence_K: 5                           # candidate pool size (incl. student's sentence)
    sentence_segmenter: "math_aware"        # "math_aware" | "punkt" | "newline_only"
    sentence_max_len: 80                    # max tokens per teacher candidate
    sentence_min_len: 5                     # merge sentences shorter than this
    sentence_max_len_merge: 200             # split sentences longer than this
    sentence_length_normalize: true         # length-normalize logprobs before softmax
    sentence_softmax_temperature: 1.0       # softmax temp on logprobs
    sentence_kl_direction: "forward"        # "forward" | "reverse" | "symmetric"
    sentence_dedup_threshold: 0.95          # candidate dedup similarity
    sentence_oversample_factor: 2           # how many to sample before dedup → top K-1
```

Default `distillation_type: "token"` preserves baseline behavior.

## 7. Validation plan

### 7.1 Sanity checks (must pass)

1. **Distillation type fallback**: with `distillation_type: "token"`, training run is bit-exact to current OPD (50 steps, same seed).

2. **K=1 degenerate test**: with K=1 (no teacher candidates, only student's sentence), KL is identically 0 (single-element softmax gives mass 1.0 to the only candidate). Loss should be 0; gradients should be 0. This validates the pool mechanics.

3. **Teacher=student degenerate test**: load student's checkpoint into teacher. All log_p_T should equal log_q_S; KL should be ≈ 0; loss curves should look like a no-op.

4. **Segmenter unit tests**: hand-crafted strings with LaTeX, decimals, multi-punctuation; verify boundaries are correct.

### 7.2 Diagnostic metrics (per training step)

- `slopd/avg_kl_per_sentence`: mean $\mathcal{L}_i$
- `slopd/student_in_pool_rank`: where student's sentence ranks among K by $\tilde p_T$ (1 = teacher loves it, K = teacher hates it). Should trend toward 1 over training.
- `slopd/effective_K`: post-dedup average pool size. Should stay close to configured K; if drops to 2-3, candidate diversity is insufficient.
- `slopd/sentence_length_dist`: histogram. Catches segmenter pathology.
- `slopd/teacher_avg_logprob_on_student_sent`: $\overline{\log P}_T(s_i)$. Increasing = student writing more teacher-like sentences.

### 7.3 Main experiments

Setup: Qwen3-1.7B ← Qwen3-4B, TP=2 CP=2, 8 GPU, DeepScaler, max_steps=1000, seed=42.

| Experiment | Granularity | K | Segmenter | Notes |
|-----------|-------------|---|-----------|-------|
| Baseline | token | – | – | vanilla OPD |
| TIP-baseline | token | – | – | TIP token selection (existing strong baseline) |
| **SLOPD-main** | sentence | 5 | math_aware | **main result** |
| Ablate K | sentence | {2, 3, 5, 8} | math_aware | K choice |
| Ablate segmenter | sentence | 5 | {newline_only, punkt, math_aware} | segmenter robustness |
| Ablate length-norm | sentence | 5 | math_aware | normalization on/off |
| Ablate KL direction | sentence | 5 | math_aware | forward/reverse/symmetric |
| Hybrid | mixed | – | – | SLOPD + per-token KL with $\alpha=0.5$ |

Eval: MATH-500, AIME 2024/2025, GSM8K.

### 7.4 Ablation hypotheses

- **Granularity matters most on long CoT**: SLOPD vs token-OPD gap should be largest on AIME (long CoT, high mistake-propagation), smaller on GSM8K (short, less mistake-propagation).
- **K=4-5 is sufficient**: K>5 marginal gain plateaus; K=2 (only one teacher candidate) underperforms.
- **Segmenter is non-critical above a threshold**: math_aware ≈ punkt > newline_only; method tolerates moderate boundary noise.
- **Length normalization is necessary**: ablation without normalization should show training instability or degeneration.

## 8. Code change estimate

| # | File | Change | LOC |
|---|------|--------|-----|
| 1 | `algorithms/loss_functions.py` | New `SentenceLevelDistillationLossFn` class | +200 |
| 2 | `algorithms/distillation.py` | Branch on `distillation_type` config | +20 |
| 3 | `algorithms/sentence_segmenter.py` | New file: segmenter + tests | +150 |
| 4 | `models/policy/lm_policy.py` | Teacher candidate sampling API | +50 |
| 5 | `models/policy/workers/dtensor_policy_worker_v2.py` | Sentence-batched forward path | +80 |
| 6 | `examples/configs/distillation_math.yaml` | New config block | +12 |
| 7 | `tests/unit/algorithms/test_slopd.py` | Unit tests | +150 |

**Total ~660 lines.** Significantly more than TIDE's 63 lines, reflecting the genuine architectural change rather than a per-token reweighting tweak.

## 9. Known limitations

1. **Sample-based KL has variance**: pool composition is stochastic. Multi-seed runs must report variance bars. Practical mitigation: K≥4 and consistent random seed across rollout/training.

2. **Segmenter is method-critical**: in pathological inputs (no punctuation, e.g., raw stream output), method degrades. Mitigation: enforce min/max sentence length; fall back to fixed-window splits as last resort.

3. **Teacher generation cost**: teacher must now sample candidates (more than just forward). Net teacher compute is lower than token-level (Section 3.4), but per-prompt latency may be higher due to autoregressive generation. Mitigation: batch candidate generation across prompts in the same step.

4. **Multi-turn / tool-use rollouts**: current formulation assumes single-turn. Multi-turn requires re-deciding sentence boundaries across turn breaks; left as future work.

5. **Megatron path**: not supported initially. Same restriction as current OPD.

## 10. Relation to existing work

- **Token-level OPD (current NeMo-RL)**: degenerate special case (sentence = token, pool = vocab top-k).
- **TIP** (Token Importance in OPD): operates at token level, selects informative tokens via entropy/divergence. Orthogonal to SLOPD — TIP and SLOPD can compose (apply TIP-style sentence selection on top of SLOPD).
- **Process Reward Models (PRMs)**: provide step-level signal at *inference time*. SLOPD brings step-level supervision into *training time*, in distillation form rather than RL reward form.
- **GKD** (Generalized Knowledge Distillation): mixes teacher and student rollouts at *trajectory* level. SLOPD operates within a single (student) trajectory but aligns at sentence level — orthogonal axis.
- **Listwise ranking distillation** (IR literature): SLOPD's pool-level KL is structurally a listwise ranking loss, where teacher's preference order over the candidate pool is the target. The novel contribution is applying this to LLM reasoning with on-policy student rollouts.
- **DAGGER** (imitation learning): DAGGER addresses distribution mismatch by querying expert at student-visited states. Token-level DAGGER on LLMs is overly fine-grained (any token mismatch counts). SLOPD is effectively *sentence-level DAGGER*: query teacher at the right granularity, where "first divergence" detection becomes optional rather than required.

## 11. Naming

**SLOPD** = **S**entence-**L**evel **O**n-**P**olicy **D**istillation.

Emphasizes:
- **Sentence-Level**: granularity is the central design choice and primary novelty.
- **On-Policy**: trajectory still comes from student rollout, preserving OPD's exposure-bias benefit.
- **Distillation**: target remains teacher behavior matching, not RL-style reward maximization.

## 12. Open questions

1. **Optimal K**: theoretical minimum needed for low-variance KL? May depend on teacher/student divergence magnitude.
2. **Adaptive K per sentence**: high-uncertainty sentences (large student entropy) might benefit from larger K; routine sentences could use K=2.
3. **Pool composition strategy**: should we always include student's sentence? What if student's sentence is so bad it dominates the softmax denominator and corrupts the gradient?
4. **Cross-sentence dependencies**: current loss treats sentences independently. Reasoning has long-range structure (later sentences depend on earlier choices). A trajectory-level coherence term may help — left as future work.
5. **Teacher candidate diversity collapse**: at high training temperature, teacher candidates may all sound alike. Need empirical study of when oversample×dedup is sufficient and when active diversity injection (e.g., prompt-based "give me a different approach") is needed.
