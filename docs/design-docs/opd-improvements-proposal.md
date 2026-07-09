# OPD 改进提案：Prefix-Length Warmup

> 状态：Draft v3 · 日期：2026-07-09 · 作者：howu

## 0. TL;DR

针对现有 On-Policy Distillation（OPD，实现见 `nemo_rl/algorithms/loss_functions.py` 的 `DistillationLossFn`、`nemo_rl/algorithms/distillation.py`、`examples/configs/distillation_math.yaml`），提出一项 loss-mask 侧的改动：**Prefix-Length Warmup** —— 前期只在序列前 N 个 token 上算 distillation loss，N 随训练步数按 schedule 从小递增到全长。

**核心假设**：当 student 在 CoT 中间某一步走错，后续 token 已经处在"错误轨迹"上，教师对这段 tail 的信号更多是"如何从错误里挣扎自救"而不是"如何正确推理"。用整段 tail 的 KL 直接更新 student，会让 student 学到"错了以后如何写得像 teacher"，而不是"如何一开始就别错"。前缀 warmup 让 student 先学会"起手做对"，再逐步放宽让它面对更长的自我生成 context。

**改动维度**：序列长度维度上的 loss curriculum。不改 rollout（student 照旧生成整条序列），不改 loss 数学形式，不引入新模型副本。**只改 mask**。

**为什么这一个 idea 而不是别的**：调研（§7）显示 REOPOLD / AdaKD / DOPD 已经从"逐 token 加权"角度覆盖了教师置信度 gating 方向；PPD / TRB / TrOPD 已经覆盖了 trust-region 方向。**"per-example 前缀截断 + 长度 warmup"在 LLM OPD 文献里没有直接对应工作**，是最干净的空白。

---

## 1. 背景与动机

### 1.1 当前 OPD 行为
- Loss 位于 `nemo_rl/algorithms/loss_functions.py:966-1226`（`DistillationLossFn`），基于 teacher 提供的 top-k logits + indices，在 k 维上对 student 做重归一化后计算 KL（forward / reverse / mixed）。
- 掩码合成在 `loss_functions.py:1211`：`mask = token_mask * sample_mask.unsqueeze(-1)`，随后 `masked_mean`。
- 训练循环见 `nemo_rl/algorithms/distillation.py:697-701`：每一步用当前 student rollout 一条 trajectory，让 teacher 对同一序列输出 top-k 分布，直接算 KL。
- **关键性质**：整条 student rollout 上所有 valid token 权重相同——无论它在序列头（还没跑偏）还是尾部（可能已经错到深处）。

### 1.2 问题陈述

CoT 推理任务里，一次 rollout 常常是这样的结构：

```
[prompt] [reasoning token 1..k, on-track] [token k+1 走错] [tail 100+ tokens, 沿错误路径展开]
```

此时对 tail 100+ tokens 计算 `KL(teacher ‖ student)` 有两个层次的问题：

1. **信号错位**。教师给出的"如果你已经写到这里、接下来该怎么写"是一种**条件正确**——它在错误 context 下"最合理续写"。student 学到的信号更接近"错了以后如何写得漂亮"，不是"如何一开始就别错"。
2. **前缀-尾部不对称**。前缀是接近 teacher 分布支持的合法 rollout；尾部是 student 已经偏出的 OOD 段。给两者相同权重相当于用最大权重去学 student 最不该模仿自己的部分。

### 1.3 为什么"截断后半段"比"逐 token gating"更贴 CoT 的实际

Idea 1（逐 token 教师置信度 gate）允许"错完又对回来"——某个高熵位置被压低，但两侧仍照常学。但 CoT 推理的错误通常**不可逆**：一步算错 → 后续所有步都在错误结论上做推理 → 教师在这段 tail 上的信号在语言学上是合法的但在**任务语义上**是无价值的（甚至有害）。

**单调性归纳偏置**：一旦 gate 关闭就不再打开。这个约束在 CoT 场景下比"允许打洞"更接近 ground truth 的错误传播结构。

### 1.4 与已有工作的关系（详见 §7）

- LLM OPD 侧未见 per-example 前缀截断或长度 warmup 的直接工作。
- 序列长度 curriculum 在 GRPO/RLHF 里存在，但通常是**筛例子**（短 example → 长 example），不是**单条 example 内截断**。
- Speculative decoding 训练里有 "teacher verifies student prefix" 的机制，但用于 draft model，不是 OPD。
- First-Token Distillation (FIRST) 是极端形式（只学首 token），offline。

---

## 2. 方法

### 2.1 核心机制

在 `DistillationLossFn` 的 mask 合成处（`loss_functions.py:1211`），额外乘一个**位置 mask** `w_pos[b, i]`：

```
mask = token_mask * sample_mask.unsqueeze(-1) * w_pos
```

其中 `w_pos[b, i]` 只是"当前 step 允许 loss 施加到 sample b 的位置 i"的指示。**Rollout 完全不动**——student 照旧生成整条序列，只是 loss 不算 tail。

### 2.2 三种变体（从简到繁）

| 变体 | `w_pos` 定义 | 特点 |
|------|-------------|------|
| **A. 固定长度前缀** | `w_pos[b, i] = 1{i ≤ N}`，N 是超参 | 消融基线；用于验证"截尾"本身有效 |
| **B. 长度 warmup（主推）** | `w_pos[b, i] = 1{i ≤ N(t)}`，N 按训练步 t 单调增长 | 核心 idea；`N(T_end) = L_max` |
| **C. 动态截断（延后）** | `w_pos[b, i] = 1{i < τ(b)}`，τ(b) = student 在序列 b 中首次"走偏"的位置 | 概念最贴但依赖 τ 的定义，见 §2.4 |

### 2.3 变体 B 的 schedule

**分段阶梯 schedule（首选）**：

```
N(t) / L_max = 0.25    for t ∈ [0,       0.2 · T]
             = 0.50    for t ∈ [0.2·T,   0.4 · T]
             = 0.75    for t ∈ [0.4·T,   0.6 · T]
             = 1.00    for t ∈ [0.6·T,   T]
```

**为什么分段而非线性**：
- 每个 plateau 让 student 在当前"允许长度"上稳定收敛，再放宽。这更接近 curriculum learning 的标准做法。
- 线性增长里"边界 token"的权重变化太连续，看不清哪个长度导致的收益。

**基线对照**：
- E-B0：full length 从头到尾（等价于 `N_0 = L_max`，即现状）
- E-B1：上述四段
- E-B2：更慢 warmup（六段，每段 T/6）
- E-B3：更快 warmup（两段，25% → 100%）

### 2.4 变体 C 的 τ 定义（可选，后期做）

变体 C 的核心难点是"student 在哪里开始跑偏"没有 canonical 定义。四种候选：

| 定义 | 计算成本 | 主要问题 |
|------|---------|---------|
| **τ1**: teacher log-prob(student_token_i) 首次跌破阈值 | 零成本（已有 `teacher_topk_logprobs` + `input_ids`） | 教师"没把握"≠"学生错了"；过渡 token 会被误判 |
| **τ2**: student 选的 token 不在 teacher top-k 里的首个位置 | 零成本 | 太粗；top-k=20，学生选第 21 名也算 |
| **τ3**: teacher-student token-KL 首次超过阈值的位置 | 零成本 | 最贴合"分歧点"，但 loss 本身就是 KL，会形成自相关（见下） |
| **τ4**: 基于最终 reward 反推分歧点 | 需 reward + 归因 | 理论最干净但归因难；且只对可 verify 任务可行 |

**τ3 的自相关问题**：如果用 KL 阈值 gate KL loss，阈值附近的 token 会反复进出 loss。**缓解**：门控信号用 stop-gradient 的 KL（`KL_gate = KL(...).detach()`），不让 gate 的开合梯度反向回到 KL 计算里。

**建议**：**先做 B，跑通再考虑 C**。C 需要一个额外的超参（阈值），且 τ 定义的选择本身就是一个独立研究问题，不应把 §2 的主 experiment 卡在这里。

### 2.5 与 Apple "Unmasking OPD" 的 tension

Apple 2026 ("Unmasking On-Policy Distillation", alphaxiv 2605.10889) 报告：**教师指导在学生"错误"轨迹上比"正确"轨迹上更接近 ideal gradient**——即错误轨迹整体上教师信号更有价值。

这看似和本 proposal 的直觉矛盾（"错误轨迹的 tail 信号差"），但**粒度不同**：
- Apple 的分析：整条 trajectory 按最终 reward 二分（对 / 错），比较**两组轨迹整体**的教师信号质量。
- 本 proposal：单条 trajectory **内部**，比较分歧点**前** vs. 分歧点**后**的教师信号质量。

两者不直接冲突。但需要在实验中直接回应：

**E-C1（可选验证实验，独立于主消融）**：把训练 rollout 按 `(correctness × position_relative_to_divergence)` 分四组，测每组 loss 对 val pass@1 的**边际贡献**。这个实验本身就有独立价值。

### 2.6 一个需要正视的失败模式

假设 student 在 token 50 走错。你只学 token 1..50。但如果 student **恰恰是因为** token 1..50 太自信、没在关键分歧位置学到"这里应该分散押注"才在 token 50 走错，那 gate 前 50 token 就是在 reinforce 让它走错的模式。

**症状**：val pass@1 短期看着还行（前缀学得更好），但**多样性坍缩**（一直走同一条错路径）。
**监测指标**：训练中 rollout 的 unique-completion rate 和 diversity（top-1 concentration）。若这些指标显著下降，就是这个失败模式在触发。
**缓解**：变体 B 的分段 warmup 本身就是缓解——最后阶段回到 full-length，会重新让 student 面对"错了以后如何"的信号。

---

## 3. 实现落点

### 3.1 配置扩展

`examples/configs/distillation_math.yaml`：

```yaml
loss_fn:
  kl_type: mixed
  mixed_kl_weight: 0.5
  zero_outside_topk: true
  prefix_length_warmup:
    mode: none              # one of: none | fixed | stepwise | dynamic
    fixed_prefix_ratio: 0.5 # only used for mode=fixed
    stepwise_schedule:      # only used for mode=stepwise
      - {until_step_frac: 0.2, prefix_ratio: 0.25}
      - {until_step_frac: 0.4, prefix_ratio: 0.50}
      - {until_step_frac: 0.6, prefix_ratio: 0.75}
      - {until_step_frac: 1.0, prefix_ratio: 1.00}
    dynamic_tau_kind: null  # only used for mode=dynamic; one of: teacher_logprob | out_of_topk | token_kl
    dynamic_tau_threshold: null
```

对应的 TypedDict 位于 `DistillationLossConfig`（`loss_functions.py:966-969`），加 `prefix_length_warmup: NotRequired[dict]`。

### 3.2 代码变更

1. **`DistillationLossConfig`**（`loss_functions.py:966`）扩字段。
2. **`DistillationLossFn.__init__`**（`loss_functions.py:984`）读 schedule，缓存成便于查询的形式。
3. **训练循环侧**（`distillation.py:697-701` 附近）把当前 `global_step` 和 `max_num_steps` 传到 loss（可以通过 `train_data` dict 传，也可以通过 `DistillationLossFn` 的状态）。这是**唯一需要触及主循环的改动**，因为 loss 本身没有 step 信息。
4. **`DistillationLossFn.__call__`** 内部（`loss_functions.py:1211` 附近）：
   - 计算 `N(t)` 或 `τ(b)` → 得到 `w_pos: [B, S-1]`
   - `mask = token_mask * sample_mask.unsqueeze(-1) * w_pos`
   - 其余不变
5. **指标**（`loss_functions.py:1221`）：
   - `prefix_ratio_current`: 当前 step 的 `N(t)/L_max`
   - `effective_valid_tokens`: `mask.sum()`（用于确认归一化是否合理）
   - `prefix_kl_mean` / `tail_kl_mean`: 用 `w_pos` 和 `1 - w_pos` 分别 mask 后算平均 KL，用于事后诊断（tail 上的 KL 是不是真的比 prefix 高？）

### 3.3 归一化的坑

现有 `masked_mean` 用 `global_normalization_factor=global_valid_toks`（`loss_functions.py:1216`）。加了 `w_pos` 之后有两种选择：

- **A**：`global_valid_toks` 不变，`w_pos` 只是把某些 token 权重压到 0。等价于"loss scale 随 warmup ratio 缩小"——早期 loss 天然变小，等效于 lr warmup。
- **B**：重新计算 `global_valid_toks` = 当前 mask 后的 valid tokens，让 loss 的绝对量级在 warmup 中保持稳定。

**推荐 B**：避免 loss scale 和 warmup schedule 耦合，实验更可解释。实现上要把 `w_pos` 也参与 `global_valid_toks` 的 all-reduce。

---

## 4. 消融矩阵

固定其他超参 + seed，在 `distillation_math` recipe 上跑：

| 实验 | mode | schedule / N | 目的 |
|------|------|-------------|------|
| E0   | none | — | baseline（full length，现状） |
| E-A1 | fixed | 0.5 | "截尾一半"本身有没有用？ |
| E-A2 | fixed | 0.25 | 截得更狠 |
| E-B1 | stepwise | 4段 25/50/75/100 | **主推 schedule** |
| E-B2 | stepwise | 6段 | 更慢 warmup |
| E-B3 | stepwise | 2段 25→100 | 更快 warmup |
| E-C1（可选） | dynamic | token_kl, τ=0.5 | 动态截断是否比固定长度更好 |

**主指标**：math validation pass@1（`val_reward`），以及 pass@k / diversity 指标（如果 recipe 支持）。
**副指标**：
- `prefix_kl_mean` vs `tail_kl_mean` 曲线：验证核心假设（tail KL 是否显著更高？如果不高，前提本身错）
- Rollout diversity（top-1 concentration）：监测 §2.6 说的多样性坍缩失败模式
- 训练 wall-clock：应该无变化或略降（warmup 期 loss token 更少）

**决策**：
1. E-A1 vs E0：若 E-A1 显著劣于 E0 → 核心假设错，整个方向砍掉。
2. E-A1 vs E-B1：若 stepwise 好于 fixed → 长度 warmup 确实带来额外收益。
3. E-B1 vs E-B2 vs E-B3：定 schedule sweet spot。

---

## 5. 兼容性 & 回滚

- 特性**默认关闭**（`mode: none`），完全保持现有 OPD 行为，无回归风险。
- Config 使用 `NotRequired[...]` 添加字段，符合 `CODING_GUIDELINES.md` 里的 "no code default" 规则——所有可选行为都在 yaml 里显式打开。
- 若消融结果不理想，直接从 yaml 关闭，或整个 PR 回滚，`DistillationLossFn` 主路径不受影响。
- **唯一稍侵入的点**：需要把 `global_step` / `max_num_steps` 从主循环传进 loss。可以通过在 `train_data` dict 里加一个 `_meta` 字段实现，不改函数签名。

---

## 6. 实施顺序与时间盒

| 阶段 | 内容 | 预计工作量 |
|------|------|-----------|
| Step 1 | 配置 + `w_pos` 实现（先只做 stepwise mode）+ metric | 0.5 天 |
| Step 2 | 把 `global_step` 传入 loss 的 plumbing + 单测 | 0.5 天 |
| Step 3 | 跑 E0 / E-A1 / E-A2，判断截尾方向 | 1–2 天（含训练 wall-clock） |
| Step 4 | 若 A 有信号：跑 E-B1 / E-B2 / E-B3 | 2 天 |
| Step 5 | 分析 `prefix_kl_mean` vs `tail_kl_mean`、diversity 曲线，写小结 | 0.5 天 |
| Step 6（可选） | 实现 dynamic τ + E-C1 | 2–3 天 |
| Step 7（可选） | Apple tension 验证实验（`correctness × pre/post-divergence` 四格） | 1–2 天 |

**关键决策点**：Step 3 之后判断是否继续。若 E-A1 明显劣于 E0，砍掉整个方向；若 E-A1 ~ E0，可能是 fixed 太粗，直接跳到 Step 4；若 E-A1 明显好，则整个假设成立，进入 Step 4。

---

## 7. 已知风险

1. **多样性坍缩**（§2.6）：主要风险，靠 rollout diversity 指标监测。
2. **归一化坑**（§3.3）：处理不好会让 warmup 期 loss 变小 → 隐式 lr warmup → 混淆实验解释。推荐方案 B。
3. **与 sequence packing 的交互**：Sequence packing 按 `input_lengths` 分箱（`batched_data_dict.py:428+`），不看 `token_mask`。我们的改动只把 `token_mask` 里的某些位置从 1 改到 0，不改变序列长度或 packing 结构；worker 层 `global_valid_toks` 从 `token_mask` 求和自然重算（`dtensor_policy_worker.py:565`）。**兼容性确认无问题**。
4. **Rollout 长度分布**：若 rollout 长度差异很大（比如 math 上从 100 到 2000），`prefix_ratio = 0.25` 对短序列是 25 token、对长序列是 500 token，语义不完全一致。可以考虑 `min(N_abs, prefix_ratio * L)` 的 hybrid，但先跑纯 ratio 版本。
5. **Apple 反向证据**（§2.5）：错误轨迹整体信号更好这一点如果确实成立、并且在 token-level 也成立（即错误轨迹的 tail 也是"更 aligned"的），那本 proposal 的前提就是错的。这是 core scientific risk，靠 E-C1 直接检验。

---

## 8. 相关工作（2026-07-09 调研快照）

**调研方法**：deep-research workflow，5 个搜索角度、22 个源、68 条 claim、25 条经 3-vote 对抗验证，20 条确认。所有 accuracy 数字均为原作者自报，缺乏独立复现。

### 8.1 基线：主流 OPD recipe

| 方案 | 序列长度处理 | 备注 |
|------|-------------|------|
| GKD (Agarwal 2023, arXiv:2306.13649) | full-length，token-uniform KL | reverse-KL mode-seeking |
| verl async OPD trainer | full-length，forward KL on teacher top-k | 仅 attention mask |
| **NeMo-RL `DistillationLossFn`** | full-length，token-uniform | 我们的基线 |

### 8.2 相关方向（都不是同一个 idea）

| 工作 | 时间 | 做的事 | 与本 proposal 的差别 |
|------|------|--------|---------------------|
| **REOPOLD** (arXiv:2603.11137) | 2026 | Entropy-based top-p% **逐 token** 动态采样 | 逐 token gating，不是前缀截断 |
| **AdaKD / LATF** | 2026 | Top-r% 难度 token 过滤（Hellinger 距离） | 同上，逐 token |
| **TIP** (Token Importance in OPD) | 2026 | 报告 10-50% token 子集匹敌 full-token | 支持"稀疏化 loss 可行"，但选的 token 未按位置 |
| **DOPD** (arXiv:2606.30626) | 2026 | 4-cell token routing（advantage gap + 双置信度） | 逐 token routing |
| **Apple "Unmasking OPD"** (alphaxiv 2605.10889) | 2026 | 错误轨迹整体更 aligned 的理论分析 | 见 §2.5，可能是核心 tension |
| **First-Token Distillation (FIRST)** | 2024 | 极端形式：只学首 token | Offline SFT distillation，非 OPD |
| **GRPO 里的长度 curriculum** | 常见 | 按 example 长度筛（短 → 长） | 筛例子，不是单条例子截断 |
| **Speculative decoding 训练** | 常见 | Teacher verifies student prefix | 用于 draft model 训练，不是 OPD |

**明确空白**：**"per-example 前缀截断 + 长度 warmup"在 LLM OPD 文献里未见直接对应工作**。这是本 proposal 的原创点。

### 8.3 领域现状 caveat

- **OPD 在数学推理上稳定性尚未解决**。仅过去 12 个月出现的 remediation 方法：EOPD, REOPOLD, HolderPO, stop-gradient TopK, TRB, TrOPD, DOPD, TIP, OPD+, PG-OPD, ESR, TRD。每篇都自称最优。
- **一个具体理论 bug**："Many Faces of OPD"（chatpaper f927484a）指出 TopK truncation 破坏了 `Σ π^S ∇log π^S = 0` 恒等式，导致 reverse-KL 有系统性梯度偏差；stop-gradient TopK 是修复方案。NeMo-RL 的 `zero_outside_topk` 路径**可能同样受影响**——本 proposal 的实验应先在 baseline 上验一次这个 bug 是否存在，避免 confound。
- **教师能力 U 形曲线**（Tsinghua 2026）：更强的教师并不总是更好；这与"错误轨迹尾部信号有噪声"的直觉方向一致。
- **独立复现空缺**：TRB/TrOPD/DOPD 的准确率数字都只有原作者一份。

---

## 9. 附录：关键文件锚点

- Loss 主实现：`nemo_rl/algorithms/loss_functions.py:966-1226`（`DistillationLossFn`）
- 掩码合成（改动落点）：`nemo_rl/algorithms/loss_functions.py:1211`
- 归一化：`nemo_rl/algorithms/loss_functions.py:1213-1217`（`masked_mean` + `global_valid_toks`）
- 训练循环 & teacher top-k 采集（需传 `global_step`）：`nemo_rl/algorithms/distillation.py:697-701`
- 配置 TypedDict：`nemo_rl/algorithms/loss_functions.py:966-969`
- YAML exemplar：`examples/configs/distillation_math.yaml`
- 已有 OPD 设计文档：`docs/design-docs/opd.md`
- Sequence packing 交互参考：`docs/design-docs/sequence-packing-and-dynamic-batching.md`
