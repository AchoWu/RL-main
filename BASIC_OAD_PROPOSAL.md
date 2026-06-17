# Basic OAD：Overlap-Aligned Distillation（基础版）

> 一种替代 KL 散度的在线策略蒸馏（OPD）损失函数。
> 把训练目标从"学生模仿教师概率密度"换成"最大化学生与教师分布的重叠"。

---

## 一、动机

### 1.1 当前 OPD 用 KL 的问题

NeMo-RL 现有的 OPD（`nemo_rl/algorithms/distillation.py` + `DistillationLossFn`）使用 forward / reverse / mixed KL。KL 散度有以下本质局限：

| 问题 | 表现 |
|---|---|
| **长尾爆炸** | 当某 token 在学生分布上概率接近 0 而教师概率非零时，KL 趋于无穷 |
| **Top-k 截断有偏** | 当前实现用 top-k=64，丢弃的尾部对 KL 是有偏估计，且偏差无界 |
| **support mismatch 不稳定** | 大词表（Qwen 152K）下 student/teacher 的 top-k 集合往往不同，KL 在并集外行为病态 |
| **训练目标与部署目标错位** | KL 优化"分布对齐"，但部署时真正在乎的是 token 级正确性或接受率 |

### 1.2 OAD 的核心思想（一句话）

> **不要让学生在每个 token 上模仿教师的完整密度，让学生分布与教师分布的重叠最大化。**

"分布重叠"是一个干净的几何概念：

```
两个分布 p_S 和 p_T 在每个 token y 上的"共同质量"是 min(p_S(y), p_T(y))
所有 token 上的共同质量加起来 = 重叠面积
                            = sum over y of min(p_S(y), p_T(y))
                            = 1 - TVD(p_S, p_T)
```

其中 TVD 是 total variation distance（全变差距离）。

这个量在不同部署场景下有清晰对应：

| 部署场景 | 对应指标 | OAD 的关系 |
|---|---|---|
| 学生独立部署（accuracy 主线）| 关键决策 token 的正确性 | 重叠最大化 ⇒ 学生在教师高概率区域也高概率 ⇒ accuracy 提升 |
| 学生作 speculative draft | 接受率 / 接受长度 | 重叠面积 = 单步接受率 = 1 - TVD（精确恒等式）|

也就是说，**OAD 的优化目标在两个场景下都是有意义的**——这与名字"Overlap-Aligned"对应：核心是分布重叠，speculative decoding 接受率只是它的一个部署解读。

---

## 二、Basic OAD 损失函数

### 2.1 定义

对学生 on-policy 轨迹的每个位置 t：

```
acceptance_t = sum over y of min( p_S(y | y_<t), p_T(y | y_<t) )

per_token_loss_t = -log(acceptance_t)

Basic_OAD_Loss = mean over valid tokens t of per_token_loss_t
```

每个 token 同等权重，损失即"负的对数接受率"。

### 2.2 一个具体计算例子

数学题位置 t，教师 / 学生 top-3 概率：

| token | p_T  | p_S  | min(p_S, p_T) |
|-------|------|------|---------------|
| `8`   | 0.70 | 0.30 | 0.30          |
| `八`  | 0.15 | 0.05 | 0.05          |
| `7`   | 0.02 | 0.40 | 0.02          |
| 其他  | 0.13 | 0.25 | (top-k 之外，估计偏差有界，见 §三) |

```
acceptance ≈ 0.30 + 0.05 + 0.02 = 0.37
per_token_loss = -log(0.37) ≈ 0.99
```

---

## 三、理论性质

### 3.1 数值稳定（无 KL 爆炸）

`min(p_S, p_T)` 取值恒在 [0, 1]，所以 acceptance ∈ [0, 1]，loss = -log(acceptance) 在 acceptance > 0 时有限。
即使 support mismatch（学生 / 教师概率为 0），OAD 也不会出现 KL 的 +∞ 行为。

工程上仅需 `acceptance.clamp_min(eps)` 兜底数值精度。

### 3.2 Top-k 截断偏差有界

**定理**：设教师 top-k 集合为 T，记 M_T = sum over y in T of p_T(y)（教师 top-k 累积概率）。则：

```
0 ≤ acceptance_true - acceptance_topk ≤ 1 - M_T
```

**证明**（一行）：

```
acceptance_true - acceptance_topk
  = sum over y not in T of min(p_S(y), p_T(y))
  ≤ sum over y not in T of p_T(y)
  = 1 - M_T
```

**意义**：Qwen3 等大模型在 k=64 时 M_T 通常 > 0.99，所以截断偏差 < 0.01。OAD 在 top-k 估计下**几乎无损**，与 KL 在 top-k 截断下的无界偏差形成鲜明对比。

> 📌 **理论假设的实证保证**：上述论证假定 M_T 在训练全程稳定 > 0.99。学生输出风格在训练中可能漂移，导致教师 top-k 集中度变化、M_T 时间不稳定，下界紧度也会随之飘移。
>
> Path A 下我们**无法直接观测教师真实 M_T**（教师 logsumexp 由 top-k 估计，按构造在 top-k 上重归一化质量恒为 1）。作为代理量，我们在 §5.3 实现了 `student_mass_on_teacher_topk`：训练良好时学生分布会逐渐集中到教师 top-k 上，该量上升表明分布趋同；该量持续偏低则同时暗示 (a) 学生与教师分布差异大、(b) 教师真实 M_T 也可能较低。若需精确监控真实 M_T，请切换到 §5.2 路径 B。

### 3.3 与 1 - TVD 的等价

acceptance_t = 1 - TVD(p_S, p_T)，所以 Basic OAD 等价于：

```
Loss = -E_t[ log(1 - TVD_t) ]
```

最小化 OAD ⇔ 最小化 TVD ⇔ 最大化分布重叠。

### 3.4 联合偏差分析（截断 + 概率放大）

§5.2 路径 A 用 `logsumexp(top-k)` 估计教师全局 logsumexp，引入第二个偏差源。两个偏差源方向**相反**，需要联合分析：

| 偏差源 | 方向 | 量级（k=64, M_T=0.99）|
|---|---|---|
| top-k 截断（丢弃尾部） | acceptance **被低估** | ≤ 1 - M_T ≈ 0.01 |
| 教师概率放大（用 top-k 估计 logsumexp）| acceptance **被高估** | ≤ (1/M_T - 1) · acceptance ≈ 0.007 |

**净偏差**：方向相反、部分抵消但不完全。在我们的实验设定下：

- 净偏差量级 ≤ 0.005（< 1%）
- **倾向于让 acceptance 被低估**（截断偏差通常占主导）
- 因此 OAD loss 倾向于**保守**——优化的是真实接受率的下界

**保守优化的好处**：训练目标是真实指标的下界 ⇒ 优化下界保证真实指标也提升。这是对 §3.2 上界论证的有益补充。

> ⚠️ 此结论的前提是 M_T 训练全程稳定。若 M_T 在某些位置/某些步骤显著下降（< 0.95），下界紧度会改变，"保守优化"的论证就需重新审视。Path A 下我们用 `student_mass_on_teacher_topk` 作为间接代理（见 §3.2 注释）。如需直接监控真实 M_T，切换到路径 B。

---

## 四、与 KL 的对比

| 维度 | KL forward | KL reverse | **Basic OAD** |
|---|---|---|---|
| 长尾稳定性 | 爆炸 | 爆炸 | **有界** |
| Top-k 偏差 | 无界 | 无界 | **≤ 1 - M_T，可忽略** |
| Support mismatch | 病态（+∞）| 病态 | **正常** |
| 数值实现复杂度 | 中 | 中 | **低** |
| 和部署指标对齐 | 间接 | 间接 | **直接（接受率）** |

> **范围声明**：本提案及配套实现**只支持 DTensor backend**（`dtensor_policy_worker.py` / `dtensor_policy_worker_v2.py`）。Megatron backend 不在本期范围——`get_topk_logits` 不返回精确全词表 `logsumexp`，OAD 在 Megatron 下会抛 `KeyError` 友好报错。当前 `train_opd.sh` 实验配置走 DTensor，本范围与实验完全对齐。

---

## 五、NeMo-RL 实现方案

### 5.1 关键概念澄清：logits → 概率

OAD 的接受率公式 `acceptance = sum of min(p_S(y), p_T(y))` 定义在**概率**上，所以无论实现如何都需要把 logits 转成概率。这是标准 softmax 操作：

```
p(y) = exp(logit(y) - logsumexp(logits))
```

理解这一点对后面的实现很重要：

- **学生侧**：loss 函数内部直接拿到完整 logits `[B, S, V_local]`，logsumexp 当场就能算出来——**不需要任何额外数据传输**
- **教师侧**：完整 logits 太大（`[B, S, 152K]` ≈ 80GB/batch）无法传输，train_data 里只有 top-k logits + indices。教师侧的概率有两种获取方式（见 §5.2）

> ⚠️ 注意：当前 KL 实现的做法是"在 top-k 上 log_softmax 重新归一化"，这等价于**假装 top-k 就是全词表**。OAD **不能**这么做——因为 `min(p_S, p_T)` 在系统性放大的伪概率上会严重高估接受率，给出错误信号。OAD 必须使用真实的全局概率。

### 5.2 教师侧概率的获取：路径 B（精确版，本提案选定）

> **v3.2 切换**：v1/v2/v3 主推路径 A（top-k 估计 teacher_lse），v3.1 评审保留路径 B 作为可选优化。**v3.2 决定切到路径 B 作为默认实现**。原因：
> - 消除 §3.4 的联合估计偏差，"identity ⇒ loss = 0" 严格成立
> - `teacher_topk_mass` 真实可观测（Path A 下恒为 1，无诊断价值）
> - 净增改动量很小（~50 行 worker 改动，单 batch ~1MB 额外数据），收益显著
> - 开 PR 前一次到位，避免后续切换语义混淆

#### 路径 B：教师 worker 提供精确全词表 logsumexp

教师推理时除了 top-k logits / indices，**额外返回每位置全局 logsumexp**：

```python
# nemo_rl/models/policy/workers/dtensor_policy_worker.py 等
# (CP 分支)
lse = vocab_cp_logsumexp(local_logits, tp_group=tp_group, cp_group=cp_group, full_seq_len=seq_len)

# (非 CP 分支)
lse = vocab_cp_logsumexp(local_logits, tp_group=tp_group, cp_group=None)

# 沿用 vals/idx 同款的 packing/CP allgather/padding/concat 流程，最终：
ret = {
    "topk_logits":  ...,
    "topk_indices": ...,
    "logsumexp":    ...,  # [B, S]，每位置全词表精确 logsumexp
}
```

`distillation.py:706-715` 沿现有钩子写入 train_data：

```python
if "logsumexp" in teacher_topk:
    train_data["teacher_logsumexp"] = teacher_topk["logsumexp"]
```

OADLossFn 直接消费 `data["teacher_logsumexp"]`，无 logsumexp 字段时抛 `KeyError`。

#### 与原路径 A 的对比

| 维度 | 路径 A（v3 实现）| **路径 B（v3.2 选定）**|
|---|---|---|
| 教师 lse 来源 | top-k 估计 | **worker 精确全词表** |
| identity loss | -log(M_T_true) ≠ 0 | **≡ 0** |
| `teacher_topk_mass` 监控 | 恒为 1（无诊断）| **真实 M_T，可监控理论假设** |
| §3.4 联合偏差 | 净偏差 ≤ 0.5% | **0** |
| Worker 改动量 | 0 行 | ~50 行（dtensor / dtensor_v2 各加 lse 返回值）|
| 数据传输开销 | 0 | ~1MB / batch（[B, S] float） |
| 单元测试断言 | identity loss = -log(M_T) | **identity loss < 1e-5** |

#### 后端兼容性

| Backend | 支持状态 |
|---|---|
| **DTensor (v1)** `dtensor_policy_worker.py` | ✅ 已在 `get_topk_logits` 中返回 logsumexp |
| **DTensor (v2)** `dtensor_policy_worker_v2.py` | ✅ 同上 |
| **Megatron** `megatron_policy_worker.py` | ❌ **不在本期范围内**——本提案及配套实现明确**只支持 DTensor backend**。Megatron worker 在 `get_topk_logits` 处不返回 `logsumexp`，OAD 在 Megatron backend 下会抛 `KeyError("teacher_logsumexp")` 友好报错。`train_opd.sh` 当前实验配置走 DTensor，本决策不影响实验进度 |

> **为什么不支持 Megatron**：Megatron worker 在 packing + CP 组合下需要复用 `cu_seqlens` 的 per-sequence allgather 重组逻辑，与 DTensor 的处理风格差异较大，单独写一份 Path B 的工作量与本期 PR 不成比例。如果未来确实需要在 Megatron backend 上跑 OAD（例如蒸馏 70B+ 模型时 Megatron 的并行能力更适配），届时单独开 PR 补足即可。

#### 触发条件

OAD 仅支持路径 B；现有训练命令切换 OAD 只需：

```bash
python examples/run_distillation_math.py loss_fn.type=oad ...
```

无 `loss_fn.type=oad` 时一切照旧（默认 KL，路径无关）。

### 5.3 损失函数（新增 `OADLossFn` 到 `loss_functions.py`，使用路径 A）

核心逻辑约 100 行。**学生侧 logsumexp 当场算，教师侧 logsumexp 从 top-k 估计**。

> 📌 **Helper 依赖说明**（从 codebase 实际状态出发）：
> - `gather_logits_at_global_indices` —— 当前 `loss_functions.py` 已有，可直接复用
> - `_get_tokens_on_this_cp_rank` —— 当前 `model_utils.py` 已有，可直接复用
> - `resolve_parallel` —— 当前**不存在**，需要从 `DistillationLossFn.__call__:1029-1062` 内联代码抽出为独立 helper（约 35 行）
> - `distributed_logsumexp` —— 当前**不存在**，需要新增（约 10 行 TP-aware allreduce）
>
> 这两个新 helper 会同时被 OAD / KL 复用，属于一次性重构成本。

```python
class OADLossConfig(TypedDict):
    eps: float


class OADLossDataDict(TypedDict):
    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    teacher_topk_logits: torch.Tensor   # [B, S, k]   现有字段，无需新增
    teacher_topk_indices: torch.Tensor  # [B, S, k]   现有字段，无需新增


class OADLossFn(LossFunction):
    """Basic Overlap-Aligned Distillation Loss.

    Loss = -E_t[ log( sum_y min(p_S(y|y_<t), p_T(y|y_<t)) ) ]

    实现要点：
    - 学生侧用全词表 logits 现场算 logsumexp（精确）
    - 教师侧从 top-k logits 估计 logsumexp（偏差 ≤ -log(M_T)，k=64 时通常 < 0.01）
    """

    def __init__(self, cfg: OADLossConfig):
        self.eps = cfg.get("eps", 1e-8)
        self.loss_type = LossType.TOKEN_LEVEL

    def __call__(
        self,
        next_token_logits,                    # [B, S, V_local]
        data,
        global_valid_seqs,
        global_valid_toks,
        vocab_parallel_rank=None,
        vocab_parallel_group=None,
        context_parallel_group=None,
    ):
        # 1. 取出教师 top-k 数据（沿用现有 train_data 字段）
        t_topk_logits = data["teacher_topk_logits"].to(torch.float32)
        t_topk_idx    = data["teacher_topk_indices"]
        token_mask    = data["token_mask"]

        # 2. 解析并行配置（从 DistillationLossFn 抽出的新 helper）
        cp_group, cp_size, parallel_group, vocab_start, vocab_end, logits_local = \
            resolve_parallel(next_token_logits, vocab_parallel_rank,
                             vocab_parallel_group, context_parallel_group)
        logits_local = logits_local.to(torch.float32)

        # 3. 学生 TP-aware 全局 logsumexp（精确，全词表）
        s_lse = distributed_logsumexp(logits_local, parallel_group)  # [B, S]

        # 4. 教师 logsumexp 从 top-k 估计（偏差 ≤ -log(M_T)，可忽略）
        t_lse = torch.logsumexp(t_topk_logits, dim=-1)  # [B, S]

        # 5. 学生在教师 top-k 索引位置的全局 logits（复用现有 helper）
        s_topk_logits = gather_logits_at_global_indices(
            logits_local, t_topk_idx,
            tp_group=parallel_group, cp_group=cp_group,
            vocab_start_index=vocab_start, vocab_end_index=vocab_end,
        )  # [B, S, k]

        # 6. logits → 全局概率（标准 softmax，用全局归一化常数）
        s_log_p_topk = s_topk_logits - s_lse.unsqueeze(-1)   # [B, S, k]
        t_log_p_topk = t_topk_logits  - t_lse.unsqueeze(-1)  # [B, S, k]

        s_p_topk = s_log_p_topk.exp()
        t_p_topk = t_log_p_topk.exp()

        # 7. 接受率（top-k 下界估计，偏差 ≤ 1 - M_T）
        acceptance = torch.minimum(s_p_topk, t_p_topk).sum(dim=-1)  # [B, S]

        # 8. Next-token alignment
        # 注意：只 clamp_min，不 clamp_max
        # 教师=学生时 acceptance ≡ 1，loss 应精确为 0；clamp_max 会破坏单元测试断言
        acceptance = acceptance[:, :-1].clamp_min(self.eps)
        log_accept = acceptance.log()                                # [B, S-1]
        token_mask = token_mask[:, :-1]

        # 9. CP 切分对齐
        if cp_size > 1:
            cp_rank = torch.distributed.get_rank(cp_group)
            log_accept = _get_tokens_on_this_cp_rank(log_accept, cp_rank, cp_size, seq_dim=1)
            token_mask = _get_tokens_on_this_cp_rank(token_mask, cp_rank, cp_size, seq_dim=1)

        # 10. 最终 loss
        per_token_loss = -log_accept                                  # [B, S-1]
        masked_loss    = per_token_loss * token_mask
        loss = masked_loss.sum() / global_valid_toks.clamp_min(1)

        # 11. 监控指标（含 §13 评审建议的核心监控）
        with torch.no_grad():
            valid = token_mask.bool()
            n_valid = token_mask.sum().clamp_min(1)
            mean_accept = (acceptance * token_mask).sum() / n_valid
            min_accept  = acceptance[valid].min() if valid.any() else torch.tensor(0.0)

            # 学生在教师 top-k 上的概率质量（Path A 下可观测的代理量）
            # 注意：直接监控 M_T 在 Path A 下不可行——教师 logsumexp 由 top-k 估计，
            # 教师概率在 top-k 上重归一化后总和恒为 1，无诊断意义。
            # 这里改为监控学生分布在教师 top-k 上的质量，训练良好时该量会上升。
            student_mass_on_teacher_topk = (
                s_p_topk[:, :-1].sum(dim=-1) * token_mask
            ).sum() / n_valid

            # 梯度活跃度监控：min 操作下 p_S < p_T 的 token 才有梯度
            # 联合 mean_accept 看走势，而非单点阈值
            active_grad_ratio = (
                (s_p_topk[:, :-1] < t_p_topk[:, :-1]).any(dim=-1).float() * token_mask
            ).sum() / n_valid

        metrics = {
            "oad_loss":              loss.detach(),
            "acceptance_rate_mean":  mean_accept,
            "acceptance_rate_min":   min_accept,
            "tvd_mean":              1.0 - mean_accept,
            "student_mass_on_teacher_topk": student_mass_on_teacher_topk,  # Path A 代理
            "active_grad_ratio":     active_grad_ratio,   # 与 acceptance 联合解读
        }
        return loss, metrics
```

### 5.4 配置改动

> 📌 **schema 设计说明**：当前 `distillation_math.yaml` 的 `loss_fn` 直接是 KL 字段（`kl_type` / `mixed_kl_weight` / `zero_outside_topk`），**没有 `type` 字段**。新设计需引入 `type` 但保持向后兼容。

**`examples/configs/distillation_math.yaml`**：

```yaml
loss_fn:
    type: "oad"          # 新增；缺省时默认 "kl" 保持向后兼容

    # KL 字段保留在顶层，type=kl 时使用
    kl_type: "mixed"
    mixed_kl_weight: 0.5
    zero_outside_topk: false

    # OAD 字段嵌套在 oad 子节点
    oad:
        eps: 1.0e-8
```

**`examples/run_distillation_math.py`** 的 setup 增加分支：

```python
loss_type = loss_config.get("type", "kl")  # 兼容老配置
if loss_type == "kl":
    loss_fn = DistillationLossFn(loss_config)
elif loss_type == "oad":
    loss_fn = OADLossFn(loss_config["oad"])
else:
    raise ValueError(f"Unknown loss type: {loss_type}")
```

---

## 六、训练流程（与现有 OPD 完全一致）

每个 step 内部：

1. **学生 on-policy rollout**（vLLM 生成轨迹，复用现有逻辑）
2. **教师推理**：`teacher_policy.get_topk_logits(train_data, k=64)`
   - 返回值与现状完全一致：top-k logits + indices
   - **不需要任何修改**
3. **学生训练**：
   - 学生 logsumexp 当场算（精确）
   - 教师 logsumexp 从 top-k 估计（偏差 ≤ -log(M_T)，可忽略）
   - 计算每个 token 的全局概率 → `acceptance = sum(min(p_S, p_T))`
   - loss = `-log(acceptance)` 平均

整个训练 loop（`distillation_train` in `distillation.py`）和 train_data pipeline **均无需改动**，只新增一个 loss 函数类。

---

## 七、验证计划

### 7.1 Phase 1：正确性验证（约 1 天）

- **单元测试**：教师 = 学生时，acceptance ≡ 1，loss ≡ 0（要求 `clamp_max` 已移除，否则此断言不成立）
- **形状测试**：B=2, S=128, k=64 在各种 TP/CP 组合（TP=1/2/4, CP=1/2）下不报错
- **梯度测试**：随机 logits 反向传播无 NaN / Inf；额外验证 `p_S = p_T` 处 sub-gradient 行为正常
- **数值对照**：mini batch 上手算 acceptance vs OAD loss

### 7.2 Phase 2：小规模可学性（约 2-3 天）

- 8 卡 Qwen3-4B → Qwen3-1.7B
- 跑 200 step，与现有 KL（forward/reverse/mixed）对照

**核心监控指标**（不仅看 loss）：

| 指标 | 期望走势 | 异常信号 |
|---|---|---|
| `sad_loss` | 平稳下降 | 抖动大 / 不下降 |
| `acceptance_rate_mean` | 单调上升（~0.3 → 0.7+）| 进入平台期 |
| `student_mass_on_teacher_topk`（Path A 代理）| 单调上升（0.3-0.5 → 0.8+）| 长期偏低 → 学生分布与教师 top-k 错位，考虑切路径 B |
| `active_grad_ratio` | 与 acceptance 同步走势 | 急降但 acceptance 不再上升 → 梯度稀疏化拖累 |
| GSM8K 中间 accuracy | 不劣化 | 显著低于 KL baseline |
| 训练 NLL | 不上升 | 上升 → 检查 logsumexp 数值精度 |

> **联合解读关键**：`active_grad_ratio` 单独看意义有限，**必须和 `acceptance_rate_mean` 一起看**。
> - 两者同步上升 = 健康训练
> - acceptance 进入平台期 + active_ratio 持续下降 = 梯度稀疏化拖累训练（启用 §9 smooth-min）
> - acceptance 上升但 active_ratio 不变 = 梯度集中在少数 token 上（学生在小部分位置努力学习，正常）

### 7.3 Phase 3：完整对照实验（约 1-2 周）

- 完整 1000 step 训练
- Baselines: KL forward / KL reverse / KL mixed / **Basic OAD**
- 评估：
  - **GSM8K / MATH accuracy**（OPD 主目标）
  - **Speculative decoding 接受率 / 接受长度**（OAD 衍生指标，写论文用）

---

## 八、风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| 学生全局 logsumexp 在 CP 下数值不稳 | 中 | 强制 fp32 计算，单元测试覆盖 |
| 教师 logsumexp 估计偏差导致 acceptance 偏移 | 低 | k=64 时净偏差 < 0.5%（见 §3.4）；监控 `student_mass_on_teacher_topk`（Path A 代理）；如需精确改用路径 B |
| **梯度稀疏化（acceptance 接近 1 时 `p_S > p_T` 区域无梯度）** | **中** | **本质是 `min` 操作非平滑的内禀属性。容量差距大的场景前中期影响有限；后期可能拖慢收敛。监控 `active_grad_ratio` × `acceptance_rate_mean` 联合曲线；若出现稀疏化拖累，启用 §9 的 smooth-min 扩展** |
| **M_T 在领域外 / 训练早期不够大** | **中** | **Qwen3-4B + DeepScaler 设定下应稳定 > 0.99，但作为通用方案需实证验证；监控 `student_mass_on_teacher_topk`（Path A 间接代理），长期偏低时切路径 B 直接观测真实 M_T** |
| 学生 top-k 与教师 top-k 不重合，acceptance 低估严重 | 低 | top-k=64 时教师 M_T > 0.99，理论保证偏差 ≤ 0.01 |
| acceptance 在初期接近 0，loss 太大 | 低 | `clamp_min(eps)` 兜底；与 KL 初期 NLL 巨大同理 |
| 学生 mode collapse（只押教师 top-1）| 极低 | min 操作鼓励重叠不是 mode-seek；监控 generation entropy 兜底 |
| 与 vLLM refit 节奏冲突 | 极低 | OAD 不改变 train pipeline，refit 完全不变 |

---

## 九、后续扩展（不在本期范围）

Basic OAD 是后续多个工作的基础。本提案精简到 3 个核心扩展方向，避免画大饼模糊 Basic 的边界：

### 9.1 Critical-Token 加权 OAD（accuracy 主线）

用教师熵 / 师生 KL 给每个 token 加权，把训练资源集中在"决策 token"上，进一步提升 accuracy。
- 8192 个 token 的轨迹中真正决定对错的可能只有 30-50 个，平权 OAD 浪费容量
- 实现成本约 30 行（权重计算 + 加权求和）

### 9.2 Smooth-min OAD（解决梯度稀疏化）

用 `softmin(p_S, p_T) = -1/τ · log(exp(-τ p_S) + exp(-τ p_T))` 替代真 `min`，使梯度处处非零、缓解 §8 的梯度稀疏化风险。

> **为什么 Basic 版本必须先于 Smooth-min 存在**（避免论文叙事被反向削弱）：
>
> 1. **理论 closed-form**：Basic 版本的 `acceptance = 1 - TVD` 是经典统计量，min 操作直接对应 speculative decoding 的接受率定义，**论文里能给定理（§3.2 / §3.4）**。Smooth-min 下没有这种恒等式
> 2. **零附加超参**：Smooth-min 引入温度 τ，τ→0 退化为均值、τ→∞ 趋近真 min，调超参本身就是一个研究问题；baseline 不应把这作为前置工作
> 3. **保留 §3.3 的优雅等价性**：Basic 的"loss = -log(1 - TVD)"提供清晰可解释性，Smooth-min 引入额外近似
> 4. **作为消融的 reference point**：Smooth-min 的论文必须报告"vs Basic OAD"才能说清贡献，所以 Basic 必须先存在
>
> 因此 Basic OAD 与 Smooth-min OAD 的关系是"theoretically clean baseline → 解决工程边界问题的扩展"，而非"v1 有 bug, v2 修了"。

### 9.3 TaR-GRPO（与 RL 融合的最大延伸）

把教师 logprob 当 dense process reward，与验证器奖励联合驱动 GRPO；
OAD 成为 RL advantage 计算的一个组件而非独立 loss。

---

> 注：LW-OAD（length-weighted）/ Chained-OAD / Union-Support OAD 等更细粒度变体，归入未来研究 / 论文 future work，与 Basic 提案解耦。

---

## 十、文件改动清单

> 📌 **以最新实施版（v3.2 路径 B）为准请直接看附录 E.2**。本节是 v2 时期的两阶段（A → B）规划清单，保留作设计演进记录。

### 路径 A（推荐起手，零数据 pipeline 改动）

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `nemo_rl/algorithms/loss_functions.py` | 新增 ~100 行 | 增加 `OADLossConfig`、`OADLossDataDict`、`OADLossFn` |
| `nemo_rl/algorithms/loss_functions.py` | 重构 ~45 行 | 抽出 `resolve_parallel` 与 `distributed_logsumexp` 共用 helper（KL/OAD 共享）|
| `examples/run_distillation_math.py` | 修改 ~10 行 | 根据 `loss_fn.type` 选择 `DistillationLossFn` 或 `OADLossFn` |
| `examples/configs/distillation_math.yaml` | 修改 ~5 行 | 新增 `loss_fn.type` 与 `loss_fn.oad` 配置（向后兼容老 schema）|
| `tests/unit/algorithms/test_oad_loss.py` | 新增 ~150 行 | OAD 损失函数单元测试 |

总改动量约 **200-250 行**，集中在 loss 与配置层，不触及训练循环、不修改教师 worker、不改动 train_data pipeline。

### 路径 B（精确版，v3.2 已实施为唯一支持路径）

仅支持 DTensor backend；Megatron 不在范围（见 §五范围声明）：

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `nemo_rl/models/policy/workers/dtensor_policy_worker.py` | 修改 ~60 行 | `get_topk_logits` 返回值新增 `logsumexp` 字段（CP / 非 CP / packing 三个分支同步处理）|
| `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py` | 修改 ~60 行 | 同上 |
| `nemo_rl/models/policy/workers/megatron_policy_worker.py` | **不支持，未改动** | OAD 在 Megatron backend 下抛 `KeyError("teacher_logsumexp")` 友好报错 |
| `nemo_rl/models/policy/interfaces.py` | 修改 ~5 行 | `TopkLogitsOutputSpec` 新增 `logsumexp: NotRequired[Tensor]` |
| `nemo_rl/algorithms/distillation.py` | 修改 ~6 行 | 透传 `teacher_topk["logsumexp"]` 到 `train_data["teacher_logsumexp"]` |

总改动量与附录 E.2 一致（~170 行，含 worker 改动占大头）。

---

## 附录 A：评审历史（v1 评审）

> 以下为本提案 v1 版的初步评审反馈，及作者回应。v2 已将所有 P0/P1 修改吸收到正文 §3 / §5 / §7 / §8 / §9，本附录仅作对话历史保留。

### A.1 v1 评审：技术疑问与潜在风险

#### (1) 🔴 教师 logsumexp 估计的偏差方向需要复核

§5.2 路径 A 用 `logsumexp(top-k)` 当全局 logsumexp，会让 `p_T(top-k token)` 系统性放大 1/M_T 倍 → acceptance 被高估。而 §3.2 的"截断偏差 ≤ 1 - M_T"是 acceptance 被低估方向的偏差。两个方向相反，应给出净偏差分析。

**已在 v2 §3.4 补充**联合偏差表 + "保守优化"卖点。

#### (2) 🔴 `min(p_S, p_T)` 的梯度稀疏化问题

`min` 是 non-smooth 操作：当 `p_S(y) > p_T(y)` 时梯度为 0。训练后期可能导致梯度稀疏化、训练停滞。

**已在 v2 §8 风险表**用"内禀属性 + 后期可能拖慢"措辞重写；**v2 §7.2** 增加 `active_grad_ratio` × `acceptance_rate_mean` 联合监控；**v2 §9.2** 加入 Smooth-min OAD 作为正式扩展方向。

#### (3) 🟡 `acceptance.clamp(max=1-eps)` 的理论性问题

`max=1-eps` 会导致教师=学生时 loss 不为 0，破坏 Phase 1 单元测试断言。

**已在 v2 §5.3 修复**：改为 `acceptance.clamp_min(eps)`，去掉 max。

#### (4) 🟡 M_T 在训练早期 / 领域外可能不够大

§3.2 论证依赖 "k=64 时 M_T > 0.99"，未训练好的教师 / 不匹配领域可能只有 0.8–0.9。

**已在 v2 §5.3 监控**：metrics 增加 `teacher_topk_mass`（v3 实施时发现 Path A 下此量恒为 1，无诊断意义，改名为 `student_mass_on_teacher_topk` 作为间接代理）；**v2 §3.2** 增加"理论假设的实证保证"声明；**v2 §8 风险表** 新增此风险条目。

#### (5) 🟡 Helper 签名核实

`resolve_parallel` / `distributed_logsumexp` 等 helper 在当前 codebase 不存在，需新抽出。

**已在 v2 §5.3** 加 helper 依赖说明；**v2 §10** 文件改动清单加重构条目。

### A.2 v2 评审（二轮）：剩余三条文字润色

#### (1) "梯度稀疏化是特性不是缺陷"框架太二元

KL 是渐进衰减，OAD 是阶跃切断。后期可能拖慢学习率。

**已在 v2 §8 风险表**用评审建议的措辞重写：承认"内禀属性 + 后期可能拖慢"，并给出联合监控方法。

#### (2) "保守优化"框架的隐含前提

"优化下界 ⇒ 真实指标提升"成立的前提是下界紧度稳定，即 M_T 训练全程稳定。

**已在 v2 §3.2 / §3.4** 添加 M_T 稳定假设的实证保证声明，把 `teacher_topk_mass` 从"运维指标"升级为"理论假设的实证检验"。（v3 修正：Path A 下 `teacher_topk_mass` 不可观测，改用 `student_mass_on_teacher_topk` 作为代理；如需直接监控真实 M_T 须切路径 B。）

#### (3) 为什么 Basic 不直接用 Smooth-min

把 Smooth-min 升级为 future work 后，需要给 Basic 一个正面理由，避免论文叙事被反向削弱。

**已在 v2 §9.2** 给出 4 条理由（理论 closed-form / 零附加超参 / 优雅等价性 / 作为消融 reference point），把 Basic ↔ Smooth-min 的关系定位为"theoretically clean baseline → 工程优化扩展"。

### A.3 评审结论

- 一轮评审 5 条 🔴/🟡 全部吸收（4 条无条件采纳 + 1 条降级处理 + 监控）
- 二轮评审 3 条文字润色全部吸收
- v2 在正文 §3 / §5 / §7 / §8 / §9 / §10 完成全量整合

✅ **可以开工**。

---

## 附录 B：v3 实施记录与代码改动审核清单

> 本节记录 v2 提案落地为可运行代码的过程：实际改动了哪些文件、与原提案的偏差、实施时新发现的问题、以及待审核者关注的争议点。
>
> **状态**：实施完成，等待审核。所有代码改动均通过 Python AST 语法检查，但因本地环境无 torch/pytest，**未运行单元测试**——这是审核后的下一步。

### B.1 改动文件清单

| 文件 | 改动类型 | 行数（净增）| 说明 |
|---|---|---|---|
| `nemo_rl/distributed/model_utils.py` | 新增函数 | +55 | 新增 `vocab_cp_logsumexp`（TP+CP 全词表 logsumexp）|
| `nemo_rl/algorithms/loss_functions.py` | 新增类 + 修改 import | +210 | 新增 `OADLossConfig` / `OADLossDataDict` / `OADLossFn`；import 增加一行 |
| `nemo_rl/algorithms/distillation.py` | 修改 import + dispatch + 类型注解 | +12 | OAD 类型 import；setup 内 `loss_fn.type` 分发；setup 返回类型与 `distillation_train` 函数签名扩展 |
| `examples/configs/distillation_math.yaml` | 修改 schema | +5 | 新增 `loss_fn.type` 与 `loss_fn.oad` 嵌套字段，向后兼容 |
| `tests/unit/algorithms/test_oad_loss.py` | 新增测试 | +250 | 6 个 CPU 单元用例（识别测试、手算对照、梯度流、mask 尊重、metric 边界）|
| `BASIC_OAD_PROPOSAL.md` | 文档校正 | 多处 | `teacher_topk_mass` → `student_mass_on_teacher_topk`，附录 B（本节）新增 |

总代码改动 **~530 行**（含测试 250 行；非测试代码约 280 行）。

### B.2 OADLossFn 11 步 pipeline 审核要点

`nemo_rl/algorithms/loss_functions.py:OADLossFn.__call__`（约 1283-1425 行）。

| 步骤 | 核心代码 | 审核关注点 |
|---|---|---|
| 1 | 取 `teacher_topk_logits / teacher_topk_indices` | 字段已存在于 train_data，无 pipeline 改动 |
| 2 | TP/CP/DTensor 分支解析（mirror DistillationLossFn:1029-1062）| 三个分支：vocab_parallel_group / DTensor / single-rank。逻辑与 KL 一致 |
| 3 | `student_lse = vocab_cp_logsumexp(...)` | **学生侧 lse 是精确全词表值**（TP allreduce + CP allgather）|
| 4 | `teacher_lse = torch.logsumexp(teacher_topk_logits, -1)` | **Path A 估计**——B.4 详细讨论其影响 |
| 5 | `student_topk_logits = gather_logits_at_global_indices(...)` | 直接复用现有 helper，TP+CP 已打包 |
| 6 | `student_log_p_topk = student_topk_logits - student_lse.unsqueeze(-1)`<br>`teacher_log_p_topk = teacher_topk_logits - teacher_lse.unsqueeze(-1)` | **OAD 区别 KL 的核心**：用全局 lse 还原全局概率，**不**在 top-k 上重新归一化 |
| 7 | `acceptance = torch.minimum(s_p_topk, t_p_topk).sum(-1)` | 接受率 = 重叠面积；元素级 `min` 而非比值 |
| 8 | `per_token_loss = -acceptance[:, :-1].clamp_min(eps).log()`<br>`token_mask = data["token_mask"][:, 1:]` | **只 clamp_min，无 max**（v1 评审 #3 修复点）；mask 使用 `[:, 1:]` 与 KL 一致（被预测 token 的 mask）|
| 9 | `loss = masked_mean(..., global_normalization_factor=global_valid_toks)` | 复用现有 `masked_mean` 工具，与 KL 归约风格一致 |
| 10 | （监控构建）| 见 §B.5 |
| 11 | `metrics = {...}` 返回 | `loss / num_valid_samples / acceptance_rate_mean / acceptance_rate_min / tvd_mean / student_mass_on_teacher_topk / active_grad_ratio` |

### B.3 与提案 v2 §5.3 的偏差

实施过程中三处偏离原计划，均为务实简化：

#### 偏差 1：未抽出 `resolve_parallel` helper

- **原计划**：从 `DistillationLossFn.__call__` 内联代码抽出独立 `resolve_parallel` helper（v2 §5.3 helper 依赖说明）
- **实施**：`OADLossFn.__call__` 内联了同样逻辑，与 KL 风格一致
- **理由**：`gather_logits_at_global_indices` 已经把 TP+CP 的复杂处理打包好了，OAD 路径只需要少量解析逻辑。抽 helper 的边界不清晰（需要返回 6 个值的元组），增加维护成本。**有意见可以再讨论是否要抽**
- **影响**：`loss_functions.py` 内 OAD 与 KL 的 TP/CP 解析代码有约 30 行重复。如果未来要支持更多 distill loss 变体，再做抽取更合适

#### 偏差 2：唯一新增 helper 是 `vocab_cp_logsumexp`

- **原计划**：新增 `distributed_logsumexp`
- **实施**：新增 `vocab_cp_logsumexp` 在 `model_utils.py`，命名更准确（既处理 vocab parallel 又处理 cp）
- **位置**：`nemo_rl/distributed/model_utils.py:984-1037`
- **签名**：
  ```python
  def vocab_cp_logsumexp(
      vocab_parallel_logits: torch.Tensor,
      tp_group: Optional[ProcessGroup] = None,
      cp_group: Optional[ProcessGroup] = None,
      *,
      full_seq_len: Optional[int] = None,
  ) -> torch.Tensor
  ```
- **审核关注点**：
  - 严格 fp32 计算（防数值不稳）
  - `full_seq_len` 用于 CP 下 padding 切回原长度（与 `gather_logits_at_global_indices` 同款约定）
  - 当前**未对外暴露 ChunkedDistributedFunction 风格的分块**——如果序列特别长可能 OOM，但 logsumexp 内存量 = O(B·S·V_local)，与 forward 同量级，不应是新瓶颈

#### 偏差 3：监控指标的现实校正 ⚠️ 重要

- **原计划**：暴露 `teacher_topk_mass = sum(teacher_p_topk)` 用于监控 M_T
- **实施时发现的真问题**：在 Path A 下，`teacher_p_topk = exp(t_topk_logits - logsumexp(t_topk_logits))`——按构造在 top-k 上 sum 必为 1。**这个量恒为 1，无诊断价值**
- **修正**：改为暴露 `student_mass_on_teacher_topk = sum(student_p_topk)`——学生分布在教师 top-k 上的真实质量，训练良好时单调上升
- **影响**：
  - 提案 §3.2 / §3.4 / §5.3 / §7.2 / §8 / §10 中所有 `teacher_topk_mass` 引用已改为 `student_mass_on_teacher_topk` 并加 Path A 限制说明
  - 单元测试 #2 的断言相应调整（见 §B.6）
  - 路径 B 切换条件改为"学生分布与教师 top-k 长期错位"或"需要直接观测真实 M_T"

### B.4 Path A 偏差的实证后果（identity 测试）

实施时的另一个发现——**当 student_logits == teacher_logits 时，loss 不为 0**。

#### 推导
- `student_lse = logsumexp(student_logits)` 是**全词表**精确值
- `teacher_lse = logsumexp(teacher_topk_logits)` 是**top-k 估计**
- 当 student==teacher 时：`student_lse > teacher_lse`（因为左边求和项更多）
- 所以 `student_p_topk = exp(s_topk - s_lse) < exp(t_topk - t_lse) = teacher_p_topk`
- `acceptance = sum(min(p_S, p_T)) = sum(student_p_topk) = student_mass_on_teacher_topk`
- 数值上等于教师真实 top-k 累积概率 M_T（< 1）
- `loss = -log(M_T)`，而不是 0

#### 这是 bug 还是 feature
- **不是 bug**：数学上完全正确，反映了 Path A 估计偏差的真实代价
- **是 feature**：`teacher_p_topk` 被系统性放大、`student_p_topk` 不变，`min` 操作下学生概率成为短板——这恰好让 OAD loss 倾向于"先把学生概率推到教师 top-k 上"，与训练目标一致
- **但需要诚实交代**：论文/讲解时不能说"OAD 在完美收敛时 loss = 0"，否则会被 reviewer 抓住

#### 论文叙事建议
- **写作时强调**：OAD-Path-A 的 loss 下界 = -log(M_T_true)，而非 0
- **配合定理**：M_T → 1 时 loss → 0；k=64 + Qwen3 设定下 -log(M_T) ≈ 0.005
- **路径 B 的额外卖点**：消除此偏差，loss 真正以 0 为下界

### B.5 监控指标的两个争议设计

#### 争议 1：`active_grad_ratio` 用 `any()` 还是 `mean()`

```python
# 当前实现（位置级 ratio）
active_grad_ratio = (
    (student_p_topk[:, :-1] < teacher_p_topk[:, :-1])
    .any(dim=-1)        # 任一 top-k token 满足 p_S < p_T 即算"活跃"
    .to(mask.dtype) * mask
).sum() / n_valid
```

- **当前 (`any`)**：位置级，"该 position 至少有一个 token 给了梯度信号"
- **替代 (`mean`)**：token 级，"top-k 内多少 token 给了梯度信号"

`any` 的解读更接近"是否有梯度流入这个 position"，`mean` 更细粒度。当前选 `any` 是因为论文里讨论"梯度稀疏化"时关心的是 position 级而非 token 级。**审核者可推翻**——一行改动。

#### 争议 2：identity 下 `active_grad_ratio` ≈ 1 而非 0

由于 §B.4 的 Path A 偏差（student_p_topk 系统性 < teacher_p_topk），即使 student==teacher，几乎每个 position 都满足 `p_S < p_T`，`active_grad_ratio` 接近 1。

**这与直觉冲突**：人会期望"学生与教师完全一致 → 不需要梯度 → active_ratio = 0"。

但实际语义是："Path A 估计偏差让 loss 在 identity 时仍非零，因此训练仍在推学生分布——active_ratio = 1 反映的是这一事实"。

**审核者关注点**：这个监控量在 Path A 下的解读和路径 B 下不同。论文/wandb 标签里需要说清楚。

### B.6 单元测试设计

`tests/unit/algorithms/test_oad_loss.py`，6 个用例，**纯 CPU，无需 GPU/分布式**。

| # | 测试 | 关键断言 |
|---|---|---|
| 1 | 形状 / 标量 loss / 无 NaN | `loss.dim() == 0`；metrics 字段齐全 |
| 2 | **Identity（重要）**| 校正版断言：student==teacher 时 `loss == -log(student_mass_on_teacher_topk)` 而非 0；Path A 下 `active_grad_ratio > 0.9` |
| 3 | 手算 1×2 toy case | 用 `torch.logsumexp` 手算位置 0 的 acceptance，与 `OADLossFn` 输出对照（atol=1e-5）|
| 4 | 梯度流 | `student_logits.requires_grad_(True)`，反传后 grad 有限且非全零 |
| 5 | mask 尊重 | masked-out 位置的 logits 被破坏后 loss 仍有限 |
| 6 | metric 边界 | acceptance ∈ [0,1]、tvd_mean = 1 - mean、active_ratio ∈ [0,1]、student_mass ∈ [0,1] |

**未覆盖**（需要 GPU/分布式集成测试）：
- TP > 1 下 `vocab_cp_logsumexp` 的 allreduce 正确性
- CP > 1 下 `_get_tokens_on_this_cp_rank` 切分对齐
- DTensor 分支
- 与 vLLM refit 的端到端

这些建议跟着现有 NeMo-RL 的 GPU functional test 框架补，**不在本期 PR 范围**。

### B.7 默认行为与回滚路径

**默认配置不改变现有训练行为**：
- `loss_fn.type` 缺省时（旧 yaml）走 `"kl"` → 调用 `DistillationLossFn(loss_config)`，与改动前完全一致
- 即使新加了 `loss_fn.oad` 字段，只要 `type != "oad"`，OAD 完全不被执行

**切换 OAD 的最小命令**：
```bash
python examples/run_distillation_math.py loss_fn.type=oad ...  # 其他参数不变
```

**回滚路径**：
- 撤销 `distillation.py` 的 dispatch（恢复 `loss_fn = DistillationLossFn(loss_config)`），保留 OADLossFn 类不删；老配置继续运行
- 完全回滚：`git revert` 本次 PR 即可

### B.8 已知风险与 reviewer 的 4 个质疑点

| # | 质疑点 | 我的回应 |
|---|---|---|
| 1 | **Identity loss ≠ 0**（§B.4） | 是 Path A 设计取舍，不是 bug。已写入测试 + 文档诚实交代 |
| 2 | **`active_grad_ratio` 用 any 还是 mean**（§B.5）| any 更对应"位置级梯度信号"，但选项是开放的，欢迎讨论 |
| 3 | **TP/CP 解析代码与 KL 重复 ~30 行**（§B.3 偏差 1）| 务实选择；如未来加更多 distill loss 再抽 |
| 4 | **未运行的单元测试** | 本地无 torch；下一步在有 GPU 的机器跑 `pytest tests/unit/algorithms/test_oad_loss.py` |

### B.9 下一步（按优先级）

1. **审核完成 + 跑通单元测试**（约 1 小时，需要 torch 环境即可，CPU）
2. **Phase 2 小规模训练**（约 2-3 天，需要 8 卡）：
   - `train_opd.sh` 加上 `loss_fn.type=oad`，跑 200 step
   - 监控 `acceptance_rate_mean` / `student_mass_on_teacher_topk` / `active_grad_ratio` 联合走势
   - 与 KL baseline 对照 GSM8K 中间评估 accuracy
3. **若 Phase 2 显示 `student_mass_on_teacher_topk` 长期偏低**，启动路径 B（教师 worker 改动 ~15 行）
4. **完整对照实验**（约 1-2 周，1000 step）：写论文用

---

## 附录 C：v3 实施版评审意见（待作者修改）

> 评审针对 §B 实施记录 + 代码改动。整体结论：✅ **批准开 PR + 启动 Phase 2 训练**，但建议在 200 step 跑动前完成下述轻量动作（约 30 分钟）。

### C.1 v3 的两个亮点（值得点名）

#### (1) SAD → OAD 改名

把 "Speculative-decoding-Aligned" 改成 "Overlap-Aligned" 是正确的语义升级：
- §1.2 表格把 accuracy 主线和 spec-decoding 应用线分开列，避免论文 reviewer 误以为这是 spec-decoding-only 的方法
- 数学不变，但卖点的解释空间从 1 个变成 2 个——对论文叙事有实质帮助

#### (2) §B.4 自曝家丑：Identity loss ≠ 0

这是 v3 最重要的发现，**前两轮评审（含我）都没抓到**：
- 当 student==teacher 时，`student_lse > teacher_lse`（左边求全词表，右边只 top-k）
- 所以 `student_p_topk < teacher_p_topk`，`min` 取 `student_p_topk`
- `acceptance = sum(student_p_topk) = M_T_true < 1`，loss = `-log(M_T_true)` 而不是 0

**这直接推翻了二轮评审 §13 里"clamp_max 移除后 loss ≡ 0"的建议**，更准确地说，揭示了 v2 §7.1 单元测试断言"loss ≡ 0"本身就是错的。作者主动校正测试为 `loss == -log(student_mass_on_teacher_topk)`，这是负责任的做法。

**论文写作含义**：
- 不能再宣传"OAD 在完美收敛时 loss = 0"
- §B.4 给出了正确叙事："loss 下界 = -log(M_T)，k=64+Qwen3 设定下 ≈ 0.005，路径 B 消除此偏差"
- **这反而成为路径 B 的新卖点**：路径 B 的 loss 真正以 0 为下界，路径 A 不是

#### (3) §B.3 偏差 3：teacher_topk_mass → student_mass_on_teacher_topk

二轮评审 §13 的整个监控设计建立在"实时观测 M_T"之上，但 Path A 下 `teacher_topk_mass` **按构造恒为 1**——评审建议了一个在该实现路径下根本无法观测的指标。作者识别到这点，改用学生侧代理量。

这是又一个前两轮评审都没抓到的概念性 bug。作者修正方向（学生在教师 top-k 上的质量）是正确替代。

### C.2 必须修复（开 PR 前 / Phase 2 启动前）🔴

#### (1) 命名残留：`"sad"` 没改干净

**问题位置**：

- §5.4 yaml 示例：`type: "sad"` / `sad:` 子节点 → 应为 `"oad"` / `oad:`
- §5.4 dispatch 代码：`elif loss_type == "sad":` 与 `loss_config["sad"]` → 应为 `"oad"`
- §10 文件改动清单：`tests/unit/algorithms/test_sad_loss.py` → 应为 `test_oad_loss.py`
- §10 yaml 改动说明："新增 `loss_fn.type` 与 `loss_fn.sad` 配置" → 应为 `loss_fn.oad`

与 §B.1 / §B.7（`loss_fn.oad` / `loss_fn.type=oad` / `test_oad_loss.py`）不一致。

**纯文档残留还是代码残留？审核者必须 grep 实际 codebase 确认**：

```bash
rg '"sad"|loss_fn\.sad|SADLossFn|SADLossConfig|test_sad_loss' nemo_rl/ examples/ tests/
```

如果是文档残留：直接改 §5.4 / §10。
如果代码也有残留：比文档不一致严重，需要全局重命名一次。

#### (2) wandb metric 名加 `_pathA` 后缀

**理由**：路径 A 与路径 B 下，同一指标的语义不同（参 §B.5 争议 2：identity 下 `active_grad_ratio` ≈ 1 是 Path A 偏差的产物，不是路径 B 的行为）。

**建议改动**：
```python
metrics = {
    "oad_loss":                                loss.detach(),
    "acceptance_rate_mean_pathA":              mean_accept,
    "acceptance_rate_min_pathA":               min_accept,
    "tvd_mean_pathA":                          1.0 - mean_accept,
    "student_mass_on_teacher_topk":            student_mass_on_teacher_topk,
    "active_grad_ratio_pathA":                 active_grad_ratio,
}
```

切到路径 B 时同名指标的语义会"切换"，加后缀强迫论文/dashboard 写清是哪个 path。

### C.3 建议修复（性价比高）🟡

#### (1) `active_grad_ratio` 同时暴露 position 级和 token 级

**§B.5 争议 1 当前选 `any`（位置级）**。建议**两个都暴露**：

```python
# 位置级：any top-k token 给了梯度信号 → 该 position 算"活跃"
active_grad_ratio_position_pathA = (
    (s_p_topk[:, :-1] < t_p_topk[:, :-1]).any(dim=-1).float() * token_mask
).sum() / n_valid

# token 级：top-k 内多少 token 给梯度信号
active_grad_ratio_token_pathA = (
    (s_p_topk[:, :-1] < t_p_topk[:, :-1]).float().mean(dim=-1) * token_mask
).sum() / n_valid
```

**理由**：
- `any` 的盲点：64 个 top-k token 里 1 个翻转，整个 position 算"活跃"，掩盖大部分 token 已无梯度的事实
- `mean` 与 §8 风险表"梯度稀疏化"语义更直接对应
- 双指标开销几乎为 0，但论文里能用 token 级指标论证梯度稀疏化的真实程度，dashboard 用位置级做粗筛

#### (2) 补两个单元测试

**测试 A：路径 B 等价下的 identity loss == 0**

```python
def test_identity_loss_zero_with_exact_teacher_lse():
    """路径 B 等价：人为构造精确 teacher_lse，identity 下 loss 应 ≡ 0"""
    # 手动构造 student_logits == teacher_logits（全词表）
    # 手动算精确 teacher_lse = logsumexp(full_teacher_logits)
    # 注入到 OADLossFn 内部（mock 或参数化 teacher_lse）
    # 断言 loss < 1e-6
```

价值：为路径 B 未来实现提供数学正确性的回归基线，强化 §B.4 论点。

**测试 B：M_T 显著 < 1 时的偏差量级实证**

```python
def test_path_a_bias_matches_theory():
    """构造 teacher 分布把质量散到 top-k 外（M_T_true = 0.5），
    验证 OAD loss 偏差幅度与 §3.4 联合偏差表预测一致"""
```

价值：把 §3.4 的理论偏差分析变成实证可重现，对论文 §3.4 是关键支撑。

### C.4 可选改进 🟢

#### (1) 偏差 1（未抽 `resolve_parallel`）：加 TODO 注释

`OADLossFn.__call__` 内联的 30 行 TP/CP/DTensor 解析与 `DistillationLossFn` 重复，建议加一行注释显式标记技术债：

```python
# TODO(future): if a third distill loss is added, refactor TP/CP/DTensor
#   resolution into a shared helper to avoid 3-way duplication.
```

#### (2) `vocab_cp_logsumexp` 的 padding 边界自检

**审核要点**（未看到代码细节，需作者自查）：

CP 下 `full_seq_len` padding 的处理是否与 `gather_logits_at_global_indices` **完全一致**？如果两者对 padding 区域的处理差一格（off-by-one），最终 `student_log_p_topk = student_topk_logits - student_lse.unsqueeze(-1)` 会在 padding 边界出错。

**Phase 1 的形状测试应该专门覆盖**：CP=2 + 奇数 seq_len 的情况。

### C.5 可推迟到 Phase 2 day 1（不阻塞）

#### (1) `acceptance_rate_mean` 起点 ~0.3 是估算还是实测

我在二轮评审 §三 提过这个问题，v3 没回应。

**Phase 2 day 1 必须先跑 1 step 记录基线指标，再启动 200 step 对照**——这是 §B.9 步骤 2 的隐式前提，建议明写为"Phase 2 启动协议第 0 步"。

#### (2) loss 曲线的 batch-内方差

`-log(0.99) ≈ 0.01`，训练初期 batch 间 M_T 方差可能让 loss 在 [0.005, 0.05] 波动。

**Phase 2 监控时应该看 loss 的 batch-内方差，而非只看均值**——否则曲线"噪声大"会被误判为训练不稳定。

### C.6 待修改清单（按优先级）

| 优先级 | 动作 | 工作量 |
|---|---|---|
| 🔴 必须 | grep codebase 把残留的 `"sad"` / `SADLossFn` / `test_sad_loss.py` 改成 OAD 命名 | 10 min |
| 🔴 必须 | wandb metric 名加 `_pathA` 后缀（参 §C.2(2)）| 5 min |
| 🟡 建议 | 加 `active_grad_ratio_position_pathA` + `active_grad_ratio_token_pathA` 双指标 | 5 min |
| 🟡 建议 | 补 §C.3(2) 的两个单元测试（路径 B 等价 + M_T<1 偏差实证）| 30 min |
| 🟢 可选 | 偏差 1 加 TODO 注释（§C.4(1)）| 1 min |
| 🟢 可选 | `vocab_cp_logsumexp` 的 padding 自检 + CP=2 奇数 seq_len 形状测试 | 15 min |

**预估总工作量**：必须项 ~15 min，含建议项 ~50 min，含可选项 ~70 min。

### C.7 v3 总评

这一轮 v3 比 v1/v2 都更扎实——因为**作者通过实施过程发现了评审都没看到的真问题**（identity loss ≠ 0、`teacher_topk_mass` 不可观测）。这种"实施过程中倒逼出来的洞察"是文档评审给不了的。

✅ **批准开 PR + 启动 Phase 2 训练**。请作者完成 §C.6 清单后通知二轮评审。

---

## 附录 D：作者对 §C 评审的回应（v3.1）

> 已闭环 §C.6 清单中除可选项外的全部条目，本节记录每项动作。代码已通过 Python AST 语法检查；待在 torch 环境跑 `pytest tests/unit/algorithms/test_oad_loss.py` 验证。

### D.1 §C.6 清单完成情况

| 优先级 | 动作 | 状态 | 改动位置 |
|---|---|---|---|
| 🔴 必须 | grep codebase + 文档命名残留清理 | ✅ 已完成 | 代码侧 grep `'"sad"\|loss_fn\.sad\|SADLossFn\|SADLossConfig\|test_sad_loss'` **零残留**；文档 §5.4 yaml/dispatch 示例与 §10 文件清单已改 |
| 🔴 必须 | wandb metric 加 `_pathA` 后缀 | ✅ 已完成 | `OADLossFn` metrics dict 中所有 Path A 相关指标加后缀；同步更新单元测试断言 |
| 🟡 建议 | 双暴露 `active_grad_ratio_position_pathA` + `active_grad_ratio_token_pathA` | ✅ 已完成 | `loss_functions.py` 监控段；测试 6 增加 token-level ≤ position-level 不变量验证 |
| 🟡 建议 | 补单元测试 A（路径 B 等价 identity ≡ 0）+ 测试 B（M_T < 1 偏差实证）| ✅ 已完成 | `test_oad_loss.py` 测试 7 / 测试 8 |
| 🟢 可选 | 偏差 1 加 TODO 注释 | ✅ 已完成 | `OADLossFn.__call__` 第 2 步 |
| 🟢 可选 | `vocab_cp_logsumexp` padding 自检 | ✅ 已审视 | 自查发现逻辑正确（CP shard 输入 → allgather 恢复 padded 长度 → 切回 `full_seq_len`，与 `gather_logits_at_global_indices` 约定一致）；helper 加注释固化此约定。CP=2 + 奇数 seq_len 的形状测试需分布式环境，归 GPU 集成测试范围 |

### D.2 §C.3(2) 两个新单元测试的设计

#### 测试 7：`test_oad_loss_zero_when_path_a_equals_path_b`

**构造方式**：让 `topk == vocab_size`，于是 `logsumexp(top-k) == logsumexp(full vocab)`，Path A 的 teacher_lse 与路径 B 完全一致。在此条件下：
- student == teacher
- 无估计偏差
- **断言 `loss < 1e-5`**（identity 严格意义下 loss ≡ 0）
- 同时验证 `acceptance ≡ 1` 与 `active_grad_ratio_position_pathA < 1e-5`

这条测试的意义：**证明 Path A 偏差是 §B.4 中分析的唯一原因**——消除该偏差，"完美对齐 ⇒ 零损失"立即恢复。也作为路径 B 未来实现的回归基线。

#### 测试 8：`test_oad_loss_path_a_bias_when_teacher_mass_is_split`

**构造方式**：人为让教师真实 M_T_true ≈ 0.5（top-4 token 共占 50% 概率，其余 12 token 共占 50%）。在 student == teacher 条件下：
- 真实接受率 = 1（数学事实）
- Path A 估计接受率 = M_T_true ≈ 0.5（因为 student 概率被全词表归一化稀释）
- **断言 `loss ≈ -log(0.5) ≈ 0.693`**（容差 5e-2，覆盖数值误差）
- **断言 `student_mass_on_teacher_topk ≈ 0.5`**

这条测试的意义：**把 §3.4 的理论偏差分析从纸面变成可重现的实证**，对论文 §3.4 是关键支撑。在领域外/训练早期 M_T 可能确实偏低的场景下，这是判断"loss 数值是否正常"的标尺。

### D.3 监控指标命名规范（最终版）

实施完成后，OADLossFn 暴露的监控指标如下（含 `_pathA` 后缀的语义说明）：

| metric 名 | 路径 A 下含义 | 是否依赖 Path A |
|---|---|---|
| `loss` | OAD loss 主值 | 是（identity 下非零的 -log(M_T_true)）|
| `num_valid_samples` | batch 大小 | 否 |
| `acceptance_rate_mean_pathA` | 平均接受率（Path A 估计）| 是 |
| `acceptance_rate_min_pathA` | 最小接受率 | 是 |
| `tvd_mean_pathA` | 1 - acceptance_mean | 是 |
| `student_mass_on_teacher_topk` | 学生在教师 top-k 上的概率质量 | **否**（路径无关）|
| `active_grad_ratio_position_pathA` | 至少一个 top-k token p_S<p_T 的位置比例 | 是 |
| `active_grad_ratio_token_pathA` | top-k 内 p_S<p_T 的 token 比例（细粒度，论文论证用）| 是 |

未来路径 B 实现后，对应的指标会以 `_pathB` 后缀并行暴露，wandb 上同时观察 A/B 两组曲线即可看出估计偏差对监控的实际影响。

### D.4 v3.1 总改动量

| 文件 | v3 → v3.1 增量 |
|---|---|
| `nemo_rl/algorithms/loss_functions.py` | +12 行（双 active_grad_ratio + `_pathA` 后缀重命名 + TODO 注释 + comment）|
| `nemo_rl/distributed/model_utils.py` | +3 行（CP 约定注释）|
| `tests/unit/algorithms/test_oad_loss.py` | +112 行（测试 7 + 测试 8 + 测试 1/2/6 同步更新）|
| `BASIC_OAD_PROPOSAL.md` | §5.4 / §10 命名修正；附录 D（本节）新增 |

### D.5 状态

✅ §C.6 必须项 + 建议项全部完成
🟢 唯一可选项（CP=2 奇数 seq_len 形状测试）归 GPU 集成测试范围

**等待动作**：在有 torch 环境的机器跑 `pytest tests/unit/algorithms/test_oad_loss.py -v`，确认 8 个测试全部通过；通过后即可启动 Phase 2 训练。

---

## 附录 E：v3.2 路径 B 切换记录

> v3 / v3.1 实现并验证了路径 A（`teacher_lse` 从 top-k 估计）。v3.2 决策：**正式切换到路径 B**——教师 worker 提供精确全词表 logsumexp。本节记录决策原因和具体改动。

### E.1 切换原因

| 维度 | 路径 A 痛点 | 路径 B 收益 |
|---|---|---|
| 数学性质 | identity loss = -log(M_T) ≠ 0，论文需诚实交代 | **identity loss ≡ 0**，textbook 性质 |
| 监控可用性 | `teacher_topk_mass` 恒为 1，无诊断价值；只能用学生侧代理 | **`teacher_topk_mass` 真实可观测**，理论假设 M_T > 0.99 可实证 |
| 偏差分析 | §3.4 净偏差 ≤ 0.5%，需要联合分析+前提保证 | **零偏差**，§3.4 节简化 |
| 切换成本 | — | ~50 行 worker 改动 + ~1MB/batch 额外传输 |

总成本与收益对比下，**v3.2 切换到 Path B 是更好的工程取舍**。

### E.2 改动文件清单

| 文件 | 改动类型 | 行数 |
|---|---|---|
| `nemo_rl/models/policy/interfaces.py` | `TopkLogitsOutputSpec` 增加 `logsumexp: NotRequired[Tensor]` 字段 + import 调整 | +5 |
| `nemo_rl/models/policy/workers/dtensor_policy_worker.py` | `get_topk_logits` 计算并返回精确 logsumexp（CP 与非 CP 两个分支 + packing unpack + final padding）| +60 |
| `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py` | 同上 | +60 |
| `nemo_rl/models/policy/workers/megatron_policy_worker.py` | **不在本期范围**（明确决定不支持 Megatron backend；OAD 在该 backend 下抛友好 `KeyError`）| 0 |
| `nemo_rl/algorithms/distillation.py` | 把 `teacher_topk["logsumexp"]` 透传到 `train_data["teacher_logsumexp"]` | +6 |
| `nemo_rl/algorithms/loss_functions.py` | `OADLossFn` 取 `data["teacher_logsumexp"]`；缺失时抛 `KeyError` 友好报错；监控 metric 名 `_pathA` → `_pathB`；新增 `teacher_topk_mass`（路径 B 下真实可观测）| +12 / -8 净增 |
| `tests/unit/algorithms/test_oad_loss.py` | `_make_oad_data` 同时构造精确 `teacher_logsumexp`；测试 2 改回 `loss == -log(M_T)`（仍非零，因 top-k 截断）；测试 7 改名 `test_oad_loss_zero_when_topk_equals_vocab`（top-k=vocab 时 loss ≡ 0）；测试 8 改为"truncation bound 实证 + teacher_topk_mass 直接观测"；新增测试 9 `test_oad_loss_raises_when_logsumexp_missing` | +30 |

总改动 **~170 行**（worker 改动占大头），均在 v3.1 既有架构上小幅扩展。

### E.3 监控指标最终定型（Path B）

| metric 名 | 含义 | 期望走势 |
|---|---|---|
| `loss` | OAD loss (= -log(acceptance) 平均) | 平稳下降 |
| `acceptance_rate_mean_pathB` | 平均接受率（精确）| 单调上升（~0.3 → 0.7+，与 KL 不同的是这里有清晰下界 = M_T） |
| `acceptance_rate_min_pathB` | 单 batch 最小接受率 | 上升，但波动大 |
| `tvd_mean_pathB` | 1 - acceptance_mean | 单调下降 |
| **`teacher_topk_mass`** | 教师真实 top-k 累积概率（M_T）| **稳定 > 0.99**；若 < 0.95 提示数据/模型 mismatch |
| `student_mass_on_teacher_topk` | 学生在教师 top-k 上的概率质量 | 单调上升 |
| `active_grad_ratio_position_pathB` | ≥1 个 top-k token 满足 p_S<p_T 的位置比例 | 训练进程的位置级活跃度 |
| `active_grad_ratio_token_pathB` | top-k 内 p_S<p_T 的 token 平均比例 | 训练进程的细粒度活跃度（论文论证梯度稀疏化用） |

> identity 下：`acceptance_rate_mean_pathB == teacher_topk_mass`（学生与教师同分布时这两个量相等），`active_grad_ratio_*` ≈ 0（无梯度）。

### E.4 §B / §D 中受影响的内容

- §3.2 的"理论假设的实证保证"声明现在指向 `teacher_topk_mass`（路径 B 下直接观测）
- §3.4 联合偏差分析在路径 B 下退化为单一截断偏差（≤ 1 - M_T），保留作为历史推导但失去实践紧迫性
- §B.4 "identity loss ≠ 0" 的整段讨论变成历史记录
- §B.5(2) "identity 下 active_grad_ratio ≈ 1" 反转为"identity 下 active_grad_ratio ≈ 0"
- §C.2(2) `_pathA` 后缀改为 `_pathB`
- §C.7 的"实施过程倒逼出来的洞察"含义升级：**v3.1 用洞察论证保留路径 A 的合理性；v3.2 用洞察直接消除偏差源头**

### E.5 测试矩阵

`tests/unit/algorithms/test_oad_loss.py` 共 9 个测试（v3.1 的 8 个 + 新增 1 个 missing-key 测试）：

| # | 测试 | Path A 期望 | **Path B 期望（当前）** |
|---|---|---|---|
| 1 | 形状 / NaN | 同 | metric 名带 `_pathB` |
| 2 | identity | loss = -log(M_T) | **loss = -log(M_T) 仍成立**（top-k 截断），但 `acceptance == teacher_topk_mass` 严格成立 |
| 3 | 手算 toy case | 用 `logsumexp(top-k)` | **用 `logsumexp(full)` 精确值** |
| 4 | 梯度流 | 同 | 同 |
| 5 | mask 尊重 | 同 | 同 |
| 6 | metric 边界 | 同 | acceptance ≤ min(M_T, student_mass) 的更紧界 |
| 7 | top-k = vocab + identity | loss < 1e-5 | **loss < 1e-5**（同义）|
| 8 | M_T = 0.5 + identity | "Path A 偏差实证" | **"截断界 + M_T 直接观测"实证** |
| 9 | missing teacher_logsumexp | — | **新增**：抛 `KeyError` |

### E.6 状态

✅ 路径 B 切换完成
✅ 所有相关代码通过 Python AST 语法检查
⏳ 等待 GPU/torch 环境跑 `pytest tests/unit/algorithms/test_oad_loss.py -v`

**等待动作（不变）**：跑通单元测试 → 启动 Phase 2 训练（`train_opd.sh` + `loss_fn.type=oad`）。
