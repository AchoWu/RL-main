# OPD — On-Policy Distillation

## 1. 概述

**OPD (On-Policy Distillation)** 是本仓库中通过 `nemo_rl/algorithms/distillation.py`
提供的知识蒸馏算法。它属于 NeMo RL 支持的核心算法之一（GRPO / GSPO / DAPO / DPO /
SFT / **On-policy Distillation**），核心思想是：

- 学生 (student) 模型自己在线 (on-policy) 生成回答；
- 教师 (teacher) 模型对**学生生成的序列**给出 top-k logits（及可选的 full-vocab
  logsumexp）；
- 学生通过 KL 散度或 OAD (Overlap-Aligned Distillation) 损失，向教师分布对齐。

与传统离线蒸馏最大的区别：数据流不是"教师生成 → 学生模仿"，而是"学生生成 →
教师打分 → 学生更新"，因此更贴近学生自身的分布，避免 distribution shift。

入口脚本 & 配置：
- `train_opd.sh` — 项目里的一键训练脚本（Qwen3-1.7B 学生 / Qwen3-4B 教师）
- `examples/run_distillation_math.py` — Python 入口
- `examples/configs/distillation_math.yaml` — 默认配置

关键源码：
- `nemo_rl/algorithms/distillation.py` — 训练主循环（`setup`, `distillation_train`,
  `validate`）
- `nemo_rl/algorithms/loss_functions.py` — `DistillationLossFn` (KL) 与 `OADLossFn`

## 2. 数据流与训练循环

一个 step 的核心流程（简化自 `distillation_train`，`distillation.py:586`–`731`）：

```
for batch in dataloader:
    # ① 学生策略生成（vLLM / megatron）
    repeated_batch = batch.repeat_interleave(num_generations_per_prompt)
    refit_policy_generation(student_policy, student_generation)   # 权重同步到推理引擎
    repeated_batch, rollout_metrics = run_multi_turn_rollout(
        policy_generation=student_generation, ...
    )

    # ② 组装训练数据（token_loss_mask 只保留 assistant 段）
    flat_messages, input_lengths = batched_message_log_to_flat_message(...)
    train_data = BatchedDataDict[DistillationLossDataDict](
        input_ids, input_lengths, token_mask, sample_mask
    )

    # ③ 教师 top-k logits 推理
    teacher_policy.prepare_for_lp_inference()
    teacher_topk = teacher_policy.get_topk_logits(train_data, k=topk_logits_k)
    train_data["teacher_topk_logits"]  = teacher_topk["topk_logits"]
    train_data["teacher_topk_indices"] = teacher_topk["topk_indices"]
    if "logsumexp" in teacher_topk:                # OAD Path B
        train_data["teacher_logsumexp"] = teacher_topk["logsumexp"]

    # ④ 学生训练一步
    teacher_policy.offload_after_refit()
    student_policy.prepare_for_training()
    train_results = student_policy.train(train_data, loss_fn)
```

关键设计点：

1. **在线采样** — 学生自己在 vLLM 上生成，每一步都通过 `refit_policy_generation`
   把最新权重同步到推理引擎（`POLICY_GENERATION_STALE` 标志位控制）。
2. **仅蒸馏 assistant token** — 通过 `token_loss_mask` (assistant=1, 其他=0) 屏蔽
   prompt 部分（`distillation.py:667`–`676`）。
3. **教师 top-k 而非 full-vocab** — 教师返回 `topk_logits_k`（默认 64）个位置的
   logits，节省显存/带宽；OAD 额外要求 full-vocab `logsumexp`。
4. **Colocation** — 学生训练 & 学生推理共享 GPU（`generation.colocated.enabled=true`），
   教师训练 worker 与学生训练 worker 分别管理 offload。

## 3. 支持的损失函数

由 `loss_fn.type` 分派（`distillation.py:477`–`485`）：

### 3.1 KL 蒸馏 (`type: "kl"`)

`DistillationLossFn` (`loss_functions.py:982`) —— 三种 KL 模式：

- `kl_type: "forward"` — `KL(p_T || p_S)` = `p_T * log(p_T / p_S)`；教师驱动，覆盖
  教师概率高的位置。
- `kl_type: "reverse"` — `KL(p_S || p_T)` = `p_S * log(p_S / p_T)`；学生驱动，
  mode-seeking。
- `kl_type: "mixed"` — `w * forward + (1-w) * reverse`；`mixed_kl_weight` 控制权重。

**两种 top-k 处理策略：**

- `zero_outside_topk: false`（默认）— 把教师和学生分布都截断到 top-k，然后**在这
  k 个 token 上重新归一化**再计算 KL（`loss_functions.py:1141`–`1147`）。数学上等
  价于把 top-k 以外全部置零 + 除以 top-k 概率和。
- `zero_outside_topk: true` — 用真正的 full-vocab logprob（不重新归一化），把
  top-k 外的部分作为熵校正项 `H_rest - log_infinitesimal * P_rest` 加回来
  （`loss_functions.py:1170`–`1178`）。分布式路径使用
  `ChunkedDistributedGatherLogprob` + `ChunkedDistributedEntropy` 分块计算，兼容
  TP+CP。

**块级 reverse KL（`block_size` / `block_mode`）：**

逐 token reverse KL 在学生采样出的前缀上算每个位置的 `KL_t`，再按 token 平均。它
只对 `KL_t` 本身求导，丢掉了"当前 token 的选择会改变后续位置的 `KL_t'`"这部分梯度。
整条序列的 reverse KL 由链式法则可写成 `E_{y~q}[ sum_t KL_t ]`，它的梯度是

```
grad = E[ sum_t grad KL_t + sum_t grad log q(y_t) * sg( sum_{t' > t} KL_t' ) ]
```

第二项相当于"后续 KL 之和"作为 cost-to-go 的 score-function 项。`block_size > 1`
时保留这一项，但把 cost-to-go 截断到 k 个 token 的范围内。实现上
（`_block_kl_to_go`），每个位置的 loss 是

```
KL_t + (log q(y_t) - sg(log q(y_t))) * sg(R_t)
```

第二项前向值为 0，所以日志里的 `loss` 仍是逐 token KL 均值，和 baseline 可直接比较。
`log q(y_t)` 是学生在全词表下对采样 token 的 logprob（即采样分布），
取法与 `ClippedPGLossFn` 共用 `_get_next_token_logprobs`。两种截断方式
（`KL_{+j}` 表示 t 之后第 j 个 response token 的 KL）：

- `block_mode: "fixed"` — 从第一个 response token 起每 k 个切一块，互不重叠，最后一块
  可不满。`R_t` 为 t 所在块内、t 之后各 token 的 KL 之和。这正好是
  `sum_块 KL(q(y_块 | 前缀) || p(y_块 | 前缀))` 在**块前缀分布 stop-grad** 下的精确
  梯度：块内的 token 对后续块的影响不计入。块首 token 看 k-1 步，块尾 token 看 0 步。
- `block_mode: "sliding"` — `R_t = sum_{j=1}^{k-1} (k-j)/k * KL_{+j}`。它等于把
  fixed 的块边界平移 0..k-1 个 token 后取平均（`KL_{+j}` 与 t 同块的概率是
  (k-j)/k），也等于"所有 k-token 滑窗的块 KL 之和 / k"的梯度——这里的滑窗包括在
  response 两端被截断的窗口（起点从 1-k 到 L-1，L 为 response 长度）；只算完整窗口时
  两端 token 的 `grad KL_t` 计入次数不足 k，就不是这个目标了。和 fixed 优化的是同一
  件事，只是去掉了块边界带来的偏差，score 项的平均权重也与 fixed 相同（(k-1)/2）。
- `block_size: 1`（默认）— `R_t = 0`，与原逐 token KL 完全等价；`block_size` ≥
  response 长度的 fixed 即为整条序列 reverse KL 的梯度。

**目标的确切含义。** `KL_t` 是在 teacher top-k 上**重新归一化**后的 KL（`KL~_t`），而
采样分布是全词表 q。所以优化的是 `E_{y~q}[ sum KL~_t ]` 的（截断后）精确梯度；只有
top-k 覆盖整个词表时，它才等于截断前块联合分布的 KL。

**限制与注意事项：**

- 只支持 `kl_type: "reverse"`：期望是对学生采样取的，forward KL 需要教师采样。
- 不支持 `zero_outside_topk: true`：它给每个 token 加 `100 * P_rest` 的替代项
  （P_rest=0.2 时约为 20），累加进 `R_t` 后会达到上百，压过 KL 项本身，
  `max_grad_norm` 裁剪又会把 KL 梯度一起缩小。
- `R_t >= 0`，不减 baseline 时 score 项会按后续 KL 统一压低每个采样 token 的概率，
  期望正确但噪声随 k 增大。`block_baseline: true` 时改乘 `R_t - b_t`
  （`_block_kl_baseline`），`b_t` 是同一条序列里**已由 y_<t 确定**的 `R_s` 的均值：
  `R_s` 用到的 KL 最远到 s+k-1，位置 m 的 KL 只依赖 m 之前的 token，所以
  s <= t-k+1 时 `R_s` 与 y_t 无关，`E[grad log q(y_t) * b_t] = 0`，仍然无偏。
  fixed 取之前各块里与 t 块内位置相同的 `R_s`（块内不同位置的 `R` 本身差很多），
  sliding 取全部 s <= t-k+1；没有这样的 s 时 `b_t = 0`。没用 batch / 序列均值：
  `train_micro_batch_size=1` 时 loss 只看得到一条序列，序列均值含 t 之后的 KL，
  相当于多减了一个缩放后的整条序列 cost-to-go，会引入偏差。
- score 项要求样本来自**当前**学生（on-policy）：
  - `train_global_batch_size` 不等于每步 rollout 数时（默认 YAML 为 64 < 128），同一批
    rollout 会更新多次，第二次起样本来自旧策略，score 项有偏（代码不做 importance
    ratio 修正）。所以 `block_size > 1` 时 `distillation.setup` 会 assert
    `train_global_batch_size == num_prompts_per_step * num_generations_per_prompt`；
    `train_bopd.sh` 用 gbs=128，每批 rollout 只更新一次。
  - vLLM 与训练端 logprob 的数值差异会带来小偏差。
  - 生成时如果 `top_p < 1` 或设置了 `top_k`，采样分布就不是 q，也会引入偏差；
    默认 `temperature=1.0, top_p=1.0, top_k=null` 没有这个问题。
- 显存：要多做一次全词表 log_softmax，结果保留到 backward。16k 序列在 tp=cp=1 时
  每个 microbatch 约多占 10 GB，TP2/CP2 时每卡约 2.5 GB。
- 单测 `test_distillation_block_kl_gradient_matches_exact_objective` 在 V=3 的玩具
  自回归模型上枚举全部序列（多块、top-k < V、多 token prompt、右侧 padding），验证
  期望梯度等于上述目标的精确梯度；`test_block_kl_to_go_sliding_is_fixed_averaged_over_offsets`
  验证 sliding = fixed 在所有块偏移下的平均。梯度测试同时跑 `block_baseline` 开/关，
  开启时期望梯度仍须等于精确目标，即验证 baseline 无偏。

### 3.2 OAD (`type: "oad"`) — Overlap-Aligned Distillation

`OADLossFn` (`loss_functions.py:1261`)。设计动机见 `BASIC_OAD_PROPOSAL.md`。

**核心公式：**

```
Loss_t = -log( sum_{y in top-k of teacher} min(p_S(y|y_<t), p_T(y|y_<t)) )
```

即"每一步 loss = -log(学生与教师在 top-k 上的重叠率)"。重叠率 (`acceptance`)
恰好是 speculative-decoding 里的**接受率**，也等价于 `1 - TVD`
（总变差距离）在 top-k 支撑集上的近似值。

**实现要点 (Path B):**

- 学生 `logsumexp` 由 `vocab_cp_logsumexp` 精确计算（TP+CP 感知）。
- 教师 `logsumexp` 从 worker 精确传入（`data["teacher_logsumexp"]`）— 保证
  `student == teacher` 时 `loss = 0`，无系统偏差。
- Path B 目前**要求 DTensor 后端**；Megatron 后端暂未实现 `logsumexp` 输出，
  会在 loss 处抛 `KeyError` 提示。
- 相比 KL：OAD 只在**学生高估**的 token 上产生梯度
  (`p_S > p_T` 时 `min = p_T`，对学生无梯度)；OAD 论文认为这更贴合"截断学生
  超过教师的部分"，避免破坏学生已经掌握的知识。

**监控指标** (`loss_functions.py:1412`–`1457`) —— 关键点是返回**每-microbatch 的
"和" + token 计数**，而不是已经取均值的量。因为
`dtensor_policy_worker_v2.py:863` 会把每个 loss-fn metric 预先除以
`num_global_batches`，然后 `distillation.py:785` 用 `np.sum` 跨 microbatch 汇总，
这个"预除的和"trick 只对 sum-style 量成立。返回 sum + count 后，
`distillation.py:898`–`931` 直接打印 `sum / count` 得到真实的 `[0, 1]` 概率均值：

- `oad_acceptance_sum / oad_token_count` = 平均接受率 (Path B 精确)
- `oad_teacher_topk_mass_sum / oad_token_count` = 教师 top-k 覆盖的概率质量
  `M_T`（用于评估 top-k 截断带来的偏差上界 `1 - M_T`）
- `oad_student_mass_on_teacher_topk_sum / oad_token_count` = 学生在教师 top-k 上
  的概率质量
- `oad_active_grad_position_sum / oad_token_count` = 至少有一个 token 学生
  过估的位置比例
- `oad_active_grad_token_sum / oad_token_count` = token 级别的过估比例
- `oad_min_accept_pathB` = 该批次最小接受率（诊断异常位置）

## 4. 配置结构

摘自 `examples/configs/distillation_math.yaml`：

```yaml
distillation:
    num_prompts_per_step: 128           # 每步 prompt 数
    num_generations_per_prompt: 1        # 每 prompt 生成数（KL/OAD 通常=1）
    max_rollout_turns: 1                 # 数学题单轮
    max_num_steps: 1000
    max_num_epochs: 10
    val_batch_size: 64
    val_period: 20
    val_at_start: false
    max_val_samples: 512
    topk_logits_k: 64                    # 教师 top-k 截断
    seed: 42

loss_fn:
    type: "kl"                           # 或 "oad"
    kl_type: "mixed"                     # forward / reverse / mixed
    mixed_kl_weight: 0.5
    zero_outside_topk: false
    block_size: 1                        # >1: 块级 reverse KL
    block_mode: "fixed"                  # fixed / sliding（fixed 对所有块偏移的平均）
    block_baseline: false                # true: score 项减去只依赖 y_<t 的 baseline
    oad:
        eps: 1.0e-8

policy: &POLICY_BASE
    model_name: "Qwen/Qwen3-1.7B-Base"   # 学生
    train_global_batch_size: 64
    train_micro_batch_size: 1
    max_total_sequence_length: 8192
    precision: "bfloat16"
    dtensor_cfg:
        enabled: true
        tensor_parallel_size: 2
        context_parallel_size: 2
    optimizer: {name: torch.optim.AdamW, kwargs: {lr: 2.0e-5, ...}}
    generation:
        backend: "vllm"
        vllm_cfg:
            tensor_parallel_size: 1
            gpu_memory_utilization: 0.6
        colocated: {enabled: true}

teacher:
    <<: *POLICY_BASE
    model_name: "Qwen/Qwen3-4B"          # 教师
    dtensor_cfg:
        tensor_parallel_size: 4
        context_parallel_size: 2

data:
    dataset_name: "DeepScaler"
    prompt_file: "examples/prompts/cot.txt"

env:
    math:
        num_workers: 8

cluster:
    gpus_per_node: 8
    num_nodes: 1
```

**TypedDict 层次**（`distillation.py:74`–`122`）：

```
MasterConfig
├── policy:       PolicyConfig            # 学生
├── teacher:      PolicyConfig            # 教师
├── loss_fn:      DistillationLossConfig  # 或 OADLossConfig
├── env:          dict[str, Any]
├── data:         DataConfig
├── distillation: DistillationConfig
├── logger:       LoggerConfig
├── cluster:      ClusterConfig
└── checkpointing: CheckpointingConfig
```

## 5. 前置约束

1. **词表严格一致** — `check_vocab_equality` (`distillation.py:128`) 会检查：
   - `tokenizer.get_vocab()` 完全相等
   - `len(tokenizer)` 相等
   - `AutoConfig.vocab_size` 相等

   任一失败即抛 assert。可通过 `NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true` 跳过
   （慎用）。

2. **DTensor 后端约束** — 不支持 DTensor SP + sequence_packing 同时开启（见
   `distillation.py:210`–`214`，issue #1178）。

3. **OAD 后端约束** — 目前 OAD Path B 只支持 DTensor（Megatron worker 暂未输出
   `logsumexp`）。

4. **Colocation & Refit** — `POLICY_GENERATION_STALE` 状态机确保每次 rollout 前
   学生权重被同步到 vLLM；训练完成后再翻转为 stale。

## 6. Checkpoint

- 学生 policy weights / optimizer / tokenizer / dataloader state 通过
  `CheckpointManager` 保存（`distillation.py:848`–`872`）。
- 支持 top-k 保留（`checkpointing.keep_top_k`），依据 `metric_name`
  （如 `"val:accuracy"`）选取最优。
- 教师**不保存**（`init_optimizer=False, init_reference_model=False`，
  `distillation.py:386`–`395`）。
- 输出仍为 PyTorch DCP，需要用 `examples/converters/convert_dcp_to_hf.py`
  转回 HF 格式。

## 7. 运行方式

**项目自带的 8-GPU 训练脚本** (`train_opd.sh`):

```bash
bash train_opd.sh
# 等价于：
python examples/run_distillation_math.py \
    --config examples/configs/distillation_math.yaml \
    policy.model_name="<local>/Qwen3-1.7B/" \
    teacher.model_name="<local>/Qwen3-4B/" \
    cluster.gpus_per_node=8 \
    policy.train_micro_batch_size=1 \
    teacher.logprob_batch_size=2 \
    distillation.max_num_epochs=3 \
    checkpointing.save_consolidated=true
```

**通用命令：**

```bash
# 默认 KL-mixed 蒸馏
uv run python examples/run_distillation_math.py

# 切换到 OAD
uv run python examples/run_distillation_math.py loss_fn.type=oad

# 多 GPU
uv run python examples/run_distillation_math.py cluster.gpus_per_node=8
```

## 8. 相关文档

- `BASIC_OAD_PROPOSAL.md` — OAD 数学推导、Path A/B 讨论、经验记录
- `BASIC_EM_KL_PROPOSAL.md` — 相关的 EM-KL 变体草案
- `docs/design-docs/loss-functions.md` — 损失函数框架
- `docs/design-docs/training-backends.md` — DTensor vs Megatron
- `docs/design-docs/generation.md` — GenerationInterface / vLLM refit 流程
