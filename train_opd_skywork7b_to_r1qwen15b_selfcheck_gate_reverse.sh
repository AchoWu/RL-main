#!/usr/bin/env bash
# ============================================================================
# OPD + Teacher Self-Check Gate (idea draft, not yet implemented in code)
# ============================================================================
# Motivation
#   标准 OPD 中，教师会在学生已经走偏的前缀上继续 teacher-force。
#   顺着错前缀纠正对学生学 backtrack 有一点用，但更多时候教师只是
#   在错的轨道上把损失继续算下去，稀释真正有信号的段。
#
# Idea
#   在学生 rollout 的自然切分点（初版：段落级，形如 "\n\n" 之后，
#   或 "Step N:" / "Wait," / "Actually," 这类回溯词）插入一个探针，
#   问教师模型"这一段之前的推理是对的还是错的"，把它给的 log-ratio
#   当成 gate。ratio 低于阈值的段，后续 token 不计 loss。
#
# Scoring: 组合方案 = (2) contrastive baseline + (3) single-token probe
#
#   (3) Single-token probe (trajectory-oriented, 判"未来"而非"过去")
#       统一模板：
#           <student_prefix>
#           Self-check: from here I should
#       只取下一 token 位置的 logits，比较 " continue" vs " restart"
#       (Qwen tokenizer 已确认两者均单 token: 3060 / 17024)。
#
#       之所以问"from here I should"而不是"reasoning so far is
#       correct"：R1 类教师的长 CoT 里频繁自我纠错，"迄今结果"经常是
#       错的但教师即将回头修正。这类段落是学生最该学的 backtrack，
#       不能被 gate 掉。问"接下来该怎么走"能让"知错要改"的段落回落
#       到 continue 侧，只有真正走死胡同才被 restart。
#
#   (2) Contrastive baseline (per-rollout, computed at gate time)
#       每条 rollout 在 gate 之前先算一次自己的 baseline b_i：
#           b_i = log P(continue | 题干 + chat opener + probe)
#               - log P(restart  | 题干 + chat opener + probe)
#       用"题干 + chat template 到 assistant 位置"作为中性锚点（此时
#       学生还没产出任何 token，是 problem-specific 的零推理起点）。
#       实际打分:
#           s_k = log P(continue | prefix_k + probe)
#               - log P(restart  | prefix_k + probe) - b_i
#       阈值卡在 0 附近，负得越多越像"教师判走死了"。
#
#       选 per-rollout 而非全局常数的理由：初期实测中性 stub 上
#       std=1.37, range=[3.4, 8.8]，全局常数被 stub/题目 prior 支配，
#       信号淹没在偏置里；per-rollout 每条 rollout 自己校准，把"这题
#       教师本来就更犹豫"的偏置消掉。
#       落地：训练时每条 rollout 前多一次 probe forward，几乎零成本
#       (KV cache 可复用 chat opener 部分)。
#       Offline diagnostic: research/selfcheck_gate/estimate_baseline.py
#
# Gate behavior
#   - 触发点：段落级切分（初版 "\n\n"，可再收紧到 backtrack 词）
#   - 每段一次教师 forward（teacher 已在做 teacher-forcing，只多跑
#     probe 尾巴那几个 token，成本可控）
#   - s < tau  → 该段之后的 token loss weight 置 0（或 gate 该段之后
#     的整个 sequence，待消融决定）
#   - 与现有 sequence-confident / teacher-margin gate 是同一问题的
#     semantic 版本；先做 correlation 消融再决定互补/替换
#
# Open questions（讨论中，先不写死）
#   - 触发点粒度：段落 vs 只在 backtrack 词触发
#   - "承认错误并回头"的段该保留还是屏蔽 → probe 措辞需能区分
#   - 是屏蔽后续 token loss 还是整条序列丢弃
#   - baseline b 是全局常数还是随 batch 重估
#   - 与已有 gate 叠加时的先后顺序
#
# 本脚本 = 32k reverse baseline + 待接入的 loss_fn.teacher_selfcheck_gate.*
# 命令行 override 目前是占位；等 nemo_rl 侧接口落地后再启用（现在跑会因
# 未知 key 报错，故整段 self-check 参数暂用 shell 变量记录，不下发）。
# ============================================================================

set -euo pipefail
trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python /group/40092/howu/PaddleOCR/test_gpu.py
' EXIT INT TERM

# ====== conda 环境 ======
source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd
echo "▶ Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# ====== 代理 ======
: "${ENV_VENUS_PROXY:=http://star-proxy.oa.com:3128}"
export NO_PROXY=localhost,127.0.0.1,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

# ====== HF 缓存 ======
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# ====== 忽略师生词表一致性检查 ======
export NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true

# ====== NCCL / vLLM 超时 ======
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800

# ====== WandB ======
export WANDB_MODE=online
export WANDB_PROJECT=nemo-distillation

# ====== Ray / venv ======
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
rm -rf /group/40092/howu/RL-main/venvs
ray stop --force 2>/dev/null || true
sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' \
    /group/40092/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py

export PYTHONPATH=/group/40092/howu/RL-main:${PYTHONPATH:-}

# ====== 模型路径 ======
TEACHER_MODEL="/dev/shm/llms/Skywork-OR1-Math-7B/"
POLICY_MODEL="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
for d in "$TEACHER_MODEL" "$POLICY_MODEL"; do
  if [[ ! -f "${d}config.json" ]]; then
    echo "❌ 模型缺失: ${d}config.json 不存在"; exit 1
  fi
done
echo "▶ Using policy=$POLICY_MODEL  teacher=$TEACHER_MODEL"

# ====== Self-check gate 超参（占位，尚未在 nemo_rl 接入） ======
# 打开后需要在 loss_fn / distillation 里新增以下配置项:
#   loss_fn.teacher_selfcheck_gate.enabled: bool
#   loss_fn.teacher_selfcheck_gate.probe_template: str
#     e.g. "\nSelf-check: the reasoning so far is"
#   loss_fn.teacher_selfcheck_gate.positive_token: str   # " correct"
#   loss_fn.teacher_selfcheck_gate.negative_token: str   # " wrong"
#   loss_fn.teacher_selfcheck_gate.split_pattern: str    # "\n\n"
#   loss_fn.teacher_selfcheck_gate.threshold: float      # tau, 应用于 s
#   loss_fn.teacher_selfcheck_gate.baseline_logratio: float  # 常数 b
#   loss_fn.teacher_selfcheck_gate.baseline_prompts_path: str  # 预算 b 用
#   loss_fn.teacher_selfcheck_gate.mask_scope: str  # "suffix" | "sequence"
SELFCHECK_PROBE='\nSelf-check: from here I should'
SELFCHECK_POS=' continue'
SELFCHECK_NEG=' restart'
SELFCHECK_SPLIT='\n\n'
SELFCHECK_TAU=0.0            # 相对于 per-rollout b_i 校准后的 log-ratio
SELFCHECK_BASELINE_MODE=per_rollout  # 每条 rollout 用题干+chat opener 位置算
SELFCHECK_MASK_SCOPE=suffix  # 探针判负后只 mask 该段之后的 token loss

RUN_NAME="opd-skywork7b-to-r1qwen1.5b-selfcheck-gate-reverse-260827"

mkdir -p /group/40092/howu/RL-main/logs
export WANDB_INIT_TIMEOUT=300

cd /group/40092/howu/RL-main
python examples/run_distillation_math.py \
      --config examples/configs/distillation_math.yaml \
      policy.model_name=/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/ \
      teacher.model_name=/dev/shm/llms/Skywork-OR1-Math-7B/ \
      cluster.gpus_per_node=8 \
      policy.train_micro_batch_size=1 \
      policy.train_global_batch_size=128 \
      policy.optimizer.kwargs.lr=1.0e-6 \
      policy.max_total_sequence_length=32768 \
      teacher.max_total_sequence_length=32768 \
      teacher.logprob_batch_size=1 \
      data.dataset_name=DAPOMath17KProcessed \
      distillation.num_generations_per_prompt=1 \
      distillation.max_num_epochs=1 \
      distillation.max_num_steps=106 \
      distillation.val_period=10 \
      distillation.val_at_start=true \
      distillation.max_val_samples=500 \
      data.validation_source=train_holdout \
      data.validation_num_samples=500 \
      data.validation_seed=42 \
      checkpointing.save_period=10 \
      checkpointing.keep_top_k=10 \
      checkpointing.save_consolidated=false \
      checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
      loss_fn.kl_type=reverse \
      loss_fn.zero_outside_topk=false \
      +data.config_name=en \
      logger.wandb_enabled=true \
      logger.wandb.name="${RUN_NAME}"
      # TODO(selfcheck-gate): 待 loss_fn 侧接入后追加:
      # +loss_fn.teacher_selfcheck_gate.enabled=true \
      # +loss_fn.teacher_selfcheck_gate.probe_template="${SELFCHECK_PROBE}" \
      # +loss_fn.teacher_selfcheck_gate.positive_token="${SELFCHECK_POS}" \
      # +loss_fn.teacher_selfcheck_gate.negative_token="${SELFCHECK_NEG}" \
      # +loss_fn.teacher_selfcheck_gate.split_pattern="${SELFCHECK_SPLIT}" \
      # +loss_fn.teacher_selfcheck_gate.threshold=${SELFCHECK_TAU} \
      # +loss_fn.teacher_selfcheck_gate.baseline_logratio=${SELFCHECK_BASELINE} \
      # +loss_fn.teacher_selfcheck_gate.mask_scope=${SELFCHECK_MASK_SCOPE}
