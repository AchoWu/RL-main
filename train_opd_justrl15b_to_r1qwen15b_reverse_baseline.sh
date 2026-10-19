#!/usr/bin/env bash
# ============================================================================
# OPD Baseline — Reverse KL (原始 OPD loss)
# Teacher: hbx/JustRL-DeepSeek-1.5B
# Student: DeepSeek-R1-Distill-Qwen-1.5B
# ============================================================================
# 用途：跟 train_opd_justrl15b_to_r1qwen15b_topkagreement_ce.sh 严格对照，
# 除了 loss 换成原始 OPD reverse-KL 之外其他全部一样，用来量化
# topk_agreement_ce 相对 reverse-KL baseline 的增量/减量。
#
# Loss (reverse KL on teacher top-k support)
#     L_t = Σ_{v ∈ T_top} q_S(v) · (log q_S(v) − log p_T(v))
# 其中 q_S / p_T 都是在 teacher top-k 上重归一化后的分布。
# 完全等价于 examples/configs/distillation_math.yaml 的默认
# loss_fn.type=kl + kl_type=reverse 组合。
#
# 变更点（相比 train_opd_justrl15b_to_r1qwen15b_topkagreement_ce.sh）
#   - loss_fn.type=kl (default), loss_fn.kl_type=reverse
#   - 去掉所有 topk_agreement_ce 参数
#   - RUN_NAME 换成 reverse-kl 前缀
# 其余（teacher/student/lr/batch/seq_len/steps/val_period/topk_k/数据集）不变。
# ============================================================================
set -euo pipefail
trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python test_gpu.py
' EXIT INT TERM

# ====== conda 环境（用 opd，env-3.12 已被清空） ======
source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd
echo "▶ Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# ====== 代理（HF / wandb 都需要，容器内默认没有代理变量） ======
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

# ====== 保存 checkpoint 超时 = 30 分钟 ======
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800

# ====== WandB 在线上传 ======
export WANDB_MODE=online
export WANDB_PROJECT=nemo-distillation

# ====== 让所有 Ray worker 使用当前 conda 环境的 Python，彻底绕过 uv ======
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# 清理旧的 venv 构建残留
rm -rf /group/40092/howu/RL-main/venvs

# 停止残留的 Ray 进程
ray stop --force 2>/dev/null || true

sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' /group/40092/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py

export PYTHONPATH=/group/40092/howu/RL-main:${PYTHONPATH:-}

# ====== 模型路径（内存盘） ======
TEACHER_MODEL="/dev/shm/llms/JustRL-DeepSeek-1.5B/"
POLICY_MODEL="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
for d in "$TEACHER_MODEL" "$POLICY_MODEL"; do
  if [[ ! -f "${d}config.json" ]]; then
    echo "❌ 模型缺失: ${d}config.json 不存在"; exit 1
  fi
done
echo "▶ Using policy=$POLICY_MODEL  teacher=$TEACHER_MODEL"

# 实验名
RUN_NAME="opd-justrl15b-to-r1qwen1.5b-reverse-kl-baseline"

mkdir -p /group/40092/howu/RL-main/logs

export HTTP_PROXY=http://star-proxy.oa.com:3128
export HTTPS_PROXY=http://star-proxy.oa.com:3128
export WANDB_INIT_TIMEOUT=300

cd /group/40092/howu/RL-main
python examples/run_distillation_math.py \
      --config examples/configs/distillation_math.yaml \
      policy.model_name=/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/ \
      teacher.model_name=/dev/shm/llms/JustRL-DeepSeek-1.5B/ \
      teacher.dtensor_cfg.tensor_parallel_size=2 \
      teacher.dtensor_cfg.context_parallel_size=2 \
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
      distillation.topk_logits_k=32 \
      data.validation_source=train_holdout \
      data.validation_num_samples=500 \
      data.validation_seed=42 \
      checkpointing.save_period=10 \
      checkpointing.keep_top_k=10 \
      checkpointing.save_consolidated=false \
      checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
      loss_fn.type=kl \
      loss_fn.kl_type=reverse \
      loss_fn.zero_outside_topk=false \
      +data.config_name=en \
      logger.wandb_enabled=true \
      logger.wandb.name="${RUN_NAME}"
