#!/usr/bin/env bash
# Matched full-token baseline:
# Skywork-OR1-Math-7B -> DeepSeek-R1-Distill-Qwen-1.5B
#
# 与 TVD warmup 实验保持一致：
#   1. reverse KL
#   2. zero_outside_topk=true
#   3. 相同模型、数据、batch size、学习率、验证和保存配置
#
# 唯一区别：
#   从第 0 步开始固定 threshold=1.0，不进行 TVD warmup。
#   direction=low 时保留 TVD < 1.0 的 token，近似覆盖全部 token。

set -euo pipefail

# ====== conda 环境 ======
source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd
echo "Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# ====== 代理 ======
: "${ENV_VENUS_PROXY:=http://star-proxy.oa.com:3128}"
export NO_PROXY=localhost,127.0.0.1,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY="$ENV_VENUS_PROXY"
export HTTPS_PROXY="$ENV_VENUS_PROXY"
export no_proxy="$NO_PROXY"
export http_proxy="$ENV_VENUS_PROXY"
export https_proxy="$ENV_VENUS_PROXY"

# ====== HF 缓存 ======
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# ====== 忽略师生词表一致性检查 ======
export NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true

# ====== 保存 checkpoint 超时：30 分钟 ======
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800

# ====== WandB ======
export WANDB_MODE=online
export WANDB_PROJECT=nemo-distillation
export WANDB_INIT_TIMEOUT=300

# ====== Ray worker 使用当前 conda 环境 ======
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

REPO_DIR=/group/40092/howu/RL-main

rm -rf "${REPO_DIR}/venvs"
ray stop --force 2>/dev/null || true

sed -i \
  's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' \
  "${REPO_DIR}/nemo_rl/distributed/ray_actor_environment_registry.py"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# ====== 模型路径 ======
TEACHER_MODEL="/dev/shm/llms/Skywork-OR1-Math-7B/"
POLICY_MODEL="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"

for model_dir in "$TEACHER_MODEL" "$POLICY_MODEL"; do
  if [[ ! -f "${model_dir}config.json" ]]; then
    echo "模型缺失：${model_dir}config.json 不存在"
    exit 1
  fi
done

echo "Using policy=$POLICY_MODEL"
echo "Using teacher=$TEACHER_MODEL"

# ====== Matched full-token 参数 ======
MODE="fixed"
DIRECTION="low"
THRESHOLD=1.0
DATESTR="$(date +%y%m%d)"

RUN_NAME="opd-skywork7b-to-r1qwen1.5b-reverse-tvdmatched-full-low-th${THRESHOLD}-step50-16k-${DATESTR}"

mkdir -p "${REPO_DIR}/logs"

cd "$REPO_DIR"

python examples/run_distillation_math.py \
    --config examples/configs/distillation_math_tvd_warmup_low.yaml \
    policy.model_name="$POLICY_MODEL" \
    teacher.model_name="$TEACHER_MODEL" \
    cluster.gpus_per_node=8 \
    policy.train_micro_batch_size=1 \
    policy.train_global_batch_size=128 \
    policy.optimizer.kwargs.lr=1.0e-6 \
    policy.max_total_sequence_length=16384 \
    teacher.max_total_sequence_length=16384 \
    teacher.logprob_batch_size=1 \
    distillation.max_num_epochs=1 \
    distillation.max_num_steps=50 \
    distillation.val_period=5 \
    distillation.val_at_start=true \
    distillation.max_val_samples=1000 \
    data.validation_source=train_holdout \
    data.validation_num_samples=1000 \
    data.validation_seed=42 \
    checkpointing.save_period=5 \
    checkpointing.keep_top_k=10 \
    checkpointing.save_consolidated=false \
    checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
    loss_fn.kl_type=reverse \
    loss_fn.zero_outside_topk=true \
    loss_fn.tvd_gate.mode="${MODE}" \
    loss_fn.tvd_gate.direction="${DIRECTION}" \
    +loss_fn.tvd_gate.threshold="${THRESHOLD}" \
    logger.wandb_enabled=true \
    logger.wandb.name="${RUN_NAME}"

python3 test_gpu.py