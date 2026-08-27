#!/usr/bin/env bash
# TVD-gated KL 蒸馏（fixed τ, 固定阈值 curriculum 切片）:
#   Skywork-OR1-Math-7B → DeepSeek-R1-Distill-Qwen-1.5B
#
# 语义：direction="low" ⇒ keep iff TVD_topk < τ；mode="fixed" ⇒ 全程恒定 τ=0.4，
# 即整个训练只让"师生已经比较接近（TVD<0.4）"的 token 进入 KL 损失。
#
# 相比 warmup-low 版本，这里 τ 不再随步数变化，用来做单点消融/对照。
# 其他一切（模型/数据/超时/tokenizer 检查/EOS mask 关闭）与
# train_opd_skywork7b_to_r1qwen15b_tvdwarmup_low.sh 对齐。
# set -euo pipefail

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

# ====== 保存 checkpoint 超时 = 30 分钟 ======
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800

# ====== WandB ======
export WANDB_MODE=online
export WANDB_PROJECT=nemo-distillation

# ====== Ray worker 用当前 conda 环境，绕过 uv ======
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
rm -rf /group/40092/howu/RL-main/venvs
ray stop --force 2>/dev/null || true
sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' /group/40092/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py

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

# ====== TVD gated（fixed τ）实验参数 ======
# MODE=fixed:      τ 全程恒定，不随步数变化
# DIRECTION=low:   keep iff TVD_topk < τ（"师生一致"的 token 才进入 loss）
# THRESHOLD=0.4:   固定 τ = 0.4
MODE="fixed"
DIRECTION="low"
THRESHOLD=0.3
RUN_NAME="opd-skywork7b-to-r1qwen1.5b-reverse-tvdfixed-${DIRECTION}-th${THRESHOLD}"
mkdir -p /group/40092/howu/RL-main/logs

# for wandb login
export HTTP_PROXY=http://star-proxy.oa.com:3128
export HTTPS_PROXY=http://star-proxy.oa.com:3128
export WANDB_INIT_TIMEOUT=300

cd /group/40092/howu/RL-main
python examples/run_distillation_math.py \
    --config examples/configs/distillation_math_tvd_gated.yaml \
    policy.model_name="$POLICY_MODEL" \
    teacher.model_name="$TEACHER_MODEL" \
    cluster.gpus_per_node=8 \
    policy.train_micro_batch_size=1 \
    policy.train_global_batch_size=128 \
    policy.optimizer.kwargs.lr=1.0e-6 \
    teacher.logprob_batch_size=1 \
    distillation.max_num_epochs=1 \
    distillation.max_num_steps=100 \
    distillation.val_period=10 \
    distillation.val_at_start=true \
    distillation.max_val_samples=1000 \
    data.validation_source=train_holdout \
    data.validation_num_samples=1000 \
    data.validation_seed=42 \
    checkpointing.save_period=25 \
    checkpointing.save_consolidated=false \
    checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
    loss_fn.kl_type=reverse \
    loss_fn.tvd_gate.mode="${MODE}" \
    +loss_fn.tvd_gate.direction="${DIRECTION}" \
    loss_fn.tvd_gate.threshold="${THRESHOLD}" \
    logger.wandb_enabled=true \
    logger.wandb.name="${RUN_NAME}"
    
python3 test_gpu.py
