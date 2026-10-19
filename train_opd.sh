#!/usr/bin/env bash
# OPD 蒸馏：JustRL2 step-900 (teacher) → DeepSeek-R1-Distill-Qwen-1.5B (student)
# 教师是 verl06/recipe/justrl2/run_justrl2_16k.sh 训练 DeepSeek-R1-Distill-Qwen-1.5B
# RL 模型（global_step_900，已转 HF 格式到 /dev/shm/llms/justrl2_step_900/）。
# 基于 main 的原始 OPD，无任何实验性改动：
#   - 16k 序列长度、reverse KL
#   - 训练 distillation.max_num_steps 步；训练前及每 distillation.val_period 步测评、每 checkpointing.save_period 步保存权重
#   - checkpoint 保存超时放宽到 30 分钟
#   - save_consolidated=false（nemo_automodel consolidate 的 all_gather_object 会误算尺寸 OOM），
#     训练结束后手工转 HF：
#       python examples/converters/convert_dcp_to_hf.py \
#         --config <ckpt>/config.yaml \
#         --dcp-ckpt-path <ckpt>/policy/weights --hf-ckpt-path <out>
set -euo pipefail

trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python /group/40092/howu/RL-main/test_gpu.py
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

# ====== HF 缓存（数据集已预下载到这里；模型走 /dev/shm 本地路径） ======
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
# 数据集已缓存，模型是本地目录，可离线；如需重新下载请注释掉这两行
# export HF_HUB_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

# ====== 保存 checkpoint 超时 = 30 分钟 ======
# 分布式集合通信默认 10 分钟，consolidated 保存 7B/1.5B 权重时 rank 间等待易超时
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
# vLLM rollout 侧超时同样放宽，避免长序列生成被误判超时
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800

# ====== WandB 在线上传 ======
export WANDB_MODE=online
export WANDB_PROJECT=nemo-distillation

# ====== 让所有 Ray worker 使用当前 conda 环境的 Python，彻底绕过 uv ======
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# 清理旧的 venv 构建残留（避免 STARTED_ENV_BUILDER 死锁）
rm -rf /group/40092/howu/RL-main/venvs

# 停止残留的 Ray 进程（避免旧 worker 缓存问题）
ray stop --force 2>/dev/null || true

sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' /group/40092/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py
# ====== 结束 ======

export PYTHONPATH=/group/40092/howu/RL-main:${PYTHONPATH:-}

# ====== 模型路径（内存盘） ======
TEACHER_MODEL="/dev/shm/llms/justrl2_step_900/"
POLICY_MODEL="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
for d in "$TEACHER_MODEL" "$POLICY_MODEL"; do
  if [[ ! -f "${d}config.json" ]]; then
    echo "❌ 模型缺失: ${d}config.json 不存在"; exit 1
  fi
done
echo "▶ Using policy=$POLICY_MODEL  teacher=$TEACHER_MODEL"

# 实验名
RUN_NAME="opd-justrl2step900-to-r1qwen15b-reverse-16k-step50"

mkdir -p /group/40092/howu/RL-main/logs

export HTTP_PROXY=http://star-proxy.oa.com:3128
export HTTPS_PROXY=http://star-proxy.oa.com:3128
export WANDB_INIT_TIMEOUT=300

cd /group/40092/howu/RL-main
python examples/run_distillation_math.py \
    --config examples/configs/distillation_math.yaml \
    policy.model_name="$POLICY_MODEL" \
    teacher.model_name="$TEACHER_MODEL" \
    cluster.gpus_per_node=8 \
    policy.train_micro_batch_size=1 \
    policy.train_global_batch_size=128 \
    policy.optimizer.kwargs.lr=1.0e-6 \
    policy.max_total_sequence_length=16384 \
    teacher.max_total_sequence_length=16384 \
    teacher.logprob_batch_size=1 \
    teacher.dtensor_cfg.tensor_parallel_size=2 \
    teacher.dtensor_cfg.context_parallel_size=2 \
    distillation.max_num_epochs=1 \
    distillation.max_num_steps=50 \
    distillation.val_period=5 \
    distillation.val_at_start=true \
    distillation.max_val_samples=1000 \
    checkpointing.save_period=5 \
    checkpointing.keep_top_k=10 \
    checkpointing.save_consolidated=false \
    checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
    logger.wandb_enabled=true \
    loss_fn.kl_type=reverse \
    logger.wandb.name="${RUN_NAME}" || true

