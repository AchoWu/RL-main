#!/usr/bin/env bash
# TVD-Gated KL Distillation 训练脚本 (τ=0.8 版本).
# 与 train_opd_tvd_gated_tau02.sh 结构保持一致，只有 THRESHOLD / TAU_TAG 不同，
# 便于做 τ-sweep 时的公平对比。
#
# 含义速查：
#   - Loss 主体是 KL（继承自 distillation_math.yaml 的 mixed KL）
#   - 每个 token 额外算 TVD_topk = 1 - Σ min(p_S, p_T) on teacher top-k
#   - 只有 TVD_topk > τ 的 token 参与 loss（差异大才学，strict >）
#   - τ = 0 → 近似 baseline；τ = 1 → 什么都不学；中间越大越激进
set -euo pipefail

# ====== 强制激活 env-3.12（否则 /root/custom.bashrc 会把当前 shell 切回 env-3.6.8） ======
# shellcheck disable=SC1091
source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate env-3.12
echo "▶ Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# 环境与代理（与 train_opd.sh 保持一致）
: "${ENV_VENUS_PROXY:=${http_proxy:-}}"
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# ====== WandB 在线上传 ======
export WANDB_MODE=online
export WANDB_PROJECT=nemo-distillation
# 归到同一 sweep group，方便和 tau02 / 其他 tau 在 wandb UI 上做并排对比
export WANDB_RUN_GROUP=tvd-gated-sweep

# ====== Attention 后端（flash-attn 可用时注释掉下面这行） ======
# export VLLM_ATTENTION_BACKEND=XFORMERS

# ====== 让所有 Ray worker 使用当前 conda 环境的 Python，彻底绕过 uv ======
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# 清理旧的 venv 构建残留（避免 STARTED_ENV_BUILDER 死锁）
rm -rf /group/40092/howu/RL-main/venvs

# 停止残留的 Ray 进程（避免旧 worker 缓存问题）
ray stop --force 2>/dev/null || true

sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' /group/40092/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py
# ====== 结束 ======

export PYTHONPATH=/group/40092/howu/RL-main:${PYTHONPATH:-}

# ====== TVD gate 实验参数 ======
# 本脚本固定 MODE=fixed，只改 THRESHOLD；TAU_TAG 会跟随自动生成。
MODE="fixed"
THRESHOLD=0.8
START=0.8
END=0.1
UNTIL=0.3

# tau 标签（用于 wandb name / checkpoint 路径），避免路径里出现小数点
TAU_TAG="tau${THRESHOLD//./}"   # 0.8 -> tau08

echo "▶ Running TVD-gated KL: MODE=${MODE} THRESHOLD=${THRESHOLD} (${TAU_TAG}) START=${START} END=${END} UNTIL=${UNTIL}"

# ====== 用 /dev/shm/llms 上的内存盘缓存（提前拷贝好），避免走网盘 IO 加载慢 ======
POLICY_MODEL="/dev/shm/llms/Qwen3-1.7B/"
TEACHER_MODEL="/dev/shm/llms/Qwen3-4B/"
if [[ ! -d "$POLICY_MODEL" || ! -d "$TEACHER_MODEL" ]]; then
  echo "⚠️  /dev/shm 缓存缺失，回退到 /group/40092 网盘（加载会慢 ~20 分钟）"
  POLICY_MODEL="/group/40092/howu/llms/Qwen3-1.7B/"
  TEACHER_MODEL="/group/40092/howu/llms/Qwen3-4B/"
fi
echo "▶ Using policy=$POLICY_MODEL  teacher=$TEACHER_MODEL"

cd /group/40092/howu/RL-main && python examples/run_distillation_math.py \
    --config examples/configs/distillation_math_tvd_gated.yaml \
    policy.model_name="$POLICY_MODEL" \
    teacher.model_name="$TEACHER_MODEL" \
    cluster.gpus_per_node=8 \
    policy.train_micro_batch_size=4 \
    teacher.logprob_batch_size=4 \
    distillation.max_num_epochs=3 \
    checkpointing.save_consolidated=true \
    checkpointing.checkpoint_dir="checkpoints/distillation-tvd-gated-${MODE}-${TAU_TAG}-qwen3-1.7B" \
    loss_fn.tvd_gate.mode="${MODE}" \
    loss_fn.tvd_gate.threshold="${THRESHOLD}" \
    loss_fn.tvd_gate.start_threshold="${START}" \
    loss_fn.tvd_gate.end_threshold="${END}" \
    loss_fn.tvd_gate.warmup_until_frac="${UNTIL}" \
    logger.wandb.name="distillation-tvd-gated-${MODE}-${TAU_TAG}-qwen3-1.7B" \
    +logger.wandb.group="tvd-gated-sweep" \
    +logger.wandb.tags="[tvd_gate,${MODE},${TAU_TAG}]"

# 占卡
python test_gpu.py