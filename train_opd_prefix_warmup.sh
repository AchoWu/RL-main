#!/usr/bin/env bash
# Prefix-Length Warmup 训练脚本 (Idea 3).
# See opd-improvements-proposal.md.
#
# 用法：
#   bash train_opd_prefix_warmup.sh                     # 默认 cosine warmup (start=0.1, until=0.4)
#   MODE=stepwise    bash train_opd_prefix_warmup.sh    # 用 yaml 里的 stepwise schedule
#   MODE=fixed RATIO=0.5  bash train_opd_prefix_warmup.sh   # E-A1
#   MODE=fixed RATIO=0.25 bash train_opd_prefix_warmup.sh   # E-A2
#   MODE=none        bash train_opd_prefix_warmup.sh    # 基线对照 (等价 E0)
#   START=0.25 UNTIL=0.3 bash train_opd_prefix_warmup.sh # 覆盖 cosine 参数
set -euo pipefail

# 环境与代理（与 train_opd.sh 保持一致）
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

export WANDB_MODE=disabled

# ====== Attention 后端（flash-attn 可用时注释掉下面这行） ======
# export VLLM_ATTENTION_BACKEND=XFORMERS

# ====== 让所有 Ray worker 使用当前 conda 环境的 Python，彻底绕过 uv ======
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# 清理旧的 venv 构建残留（避免 STARTED_ENV_BUILDER 死锁）
rm -rf /group/40143/howu/nemo-rl/venvs

# 停止残留的 Ray 进程（避免旧 worker 缓存问题）
ray stop --force 2>/dev/null || true

sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' /group/40143/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py
# ====== 结束 ======

# ====== Prefix-length warmup 实验切换 ======
# MODE   ∈ {none, fixed, stepwise, cosine}  — 默认 cosine
# RATIO  仅在 MODE=fixed 时使用（E-A1 用 0.5，E-A2 用 0.25）
# START  仅在 MODE=cosine 时使用，起始 prefix_ratio（默认 0.25）
# UNTIL  仅在 MODE=cosine 时使用，warmup 结束的 step fraction（默认 0.3）
MODE="${MODE:-cosine}"
RATIO="${RATIO:-0.5}"
START="${START:-0.1}"
UNTIL="${UNTIL:-0.4}"

echo "▶ Running OPD prefix-length warmup: MODE=${MODE} RATIO=${RATIO} START=${START} UNTIL=${UNTIL}"

python examples/run_distillation_math.py \
    --config examples/configs/distillation_math_prefix_warmup.yaml \
    policy.model_name="/group/40143/howu/llms/Qwen3-1.7B/" \
    teacher.model_name="/group/40143/howu/llms/Qwen3-4B/" \
    cluster.gpus_per_node=8 \
    policy.train_micro_batch_size=1 \
    teacher.logprob_batch_size=2 \
    checkpointing.save_consolidated=true \
    distillation.prefix_length_warmup.mode="${MODE}" \
    distillation.prefix_length_warmup.fixed_prefix_ratio="${RATIO}" \
    distillation.prefix_length_warmup.cosine_start_ratio="${START}" \
    distillation.prefix_length_warmup.cosine_warmup_until_frac="${UNTIL}"
