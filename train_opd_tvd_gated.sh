#!/usr/bin/env bash
# TVD-Gated KL Distillation 训练脚本.
# See BASIC_OAD_PROPOSAL.md and inline comments in
# examples/configs/distillation_math_tvd_gated.yaml.
#
# 当前配置：fixed threshold, τ=0.3
#
# 含义：
#   - Loss 主体是 KL（继承自 distillation_math.yaml 的 mixed KL）
#   - 每个 token 额外算 TVD_topk = 1 - Σ min(p_S, p_T) on teacher top-k
#   - 只有 TVD_topk > τ 的 token 参与 loss（差异大才学，strict >）
#   - τ = 0 → 保留 TVD_topk > 0 的 token（严格意义上会漏掉 student==teacher 的
#            极端一致 token；实际训练中稀有，接近 baseline 但不等价；
#            想要真正的 baseline 请用 MODE=none）
#   - τ = 1 → 什么都不学（sanity 下界，因 TVD ≤ 1 且严格 >）
#   - 中间 → 越大越激进
#
# 切换实验时修改下面 MODE / THRESHOLD 或 START / END / UNTIL。
set -euo pipefail

# 环境与代理（与 train_opd.sh 保持一致）
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

export WANDB_MODE=disabled

# ====== Attention 后端（flash-attn 可用时注释掉下面这行） ======
# export VLLM_ATTENTION_BACKEND=XFORMERS

# ====== 让所有 Ray worker 使用当前 conda 环境的 Python，彻底绕过 uv ======
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# 清理旧的 venv 构建残留（避免 STARTED_ENV_BUILDER 死锁）
rm -rf /group/40143/howu/RL-main/venvs

# 停止残留的 Ray 进程（避免旧 worker 缓存问题）
ray stop --force 2>/dev/null || true

sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' /group/40143/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py
# ====== 结束 ======

export PYTHONPATH=/group/40143/howu/RL-main:$PYTHONPATH

# ====== TVD gate 实验参数 ======
# MODE       ∈ {none, fixed, warmup}
# THRESHOLD  仅在 MODE=fixed 时生效；τ ∈ [0, 1]，越大越激进（只学分歧最大的）
# START      仅在 MODE=warmup 时生效；step 0 时的 τ
# END        仅在 MODE=warmup 时生效；warmup 结束后的 τ（一直保持到训练末尾）
# UNTIL      仅在 MODE=warmup 时生效；warmup 结束的 step fraction
MODE="fixed"
THRESHOLD=0.8
START=0.8
END=0.1
UNTIL=0.3

echo "▶ Running TVD-gated KL: MODE=${MODE} THRESHOLD=${THRESHOLD} START=${START} END=${END} UNTIL=${UNTIL}"

cd /group/40143/howu/RL-main && python examples/run_distillation_math.py \
    --config examples/configs/distillation_math_tvd_gated.yaml \
    policy.model_name="/group/40143/howu/llms/Qwen3-1.7B/" \
    teacher.model_name="/group/40143/howu/llms/Qwen3-4B/" \
    cluster.gpus_per_node=8 \
    policy.train_micro_batch_size=1 \
    teacher.logprob_batch_size=2 \
    distillation.max_num_epochs=3 \
    checkpointing.save_consolidated=true \
    loss_fn.tvd_gate.mode="${MODE}" \
    loss_fn.tvd_gate.threshold="${THRESHOLD}" \
    loss_fn.tvd_gate.start_threshold="${START}" \
    loss_fn.tvd_gate.end_threshold="${END}" \
    loss_fn.tvd_gate.warmup_until_frac="${UNTIL}"
