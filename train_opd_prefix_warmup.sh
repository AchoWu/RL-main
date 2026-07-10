#!/usr/bin/env bash
# Prefix-Length Warmup 训练脚本 (Idea 3).
# See opd-improvements-proposal.md.
#
# 当前配置：cosine warmup, start=0.01, until=0.2
#
# 含义：
#   - 用 S 形余弦曲线控制 prefix_ratio 随训练步数递增。
#   - 起步 (global_step=0) 时 prefix_ratio=0.01 —— 每条 rollout 只有前 1%
#     的 response token 参与 distillation loss，后 99% 被 mask 掉；
#     实际由于 ceil 语义，短序列至少保留 1 个 token。
#   - 当 global_step / max_num_steps == 0.2 时，prefix_ratio=1.0 —— 之后
#     全序列都学（回到基线 OPD 行为）。
#   - 曲线两端导数为 0：起点稳住 0.01，终点稳住 1.0，中间平滑加速；
#     不存在阶梯 mode 里那种边界跳变。
#
# 训练轨迹（max_num_steps=1000, max_num_epochs=3 → 实际 ~942 step）：
#   step   0     : ratio=0.01   起手前 1% token（极激进）
#   step 100     : ratio=0.505  曲线中点
#   step 200     : ratio=1.00   warmup 结束，全序列训练开始
#   step 200-942 : ratio=1.00   最后 ~79% 训练时长在 full length
#
# 切换实验时直接修改下面的 MODE / RATIO / START / UNTIL 变量。
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

# ====== Prefix-length warmup 实验参数 ======
# MODE   实验类型：
#   - "cosine"    S 形余弦 warmup（当前）：由 START / UNTIL 参数决定曲线
#   - "stepwise"  阶梯 schedule：由 yaml 里的 stepwise_schedule 决定
#   - "fixed"     固定 prefix_ratio 不变：由 RATIO 参数决定
#   - "none"      关闭 feature，等价于基线 OPD（loss 覆盖全部 response token）
# RATIO  只在 MODE=fixed 时生效。E-A1 用 0.5，E-A2 用 0.25。
# START  只在 MODE=cosine 时生效。step 0 时的 prefix_ratio。
#          越低 = 起步 warmup 越激进（只学更少 token）
# UNTIL  只在 MODE=cosine 时生效。global_step / max_num_steps 达到该值时
#        ratio 到 1.0；之后一直保持 1.0。越高 = warmup 期越长，full-length
#        训练时间越短。
MODE="cosine"
RATIO=0.5
START=0.01
UNTIL=0.2

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
