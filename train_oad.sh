#!/usr/bin/env bash
# Run Overlap-Aligned Distillation (OAD, Path B).
#
# OAD replaces the default KL distillation loss with the overlap-based
# acceptance loss described in BASIC_OAD_PROPOSAL.md. The only thing that
# differs from train_opd.sh is the addition of `loss_fn.type=oad`.
#
# Backend support: DTensor only. Megatron backend is not supported in this
# round (see §五范围声明 in BASIC_OAD_PROPOSAL.md). The current config uses
# DTensor, so this script works as-is.

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
rm -rf /group/40143/howu/RL-main/venvs

# 停止残留的 Ray 进程（避免旧 worker 缓存问题）
ray stop --force 2>/dev/null || true

sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' /group/40143/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py
# ====== 结束 ======

# 让 Python 能找到 nemo_rl 包（项目本身就是 nemo_rl/ 在仓库根目录）
export PYTHONPATH=/group/40143/howu/RL-main:$PYTHONPATH

cd /group/40143/howu/RL-main && python examples/run_distillation_math.py \
      loss_fn.type=oad \
      policy.model_name="/group/40143/howu/llms/Qwen3-1.7B/" \
      teacher.model_name="/group/40143/howu/llms/Qwen3-4B/" \
      cluster.gpus_per_node=8 \
      policy.train_micro_batch_size=1 \
      teacher.logprob_batch_size=2 \
      checkpointing.save_consolidated=true
