#!/usr/bin/env bash
# OPD 蒸馏：Skywork-OR1-Math-7B (teacher) → DeepSeek-R1-Distill-Qwen-1.5B (student)
# 参考 train_opd.sh，改动：
#   - 换模型（教师 7B / 学生 1.5B，均在 /dev/shm 内存盘）
#   - 从训练集固定抽取并剔除 1000 道题作为验证集（seed=42，train/val 严格互斥）
#   - 训练 1 个 epoch（剩余 39315 样本，drop_last 后 307 步）
#   - 训练前、每 25 步及最后一步测评；每 25 步及最后一步保存权重
#   - checkpoint 保存超时放宽到 30 分钟（默认 NCCL 集合通信 10 分钟，存权重容易超时）
#   - save_consolidated=false：nemo_automodel 的 consolidate_safetensors_files_on_every_rank
#     里 all_gather_object 会误算尺寸、试图分配 1024GiB 直接 OOM（当时还有 79GiB 空闲），
#     是库 bug 而非真实显存不足。改为只存 DCP 分片，训练结束后手工转 HF：
#       python examples/converters/convert_dcp_to_hf.py \
#         --config <ckpt>/config.yaml \
#         --dcp-ckpt-path <ckpt>/policy/weights --hf-ckpt-path <out>
#   - 忽略师生词表一致性检查（两者 tokenizer 完全相同，仅 config.vocab_size 的 padding 不同：
#     teacher 152064 vs student 151936；实测 teacher top-64 从不落到 >=151936 的 padding 区，
#     padding 行 logit 最大 0.637 而真实 token 达 19.25，故安全）
#   - 通过 loss_fn.mask_eos_positions=[151643] 把 EOS 目标位置从 loss 里剔除，
#     防止"教师在错误 rollout 尾部推翻推理 → 学生学会永不停止 → 序列爆炸"的现象。
#     其他一切保持原始 OPD（无 TVD gate、无其他改动）。
set -euo pipefail

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
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ====== 忽略师生词表一致性检查 ======
export NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true

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
TEACHER_MODEL="/dev/shm/llms/Skywork-OR1-Math-7B/"
POLICY_MODEL="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
for d in "$TEACHER_MODEL" "$POLICY_MODEL"; do
  if [[ ! -f "${d}config.json" ]]; then
    echo "❌ 模型缺失: ${d}config.json 不存在"; exit 1
  fi
done
echo "▶ Using policy=$POLICY_MODEL  teacher=$TEACHER_MODEL"

# 修改：更新实验名，避免自动加载原来 3 epoch 实验的 checkpoint
RUN_NAME="opd-skywork7b-to-r1qwen1.5b-mask-eos-lr1e-6-1epoch-holdout1k-seed42"

mkdir -p /group/40092/howu/RL-main/logs

# ====== 从 loss 里 mask 掉 EOS 目标位置 ======
# DeepSeek-R1-Distill-Qwen-1.5B 的 EOS token id = 151643 (<｜end▁of▁sentence｜>)。
# 动机：错误答案 rollout 里学生在 T 位置输出 EOS 提前停下，但教师（强推理器）
# 在这个上下文更倾向于继续写 "wait/but/however"，KL 会去压低 p_S(EOS) → 学生
# 越来越不敢停 → 序列长到 max_length 也答不出答案。这个开关把这些位置从
# loss 里剔除，让 EOS/continuation 决策不进入蒸馏梯度。
# 如需关闭对照，把这一整行注释掉即可（不传 mask_eos_positions ⇒ 走原始 OPD）。
EOS_MASK_ARG="+loss_fn.mask_eos_positions=[151643]"

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
    teacher.logprob_batch_size=1 \
    distillation.max_num_epochs=1 \
    distillation.max_num_steps=307 \
    distillation.val_period=25 \
    distillation.val_at_start=true \
    distillation.max_val_samples=1000 \
    data.validation_source=train_holdout \
    data.validation_num_samples=1000 \
    data.validation_seed=42 \
    checkpointing.save_period=25 \
    checkpointing.save_consolidated=false \
    checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
    ${EOS_MASK_ARG} \
    logger.wandb_enabled=true \
    logger.wandb.name="${RUN_NAME}" || true

python test_gpu.py
