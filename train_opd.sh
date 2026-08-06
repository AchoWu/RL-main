export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

# ====== Hugging Face 强制离线（关键） ======
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 明确指定缓存目录（确保所有 Ray worker 共享）
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

export WANDB_MODE=disabled

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

# 让 Python 能找到 nemo_rl 包（项目本身就是 nemo_rl/ 在仓库根目录）
export PYTHONPATH=/group/40092/howu/RL-main:$PYTHONPATH

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
      --config examples/configs/distillation_math.yaml \
      policy.model_name="$POLICY_MODEL" \
      teacher.model_name="$TEACHER_MODEL" \
      cluster.gpus_per_node=8 \
      policy.train_micro_batch_size=1 \
      teacher.logprob_batch_size=2 \
      distillation.max_num_epochs=3 \
      checkpointing.save_consolidated=true

