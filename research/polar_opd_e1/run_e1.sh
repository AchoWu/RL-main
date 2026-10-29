#!/usr/bin/env bash
# PolarOPD E1：同步数、同数据，对比 OPD 与 PolarOPD 的短训练。
# teacher = Skywork-OR1-Math-7B，student = DeepSeek-R1-Distill-Qwen-1.5B（E0 中对 PolarOPD 最有利的设定）。
#
# 组别（ARMS 按顺序依次跑，每组占满 8 卡）：
#   opd      reverse KL（基线）
#   opdT     reverse KL + teacher 温度 TEACHER_T（E0 的 best_global_T，top-64）
#   polar25  PolarOPD λ=0.25
#   polar50  PolarOPD λ=0.5（剂量对照）
# 跨组可比的量：
#   validation/accuracy          主指标
#   train/teacher_reverse_kl     对原始（T=1）teacher 的 reverse KL；各组 train/loss 的含义不同，不可直接比
#
# 用法：
#   bash research/polar_opd_e1/run_e1.sh                        # 四组全跑
#   ARMS="opd polar25" MAX_STEPS=30 bash research/polar_opd_e1/run_e1.sh
set -euo pipefail

trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python /group/40092/howu/RL-main/test_gpu.py
' EXIT INT TERM

# ====== conda 环境 ======
source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd
echo "▶ Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# ====== 代理 / 缓存 ======
export NO_PROXY=localhost,127.0.0.1,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY=http://star-proxy.oa.com:3128
export HTTPS_PROXY=http://star-proxy.oa.com:3128
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
echo "HTTPS_PROXY=$HTTPS_PROXY"
echo "HTTP_PROXY=$HTTP_PROXY"

export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# ====== 超时（同 train_opd.sh） ======
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

export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
REPO=/group/40092/howu/RL-main
sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' \
  "$REPO/nemo_rl/distributed/ray_actor_environment_registry.py"
export PYTHONPATH=$REPO:${PYTHONPATH:-}

# ====== 实验参数 ======
TEACHER="${TEACHER:-/dev/shm/llms/Skywork-OR1-Math-7B/}"
STUDENT="${STUDENT:-/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/}"
ARMS="${ARMS:-opd opdT polar25 polar50}"
TEACHER_T="${TEACHER_T:-1.41}"
MAX_STEPS="${MAX_STEPS:-200}"
VAL_PERIOD="${VAL_PERIOD:-5}"
SEQ_LEN="${SEQ_LEN:-16384}"
SEED="${SEED:-42}"
PREFIX="${PREFIX:-e1-skywork7b-to-r1qwen15b}"

for d in "$TEACHER" "$STUDENT"; do
  [[ -f "${d}config.json" ]] || { echo "❌ 模型缺失: ${d}config.json"; exit 1; }
done

arm_overrides() {
  case "$1" in
    opd)     echo "loss_fn.kl_type=reverse" ;;
    opdT)    echo "loss_fn.kl_type=reverse +loss_fn.teacher_temperature=${TEACHER_T}" ;;
    polar25) echo "loss_fn.kl_type=polar +loss_fn.polar_lambda=0.25" ;;
    polar50) echo "loss_fn.kl_type=polar +loss_fn.polar_lambda=0.5" ;;
    *) echo "❌ 未知组别: $1" >&2; return 1 ;;
  esac
}
# 先校验全部组名，避免跑了几个小时后才在后面的组上失败
for arm in $ARMS; do arm_overrides "$arm" >/dev/null; done

mkdir -p "$REPO/logs/polar_e1"
cd "$REPO"
for arm in $ARMS; do
  RUN_NAME="${PREFIX}-${arm}-s${SEED}-step${MAX_STEPS}"
  LOG="$REPO/logs/polar_e1/${RUN_NAME}.log"
  echo "[$(date '+%F %T')] ▶ ${arm}: $(arm_overrides "$arm")  log=$LOG"
  rm -rf "$REPO/venvs"
  ray stop --force 2>/dev/null || true
  # shellcheck disable=SC2046  # arm_overrides 的输出需要按空格拆成多个参数
  if ! python examples/run_distillation_math.py \
      --config examples/configs/distillation_math.yaml \
      policy.model_name="$STUDENT" \
      teacher.model_name="$TEACHER" \
      cluster.gpus_per_node=8 \
      policy.train_micro_batch_size=1 \
      policy.train_global_batch_size=128 \
      policy.optimizer.kwargs.lr=1.0e-6 \
      policy.max_total_sequence_length="$SEQ_LEN" \
      teacher.max_total_sequence_length="$SEQ_LEN" \
      teacher.logprob_batch_size=1 \
      teacher.dtensor_cfg.tensor_parallel_size=2 \
      teacher.dtensor_cfg.context_parallel_size=2 \
      distillation.seed="$SEED" \
      distillation.max_num_epochs=1 \
      distillation.max_num_steps="$MAX_STEPS" \
      distillation.val_period="$VAL_PERIOD" \
      distillation.val_at_start=true \
      distillation.max_val_samples=1000 \
      checkpointing.save_period="$VAL_PERIOD" \
      checkpointing.keep_top_k=3 \
      checkpointing.save_consolidated=false \
      checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
      logger.wandb_enabled=true \
      logger.wandb.name="$RUN_NAME" \
      $(arm_overrides "$arm") > "$LOG" 2>&1; then
    echo "❌ ${arm} 失败，见 $LOG（继续下一组）"
  fi
done
echo "[$(date '+%F %T')] done: $ARMS"
