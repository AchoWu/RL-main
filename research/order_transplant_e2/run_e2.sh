#!/usr/bin/env bash
# 序移植 FKL E2：同步数、同数据，对比 OPD / FKL / 序移植 FKL 的短训练。
# 设置沿用 E1：teacher = Skywork-OR1-Math-7B，student = DeepSeek-R1-Distill-Qwen-1.5B，
# lr 1e-6、batch 128、16k，验证集 AIME24/25/26 × VAL_REPEAT 次采样。
#
# 组别（ARMS 按顺序依次跑，每组占满 8 卡）：
#   opd      reverse KL（E1 基线）
#   fkl      forward KL（只换方向：检验支撑壁垒）
#   tp1      序移植 FKL，β=1.0（teacher 的序 + 初始学生的谱，熵钉在初始学生）
#   tp12     序移植 FKL，β=1.2（恒温器剂量：锐化）
#   tpself   无 teacher 对照：目标 = 初始学生^β 自己（β=TPSELF_BETA，默认 1.2，与 tp12 配对）；
#            只有锐化、没有 teacher 的序。β=1 时目标就是初始学生本身，没有学习信号。
# 跨组可比的量：
#   validation/accuracy(/aime<年>)   主指标；validation/truncation_rate 截断率
#   train/teacher_reverse_kl         对原始 teacher 的 reverse KL（各组 train/loss 含义不同）
#   train/window_entropy             学生在 teacher top-k 窗口内的熵（tp1 预期基本平坦）
#   train/argmax_agree               窗口内学生 argmax == teacher argmax 的比例
#   train/barrier_frac               s(teacher argmax) < e^-5 的位置比例（支撑壁垒）
#
# 用法：
#   bash research/order_transplant_e2/run_e2.sh                          # 五组全跑
#   ARMS="opd tp1" MAX_STEPS=2 VAL_PERIOD=1 PREFIX=e2-smoke bash research/order_transplant_e2/run_e2.sh
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
ARMS="${ARMS:-opd fkl tp1 tp12 tpself}"
TPSELF_BETA="${TPSELF_BETA:-1.2}"
MAX_STEPS="${MAX_STEPS:-100}"
VAL_PERIOD="${VAL_PERIOD:-10}"
VAL_YEARS="${VAL_YEARS:-2024,2025,2026}"
VAL_REPEAT="${VAL_REPEAT:-8}"  # 每题采样次数
# 每年 30 题 × VAL_REPEAT 次采样；需覆盖到最后一个（可能不满的）val batch，val_batch_size=64
N_YEARS=$(( $(tr -cd ',' <<< "$VAL_YEARS" | wc -c) + 1 ))
VAL_SAMPLES="${VAL_SAMPLES:-$(( (N_YEARS * 30 * VAL_REPEAT + 63) / 64 * 64 ))}"
SEQ_LEN="${SEQ_LEN:-16384}"
SEED="${SEED:-42}"
PREFIX="${PREFIX:-e2-skywork7b-to-r1qwen15b}"

for d in "$TEACHER" "$STUDENT"; do
  [[ -f "${d}config.json" ]] || { echo "❌ 模型缺失: ${d}config.json"; exit 1; }
done

arm_overrides() {
  case "$1" in
    opd)    echo "loss_fn.kl_type=reverse" ;;
    fkl)    echo "loss_fn.kl_type=forward" ;;
    tp1)    echo "loss_fn.kl_type=transplant +loss_fn.transplant_beta=1.0" ;;
    tp12)   echo "loss_fn.kl_type=transplant +loss_fn.transplant_beta=1.2" ;;
    tpself) echo "loss_fn.kl_type=transplant +loss_fn.transplant_beta=${TPSELF_BETA} +loss_fn.transplant_order=reference" ;;
    *) echo "❌ 未知组别: $1" >&2; return 1 ;;
  esac
}
# 先校验全部组名，避免跑了几个小时后才在后面的组上失败
for arm in $ARMS; do arm_overrides "$arm" >/dev/null; done

mkdir -p "$REPO/logs/order_transplant_e2"
cd "$REPO"
for arm in $ARMS; do
  RUN_NAME="${PREFIX}-${arm}-s${SEED}-step${MAX_STEPS}"
  LOG="$REPO/logs/order_transplant_e2/${RUN_NAME}.log"
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
      distillation.max_val_samples="$VAL_SAMPLES" \
      +data.val_aime_years="[$VAL_YEARS]" \
      +data.val_repeat="$VAL_REPEAT" \
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
