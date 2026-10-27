#!/usr/bin/env bash
# PolarOPD E0 零训练诊断：student 采样 → student/teacher 逐位置打分 → 汇总报告。
# 默认 teacher=Skywork-OR1-Math-7B，student=R1-Distill-Qwen-1.5B。
# 换 teacher / student 或规模只需覆盖环境变量，例如：
#   TEACHER=/dev/shm/llms/justrl2_step_900/ TAG=justrl2step900-to-r1qwen15b \
#   bash research/polar_opd_e0/run_e0.sh
# 复用另一组已有的学生采样（必须是同一个 STUDENT 生成的，会跳过 gen）：
#   TEACHER=/dev/shm/llms/justrl2_step_900/ TAG=justrl2step900-to-r1qwen15b \
#   GEN_FROM=/group/40092/howu/RL-main/logs/polar_e0/skywork7b-to-r1qwen15b \
#   bash research/polar_opd_e0/run_e0.sh
# 各阶段可单独重跑：STAGES="score report" bash research/polar_opd_e0/run_e0.sh
set -euo pipefail

source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd

: "${ENV_VENUS_PROXY:=http://star-proxy.oa.com:3128}"
export NO_PROXY=localhost,127.0.0.1,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY=$ENV_VENUS_PROXY HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY http_proxy=$ENV_VENUS_PROXY https_proxy=$ENV_VENUS_PROXY
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

REPO=/group/40092/howu/RL-main
TEACHER="${TEACHER:-/dev/shm/llms/Skywork-OR1-Math-7B/}"
STUDENT="${STUDENT:-/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/}"
TAG="${TAG:-skywork7b-to-r1qwen15b}"
NUM_GPUS="${NUM_GPUS:-8}"
N_PROBLEMS="${N_PROBLEMS:-256}"
N_SAMPLES="${N_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"  # = 训练时的 max_total_sequence_length
TOPK="${TOPK:-64}"  # = distillation.topk_logits_k
STAGES="${STAGES:-gen score report}"
GEN_FROM="${GEN_FROM:-}"
OUT="$REPO/logs/polar_e0/$TAG"
GEN_DIR="${GEN_FROM:-$OUT}"
if [[ -n "$GEN_FROM" ]]; then
  ls "$GEN_FROM"/gen_shard*.jsonl > /dev/null 2>&1 || { echo "❌ $GEN_FROM 下没有 gen_shard*.jsonl"; exit 1; }
  [[ " $STAGES " == *" gen "* ]] && echo "ℹ 设置了 GEN_FROM，跳过 gen"
  STAGES=" $STAGES "
  STAGES="${STAGES// gen / }"
fi

for d in "$TEACHER" "$STUDENT"; do
  [[ -f "${d}config.json" ]] || { echo "❌ 模型缺失: ${d}config.json"; exit 1; }
done
mkdir -p "$OUT"
cd "$REPO"
echo "▶ student=$STUDENT teacher=$TEACHER samples=$GEN_DIR out=$OUT"

run_sharded() {  # $1 = stage, rest = extra args; one process per GPU, fail if any shard fails
  local stage=$1; shift
  local pids=()
  for ((i = 0; i < NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python research/polar_opd_e0/e0_diag.py "$stage" \
      --out_dir "$OUT" --student "$STUDENT" --teacher "$TEACHER" \
      --shard "$i" --num_shards "$NUM_GPUS" "$@" > "$OUT/${stage}_shard$i.log" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "❌ $stage shard $i 失败，见 $OUT/${stage}_shard$i.log"; fail=1; }
  done
  return $fail
}

if [[ " $STAGES " == *" gen "* ]]; then
  echo "[$(date '+%F %T')] gen"
  run_sharded gen --n_problems "$N_PROBLEMS" --n_samples "$N_SAMPLES" --max_new_tokens "$MAX_NEW_TOKENS"
fi
if [[ " $STAGES " == *" score "* ]]; then
  echo "[$(date '+%F %T')] score"
  run_sharded score --topk "$TOPK" --gen_dir "$GEN_DIR"
fi
if [[ " $STAGES " == *" report "* ]]; then
  echo "[$(date '+%F %T')] report"
  python research/polar_opd_e0/e0_diag.py report --out_dir "$OUT" | tee "$OUT/report.log"
fi
