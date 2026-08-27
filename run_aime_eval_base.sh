#!/usr/bin/env bash
# AIME-2024 avg@32 评测：OPD 的两个 base model（student 起点 + teacher）。
# 与 run_aime_eval_batch.sh 同尺度（max_model_len=16384, T=0.6, top_p=0.95, avg@32, 8 卡 DP），
# 分数写入同一个 aime_eval_results 目录，可与各 checkpoint 并列对比。
# 用法: bash run_aime_eval_base.sh
set -euo pipefail

source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd

export HF_HOME=/root/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=WARNING

REPO=/group/40092/howu/RL-main
RESULTS=$REPO/aime_eval_results
mkdir -p "$RESULTS"

# 待评测 base 模型: <短名>:<模型目录>
# base- 前缀便于在汇总表里与 OPD checkpoint 区分
MODELS=(
  "base-r1qwen1.5b-student:/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
  "base-skywork7b-teacher:/dev/shm/llms/Skywork-OR1-Math-7B/"
)

cd "$REPO"

for entry in "${MODELS[@]}"; do
  tag="${entry%%:*}"
  model="${entry#*:}"
  out="$RESULTS/aime2024_${tag}_avg32.jsonl"
  acc_file="$RESULTS/acc_${tag}.txt"

  if [[ -s "$acc_file" ]]; then
    echo "[skip] $tag already scored: $(cat "$acc_file")"
    continue
  fi
  if [[ ! -f "${model}config.json" ]]; then
    echo "[warn] missing model, skipping: $model" >&2
    continue
  fi

  echo "=============================================================="
  echo "[$(date '+%F %T')] $tag  <- $model"
  echo "=============================================================="

  # base model 已是 HF 格式，无需 DCP 转换，直接 8 卡数据并行评测
  python vllm_opd_aime.py \
    --model-path "$model" \
    --test-file ./aime_2024.jsonl \
    --output-path "$out" \
    --data-parallel-size 8 \
    2>&1 | tee "$RESULTS/log_${tag}.txt" | grep -E "^\[eval\]|^acc:" || true

  grep -E "^acc:" "$RESULTS/log_${tag}.txt" | tail -1 > "$acc_file" || true
  echo "[done] $tag $(cat "$acc_file" 2>/dev/null)"
done

echo
echo "================== ALL RESULTS =================="
python "$REPO/tools/summarize_aime_results.py" --results-dir "$RESULTS"
