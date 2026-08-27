#!/usr/bin/env bash
# AIME-2024 + AIME-2025 + MATH-500 + Omni-MATH-Rule 评测：dapo32k-seqmean-refkl 的 step_90。
# 口径与已有 step_106 评测完全一致: max_model_len=32768, 8 卡 DP;
# AIME 32 采样, MATH-500 / Omni-MATH 4 采样。
#
# 用法: bash run_aime_eval_seqmean_refkl_step90.sh
set -euo pipefail

source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd

export HF_HOME=/root/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=WARNING

REPO=/group/40092/howu/RL-main
CKPT_ROOT=$REPO/checkpoints
STUDENT=/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/
STAGE=/dev/shm/hf_ckpts
RESULTS=$REPO/aime_eval_results_dapo32k
MAX_LEN=32768
DP=8
STEP=step_90

# 短名与 step_106 表里的命名对齐, 仅 step 段不同
short="dapo32k-seqmean-refkl-step90"
src="$CKPT_ROOT/distillation-opd-skywork7b-to-r1qwen1.5b-reverse-dapo32k-oldkl-sequence-mean-refkl-b0.01-step106-dapo32k-260824/$STEP"

mkdir -p "$STAGE" "$RESULTS"
cd "$REPO"

[[ -d "$src" ]] || { echo "missing checkpoint: $src"; exit 1; }

# 基准: <名>:<文件>:<采样数>
BENCHMARKS=(
  "aime2024:./aime_2024.jsonl:32"
  "aime2025:eval_datasets/aime_2025.jsonl:32"
  "math500:eval_datasets/math_500.jsonl:4"
  "omni-math-rule:eval_datasets/omni_math_rule.jsonl:4"
)

hf="$STAGE/dapo32k-eval-$short"
if [[ ! -f "$hf/model-00001-of-00001.safetensors" ]]; then
  echo "[convert] $src -> $hf"
  python "$REPO/tools/prepare_hf_ckpt_for_eval.py" \
    --ckpt-step-dir "$src" --out-dir "$hf" --tokenizer "$STUDENT"
fi

for b in "${BENCHMARKS[@]}"; do
  bname="${b%%:*}"
  rest="${b#*:}"
  bfile="${rest%%:*}"
  nsamp="${rest##*:}"
  out="$RESULTS/${bname}_${short}.jsonl"
  acc_file="$RESULTS/acc_${bname}_${short}.txt"

  [[ -s "$acc_file" ]] && { echo "[skip] $bname/$short 已有分数"; continue; }

  echo "=============================================================="
  echo "[$(date '+%F %T')] bench=$bname model=$short  n_sample=$nsamp  max_len=$MAX_LEN  dp=$DP"
  echo "=============================================================="

  python vllm_opd_aime.py \
    --model-path "$hf" \
    --test-file "$bfile" \
    --output-path "$out" \
    --max-model-len "$MAX_LEN" \
    --num-generation "$nsamp" \
    --data-parallel-size "$DP" \
    2>&1 | tee "$RESULTS/log_${bname}_${short}.txt" | grep -E "^\[eval\]|^acc:" || true

  grep -E "^acc:" "$RESULTS/log_${bname}_${short}.txt" | tail -1 > "$acc_file" || true
  echo "[done] $bname/$short $(cat "$acc_file" 2>/dev/null)"
done

rm -rf "$hf"   # 释放 /dev/shm

echo
echo "================== step_90 各基准 avg@N =================="
for b in aime2024 aime2025 math500 omni-math-rule; do
  f="$RESULTS/acc_${b}_${short}.txt"
  [[ -s "$f" ]] && echo "  $b: $(cat "$f")"
done
