#!/usr/bin/env bash
# AIME-2024 avg@32 / pass@k 评测：DAPO-32k 训练的 step_50 checkpoint。
#
# 与之前 16k 那批评测的差异：
#   - max_model_len=32768（与 DAPO-32k 训练一致；用 16k 会有 ~48% 样本被截断）
#   - 4 卡数据并行（原来 8 卡）
# 因此分数不能与 aime_eval_results 里那批 16k 的数字直接比较，
# 要对比请用同为 32k 的 base 基线（见文末说明）。
#
# 用法:
#   bash run_aime_eval_dapo32k_step50.sh                 # 两组都跑
#   bash run_aime_eval_dapo32k_step50.sh dapo32k-seqmean # 只跑一组
set -euo pipefail

source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd

export HF_HOME=/root/.cache/huggingface
# 4 卡：如需换卡改这里，并保证张数与 --data-parallel-size 一致
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=WARNING

REPO=/group/40092/howu/RL-main
CKPT_ROOT=$REPO/checkpoints
BASE=/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/
STAGE=/dev/shm/hf_ckpts
RESULTS=$REPO/aime_eval_results_dapo32k
MAX_LEN=32768
DP=4
STEP=step_50

mkdir -p "$STAGE" "$RESULTS"

# <短名>:<checkpoint 目录名>
EXPERIMENTS=(
  "dapo32k-nomask-reverse:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-dapo32k-nomask-reverse-step106-dapo32k-260824"
  "dapo32k-seqmean:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-dapo32k-oldkl-sequence-mean-step106-dapo32k-260824"
)

WANTED=("$@")
cd "$REPO"

for entry in "${EXPERIMENTS[@]}"; do
  short="${entry%%:*}"
  dir="${entry#*:}"

  if [[ ${#WANTED[@]} -gt 0 ]]; then
    match=0
    for w in "${WANTED[@]}"; do
      if [[ "$w" == "$short" ]]; then
        match=1
      fi
    done
    [[ $match -eq 1 ]] || continue
  fi

  tag="${short}-${STEP}-32k"
  src="$CKPT_ROOT/$dir/$STEP"
  hf="$STAGE/$tag"
  out="$RESULTS/aime2024_${tag}_avg32.jsonl"
  acc_file="$RESULTS/acc_${tag}.txt"

  if [[ -s "$acc_file" ]]; then
    echo "[skip] $tag already scored: $(cat "$acc_file")"
    continue
  fi
  if [[ ! -d "$src" ]]; then
    echo "[warn] missing checkpoint, skipping: $src" >&2
    continue
  fi

  echo "=============================================================="
  echo "[$(date '+%F %T')] $tag   max_model_len=$MAX_LEN  dp=$DP"
  echo "=============================================================="

  # 1) DCP -> HF（bf16 转换 + use_cache 修正）
  if [[ ! -f "$hf/model-00001-of-00001.safetensors" ]]; then
    echo "[convert] $src -> $hf"
    python "$REPO/tools/prepare_hf_ckpt_for_eval.py" \
      --ckpt-step-dir "$src" --out-dir "$hf" --tokenizer "$BASE"
  else
    echo "[convert] reuse existing $hf"
  fi

  # 2) 4 卡数据并行评测，32k 上下文
  echo "[eval] -> $out"
  python vllm_opd_aime.py \
    --model-path "$hf" \
    --test-file ./aime_2024.jsonl \
    --output-path "$out" \
    --max-model-len "$MAX_LEN" \
    --data-parallel-size "$DP" \
    2>&1 | tee "$RESULTS/log_${tag}.txt" | grep -E "^\[eval\]|^acc:" || true

  grep -E "^acc:" "$RESULTS/log_${tag}.txt" | tail -1 > "$acc_file" || true
  echo "[done] $tag $(cat "$acc_file" 2>/dev/null)"
  rm -rf "$hf"    # 每个 HF ckpt 约 3.5G，跑完即释放 /dev/shm
done

echo
echo "================== avg@32 =================="
python "$REPO/tools/summarize_aime_results.py" --results-dir "$RESULTS"

echo
echo "================== pass@1/8/16/32 =================="
python "$REPO/tools/aime_pass_at_k.py" "$RESULTS/aime2024_*_avg32.jsonl"
