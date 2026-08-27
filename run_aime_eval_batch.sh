#!/usr/bin/env bash
# AIME-2024 avg@32 评测：批量转换 + 评测多个 OPD checkpoint。
# 用法: bash run_aime_eval_batch.sh
set -euo pipefail

source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate aime-eval

export HF_HOME=/root/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# vLLM 0.11.2 在多实例下会各自探测 CPU 亲和性，关掉避免互相抢核
export VLLM_LOGGING_LEVEL=WARNING

REPO=/group/40092/howu/RL-main
CKPT_ROOT=$REPO/checkpoints
BASE=/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/
STAGE=/dev/shm/hf_ckpts
RESULTS=$REPO/aime_eval_results
mkdir -p "$STAGE" "$RESULTS"

# 待评测实验: <短名>:<checkpoint 目录名>
EXPERIMENTS=(
  "seqmean-refkl:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-oldkl-sequence-mean-refkl-b0.01-step50-16k-260822"
  "tokmean-refkl:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-oldkl-token-mean-refkl-b0.01-step50-16k-260822"
  "seqmean:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-oldkl-sequence-mean-step50-16k-260820"
  "nomask-reverse:distillation-opd-skywork7b-to-r1qwen1.5b-nomask-reverse-step50-260818"
)
STEPS=(step_40 step_45 step_50)

cd "$REPO"

for entry in "${EXPERIMENTS[@]}"; do
  short="${entry%%:*}"
  dir="${entry#*:}"
  for step in "${STEPS[@]}"; do
    tag="${short}-${step}"
    src="$CKPT_ROOT/$dir/$step"
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
    echo "[$(date '+%F %T')] $tag"
    echo "=============================================================="

    # 1) DCP -> HF（含 bf16 转换与 use_cache 修正）
    if [[ ! -f "$hf/model-00001-of-00001.safetensors" ]]; then
      echo "[convert] $src -> $hf"
      python "$REPO/tools/prepare_hf_ckpt_for_eval.py" \
        --ckpt-step-dir "$src" --out-dir "$hf" --tokenizer "$BASE"
    else
      echo "[convert] reuse existing $hf"
    fi

    # 2) 8 卡数据并行评测
    echo "[eval] -> $out"
    python vllm_opd_aime.py \
      --model-path "$hf" \
      --test-file ./aime_2024.jsonl \
      --output-path "$out" \
      --data-parallel-size 8 \
      2>&1 | tee "$RESULTS/log_${tag}.txt" | grep -E "^\[eval\]|^acc:" || true

    # 3) 记录分数，并释放 /dev/shm（每个 HF ckpt 3.5G）
    grep -E "^acc:" "$RESULTS/log_${tag}.txt" | tail -1 > "$acc_file" || true
    echo "[done] $tag $(cat "$acc_file" 2>/dev/null)"
    rm -rf "$hf"
  done
done

echo
echo "================== ALL RESULTS =================="
python "$REPO/tools/summarize_aime_results.py" --results-dir "$RESULTS"
