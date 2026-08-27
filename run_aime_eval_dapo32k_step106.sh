#!/usr/bin/env bash
# AIME-2024 avg@32 / pass@k 评测：DAPO-32k 4 组实验的 step_106 + 两个 base 模型。
# 全部 max_model_len=32768、8 卡数据并行，与 DAPO-32k 训练长度一致。
#
# 用法:
#   bash run_aime_eval_dapo32k_step106.sh                    # 全部 6 个模型
#   bash run_aime_eval_dapo32k_step106.sh dapo32k-seqmean    # 只跑指定模型
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
TEACHER=/dev/shm/llms/Skywork-OR1-Math-7B/
STAGE=/dev/shm/hf_ckpts
RESULTS=$REPO/aime_eval_results_dapo32k
MAX_LEN=32768
DP=8
STEP=step_106

mkdir -p "$STAGE" "$RESULTS"

# <短名>:<checkpoint 目录名>；base 模型无需转换直接评，见下方分支
EXPERIMENTS=(
  "base-r1qwen1.5b-student-32k:"
  "base-skywork7b-teacher-32k:"
  "dapo32k-nomask-reverse:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-dapo32k-nomask-reverse-step106-dapo32k-260824"
  "dapo32k-seqmean:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-dapo32k-oldkl-sequence-mean-step106-dapo32k-260824"
  "dapo32k-tokmean-refkl:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-dapo32k-oldkl-token-mean-refkl-b0.01-step106-dapo32k-260824"
  "dapo32k-seqmean-refkl:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-dapo32k-oldkl-sequence-mean-refkl-b0.01-step106-dapo32k-260824"
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

  # tag 统一带 -32k 后缀，与 aime_eval_results 里 16k 的旧结果区分
  if [[ -n "$dir" ]]; then
    tag="${short}-${STEP}-32k"
    src="$CKPT_ROOT/$dir/$STEP"
  else
    tag="$short"
    case "$short" in
      base-r1qwen1.5b-student-32k) src="$STUDENT" ;;
      base-skywork7b-teacher-32k)  src="$TEACHER" ;;
    esac
  fi
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

  if [[ -n "$dir" ]]; then
    # 训练 checkpoint: DCP -> HF（bf16 转换 + use_cache 修正），评测完删除
    hf="$STAGE/$tag"
    if [[ ! -f "$hf/model-00001-of-00001.safetensors" ]]; then
      echo "[convert] $src -> $hf"
      python "$REPO/tools/prepare_hf_ckpt_for_eval.py" \
        --ckpt-step-dir "$src" --out-dir "$hf" --tokenizer "$STUDENT"
    else
      echo "[convert] reuse existing $hf"
    fi
    model_path="$hf"
  else
    # base 模型已是 HF 格式，直接评
    model_path="$src"
    echo "[base] direct eval: $model_path"
  fi

  echo "[eval] -> $out"
  python vllm_opd_aime.py \
    --model-path "$model_path" \
    --test-file ./aime_2024.jsonl \
    --output-path "$out" \
    --max-model-len "$MAX_LEN" \
    --data-parallel-size "$DP" \
    2>&1 | tee "$RESULTS/log_${tag}.txt" | grep -E "^\[eval\]|^acc:" || true

  grep -E "^acc:" "$RESULTS/log_${tag}.txt" | tail -1 > "$acc_file" || true
  echo "[done] $tag $(cat "$acc_file" 2>/dev/null)"
  [[ -n "$dir" ]] && rm -rf "$hf"   # 释放 /dev/shm（每个约 3.5G）
done

echo
echo "================== avg@32 =================="
python "$REPO/tools/summarize_aime_results.py" --results-dir "$RESULTS"

echo
echo "================== pass@1/8/16/32 =================="
python "$REPO/tools/aime_pass_at_k.py" "$RESULTS"/aime2024_*_avg32.jsonl
