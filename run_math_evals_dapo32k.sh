#!/usr/bin/env bash
# 多基准评测：AIME2025 (32采) + MATH-500 (4采) + Omni-MATH-Rule (4采)。
# 模型: student base + DAPO-32k 4 组 step_106，全部 max_model_len=32768、8 卡 DP。
#
# 用法:
#   bash run_math_evals_dapo32k.sh                     # 全部 5 个模型
#   bash run_math_evals_dapo32k.sh dapo32k-seqmean     # 只跑指定模型
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
STEP=step_106

mkdir -p "$STAGE" "$RESULTS"

# 模型清单: <短名>:<checkpoint 目录名, 空=base>
MODELS=(
  "base-r1qwen1.5b-student-32k:"
  "dapo32k-nomask-reverse:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-dapo32k-nomask-reverse-step106-dapo32k-260824"
  "dapo32k-seqmean:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-dapo32k-oldkl-sequence-mean-step106-dapo32k-260824"
  "dapo32k-tokmean-refkl:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-dapo32k-oldkl-token-mean-refkl-b0.01-step106-dapo32k-260824"
  "dapo32k-seqmean-refkl:distillation-opd-skywork7b-to-r1qwen1.5b-reverse-dapo32k-oldkl-sequence-mean-refkl-b0.01-step106-dapo32k-260824"
)

# 基准清单: <基准名>:<jsonl 路径>:<采样数>
BENCHMARKS=(
  "aime2025:eval_datasets/aime_2025.jsonl:32"
  "math500:eval_datasets/math_500.jsonl:4"
  "omni-math-rule:eval_datasets/omni_math_rule.jsonl:4"
)

WANTED=("$@")
cd "$REPO"

for entry in "${MODELS[@]}"; do
  short="${entry%%:*}"
  dir="${entry#*:}"

  # 该模型是否还有未完成的基准
  todo=()
  for b in "${BENCHMARKS[@]}"; do
    bname="${b%%:*}"
    [[ -s "$RESULTS/acc_${bname}_${short}.txt" ]] || todo+=("$b")
  done
  if [[ ${#todo[@]} -eq 0 ]]; then
    echo "[skip] $short: 3 个基准都已完成"
    continue
  fi
  if [[ ${#WANTED[@]} -gt 0 ]]; then
    match=0
    for w in "${WANTED[@]}"; do
      [[ "$w" == "$short" ]] && match=1
    done
    [[ $match -eq 1 ]] || continue
  fi

  # 准备模型路径: base 直接用; checkpoint 转换一次，跨基准复用
  if [[ -n "$dir" ]]; then
    src="$CKPT_ROOT/$dir/$STEP"
    hf="$STAGE/dapo32k-eval-${short}"
    if [[ ! -f "$hf/model-00001-of-00001.safetensors" ]]; then
      echo "[convert] $src -> $hf"
      python "$REPO/tools/prepare_hf_ckpt_for_eval.py" \
        --ckpt-step-dir "$src" --out-dir "$hf" --tokenizer "$STUDENT"
    fi
    model_path="$hf"
  else
    model_path="$STUDENT"
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
      --model-path "$model_path" \
      --test-file "$REPO/$bfile" \
      --output-path "$out" \
      --max-model-len "$MAX_LEN" \
      --num-generation "$nsamp" \
      --data-parallel-size "$DP" \
      2>&1 | tee "$RESULTS/log_${bname}_${short}.txt" | grep -E "^\[eval\]|^acc:" || true

    grep -E "^acc:" "$RESULTS/log_${bname}_${short}.txt" | tail -1 > "$acc_file" || true
    echo "[done] $bname/$short $(cat "$acc_file" 2>/dev/null)"
  done

  [[ -n "$dir" ]] && rm -rf "$hf"   # 三个基准都跑完，释放 /dev/shm
done

echo
echo "================== 各基准 avg@N 汇总 =================="
python - <<'PYEOF'
import glob, os, re
rows = {}
for p in sorted(glob.glob('aime_eval_results_dapo32k/acc_*.txt')):
    name = os.path.basename(p)[4:-4]
    m = re.match(r'(.+?)_(base-r1qwen1\.5b-student-32k|dapo32k-[a-z-]+)$', name)
    if not m:
        continue
    bench, model = m.group(1), m.group(2)
    acc = open(p).read().strip()
    rows.setdefault(bench, {})[model] = acc

for bench, models in rows.items():
    print(f"\n--- {bench}")
    for model, acc in sorted(models.items(), key=lambda kv: -float(kv[1].split(': ')[1])):
        print(f"  {model:<40} {acc}")
PYEOF

echo
echo "================== pass@k (AIME2025, 32采) =================="
ls aime_eval_results_dapo32k/aime2025_*.jsonl >/dev/null 2>&1 && \
  python "$REPO/tools/aime_pass_at_k.py" aime_eval_results_dapo32k/aime2025_*.jsonl || echo "(尚未有 AIME2025 结果)"
