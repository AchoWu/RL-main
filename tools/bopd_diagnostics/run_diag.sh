#!/usr/bin/env bash
# Block-OPD 梯度诊断（docs/design-docs/opd.md 3.1.1）：在一个固定的 student checkpoint 上
#   Q1  score 梯度 vs direct 梯度（参数空间，各 block size）
#   Q2  y_{i+1} 是否改变未来 KL（同一 prefix 下嵌套分支采样）
#   Q3  baseline 是否降低梯度方差（KL-to-go / logit 空间 / 参数空间三层）
# 三步，每步按 GPU 分片并行：generate（vLLM 采样 rollout）→ analyze → summarize。
#
# 用法：
#   STUDENT=/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/ OUT=diag/base bash tools/bopd_diagnostics/run_diag.sh
#   STUDENT=checkpoints/distillation-<run>/step_50 OUT=diag/<run>-step50 bash tools/bopd_diagnostics/run_diag.sh
# STUDENT 若是训练 checkpoint（含 policy/weights），先自动转成 HF 放在 <ckpt>/hf。
# 可选：NUM_GPUS（默认 8）、NUM_SAMPLES（默认 512 = 4 个 128 的 batch，Q3 的跨 batch
# 统计需要 >= 2 个 batch）、ANALYZE_ARGS（透传给 analyze，如 "--block_modes sliding fixed"）、
# STAGES（默认 "generate analyze summarize"）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f /data/miniconda3/etc/profile.d/conda.sh ]]; then
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate opd
fi
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

: "${STUDENT:?set STUDENT to an HF model dir or a training checkpoint step dir}"
: "${OUT:?set OUT to the output directory}"
TEACHER="${TEACHER:-/dev/shm/llms/Skywork-OR1-Math-7B/}"
BASE_MODEL="${BASE_MODEL:-/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
ANALYZE_ARGS="${ANALYZE_ARGS:-}"
STAGES="${STAGES:-generate analyze summarize}"

if [[ -d "${STUDENT}/policy/weights" ]]; then
    HF_DIR="${STUDENT%/}/hf"
    if [[ ! -f "${HF_DIR}/config.json" ]]; then
        echo "▶ converting ${STUDENT} -> ${HF_DIR}"
        python examples/converters/convert_dcp_to_hf.py \
            --config "${STUDENT%/}/config.yaml" \
            --dcp-ckpt-path "${STUDENT%/}/policy/weights" \
            --hf-ckpt-path "${HF_DIR}"
    fi
    STUDENT="${HF_DIR}"
fi
mkdir -p "${OUT}/logs"
echo "▶ student=${STUDENT} teacher=${TEACHER} out=${OUT}"

run_shards() {
    local stage=$1; shift
    local pids=()
    for ((j = 0; j < NUM_GPUS; j++)); do
        CUDA_VISIBLE_DEVICES=$j python -m tools.bopd_diagnostics.bopd_diag "$stage" \
            --out "$OUT" --shard "$j" --num_shards "$NUM_GPUS" "$@" \
            > "${OUT}/logs/${stage}_${j}.log" 2>&1 &
        pids+=($!)
    done
    local rc=0
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    if [[ $rc -ne 0 ]]; then
        echo "❌ ${stage} failed, see ${OUT}/logs/${stage}_*.log"; exit 1
    fi
    echo "[$(date '+%F %T')] ${stage} done"
}

for stage in $STAGES; do
    case "$stage" in
        generate)
            # prompts and sampling exactly as train_bopd.sh
            run_shards generate --student "$STUDENT" --num_samples "$NUM_SAMPLES" \
                policy.model_name="$BASE_MODEL" \
                policy.max_total_sequence_length=16384 \
                +data.val_holdout_size=1000 +data.val_repeat=1
            ;;
        analyze)
            # shellcheck disable=SC2086
            run_shards analyze --student "$STUDENT" --teacher "$TEACHER" $ANALYZE_ARGS
            ;;
        summarize)
            python -m tools.bopd_diagnostics.bopd_diag summarize --out "$OUT" \
                | tee "${OUT}/logs/summarize.log"
            ;;
        *) echo "unknown stage $stage"; exit 1 ;;
    esac
done
