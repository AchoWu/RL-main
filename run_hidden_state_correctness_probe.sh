#!/usr/bin/env bash
# Frozen hidden-state probe for final trajectory correctness.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/group/40094/jingweidong/RL-main}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/hidden-state-correctness-probe}"
MODEL="${MODEL:-/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROBLEMS="${NUM_PROBLEMS:-1000}"
TRAJECTORIES_PER_PROBLEM="${TRAJECTORIES_PER_PROBLEM:-2}"
MAX_GENERATION_TOKENS="${MAX_GENERATION_TOKENS:-16384}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-8}"
LAYER_FRACTIONS="${LAYER_FRACTIONS:-0.5,0.75,0.9,1.0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd

: "${ENV_VENUS_PROXY:=http://star-proxy.oa.com:3128}"
export NO_PROXY=localhost,127.0.0.1,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY="$ENV_VENUS_PROXY"
export HTTPS_PROXY="$ENV_VENUS_PROXY"
export no_proxy="$NO_PROXY"
export http_proxy="$ENV_VENUS_PROXY"
export https_proxy="$ENV_VENUS_PROXY"
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${MODEL%/}/config.json" ]]; then
    echo "Model missing: ${MODEL%/}/config.json" >&2
    exit 1
fi

COMMON_ARGS=(
    --output-dir "$OUTPUT_DIR"
    --model "$MODEL"
    --num-problems "$NUM_PROBLEMS"
    --trajectories-per-problem "$TRAJECTORIES_PER_PROBLEM"
    --max-generation-tokens "$MAX_GENERATION_TOKENS"
    --max-checkpoints "$MAX_CHECKPOINTS"
    --layer-fractions "$LAYER_FRACTIONS"
    --attn-implementation "$ATTN_IMPLEMENTATION"
    --num-shards "$NUM_GPUS"
)

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR" "${REPO_ROOT}/logs"
for phase in generate extract; do
    echo "============================================================"
    echo "Hidden-state probe phase: $phase"
    echo "Output: $OUTPUT_DIR"
    echo "============================================================"
    phase_start=$SECONDS
    pids=()
    for ((shard = 0; shard < NUM_GPUS; shard++)); do
        CUDA_VISIBLE_DEVICES="$shard" python -m \
            research.hidden_state_probe.run_hidden_state_probe \
            "$phase" "${COMMON_ARGS[@]}" --shard-index "$shard" \
            > "${REPO_ROOT}/logs/hidden-probe-${phase}-shard${shard}.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if ((failed)); then
        echo "Phase $phase failed; inspect logs/hidden-probe-${phase}-shard*.log" >&2
        exit 1
    fi
    echo "Phase $phase completed in $((SECONDS - phase_start)) seconds"
done

CUDA_VISIBLE_DEVICES=0 python -m \
    research.hidden_state_probe.run_hidden_state_probe \
    train "${COMMON_ARGS[@]}" --shard-index 0
