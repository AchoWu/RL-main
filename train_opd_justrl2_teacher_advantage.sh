#!/usr/bin/env bash
# Reverse-KL OPD baseline plus a read-only per-token teacher advantage monitor.
set -euo pipefail
trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python test_gpu.py
' EXIT INT TERM

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JSTRL_DIR="${JSTRL_REPO:-/group/40092/howu/JustRL-II}"
BASE_MODEL="${JSTRL_BASE_MODEL:-${JSTRL_DIR}/models/JustRL-II-base-model}"
TEACHER_MODEL="${JSTRL_TEACHER_MODEL:-${JSTRL_DIR}/runs/justrl2_16k_new/hf/iter_0000099}"
CRITIC_CKPT="${JSTRL_CRITIC_CKPT:-${JSTRL_DIR}/runs/justrl2_16k_new_critic_archive/iter_0000099}"
CRITIC_EXPORT="${JSTRL_CRITIC_EXPORT:-${REPO_DIR}/critic_exports/justrl2_16k_new_iter_0000099}"
CRITIC_EXPORT_PYTHON="${CRITIC_EXPORT_PYTHON:-python}"
RUN_NAME="${OPD_RUN_NAME:-justrl2-teacher-advantage-$(date +%Y%m%d-%H%M%S)}"

for model_dir in "$BASE_MODEL" "$TEACHER_MODEL"; do
    [[ -f "${model_dir}/config.json" ]] || { echo "Missing model config: ${model_dir}/config.json" >&2; exit 1; }
done
[[ -f "${CRITIC_CKPT}/.metadata" ]] || { echo "Missing critic checkpoint: ${CRITIC_CKPT}/.metadata" >&2; exit 1; }
if [[ -f /data/miniconda3/etc/profile.d/conda.sh && -d /data/miniconda3/envs/opd ]]; then
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate opd
fi
command -v python >/dev/null || { echo "Python is unavailable; activate the OPD training environment" >&2; exit 1; }
python -c 'import torch; n=torch.cuda.device_count(); assert n == 8, f"Expected 8 visible GPUs, got {n}"'

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
if [[ -n "${ENV_VENUS_PROXY:-}" ]]; then
    export HTTP_PROXY="$ENV_VENUS_PROXY" HTTPS_PROXY="$ENV_VENUS_PROXY"
    export http_proxy="$ENV_VENUS_PROXY" https_proxy="$ENV_VENUS_PROXY"
fi

if [[ ! -f "${CRITIC_EXPORT}/value_head.safetensors" ]]; then
    "$CRITIC_EXPORT_PYTHON" "${REPO_DIR}/tools/export_justrl2_critic.py" \
        --justrl-repo "$JSTRL_DIR" --base "$BASE_MODEL" \
        --critic "$CRITIC_CKPT" --output "$CRITIC_EXPORT"
fi

cd "$REPO_DIR"
python examples/run_distillation_math.py \
    --config examples/configs/distillation_math_justrl2_teacher_advantage.yaml \
    policy.model_name="$BASE_MODEL" \
    teacher.model_name="$TEACHER_MODEL" \
    teacher_advantage_monitor.critic_hf_dir="$CRITIC_EXPORT" \
    teacher_advantage_monitor.output_jsonl="logs/${RUN_NAME}-teacher-advantages.jsonl" \
    checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
    logger.wandb_enabled=false \
    logger.wandb.name="$RUN_NAME" \
    "$@"
