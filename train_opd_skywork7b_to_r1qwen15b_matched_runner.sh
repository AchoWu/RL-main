#!/usr/bin/env bash
# Shared 50-step runner for matched reverse-OPD token-weighting ablations.

set -euo pipefail

source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd
echo "Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"

: "${ENV_VENUS_PROXY:=http://star-proxy.oa.com:3128}"
export NO_PROXY=localhost,127.0.0.1,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY="$ENV_VENUS_PROXY"
export HTTPS_PROXY="$ENV_VENUS_PROXY"
export no_proxy="$NO_PROXY"
export http_proxy="$ENV_VENUS_PROXY"
export https_proxy="$ENV_VENUS_PROXY"

export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
export NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true

export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800

export WANDB_MODE=online
export WANDB_PROJECT=nemo-distillation
export WANDB_INIT_TIMEOUT=300
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

REPO_DIR="${REPO_DIR:-/group/40092/howu/RL-main}"
TEACHER_MODEL="${TEACHER_MODEL:-/dev/shm/llms/Skywork-OR1-Math-7B/}"
POLICY_MODEL="${POLICY_MODEL:-/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/}"
OPD_CONFIG_PATH="${OPD_CONFIG_PATH:-examples/configs/distillation_math_tvd_matched_full.yaml}"
OPD_RUN_TAG="${OPD_RUN_TAG:-matched-full-token}"

for model_dir in "$TEACHER_MODEL" "$POLICY_MODEL"; do
    if [[ ! -f "${model_dir}config.json" ]]; then
        echo "Model config missing: ${model_dir}config.json"
        exit 1
    fi
done
if [[ ! -f "${REPO_DIR}/${OPD_CONFIG_PATH}" ]]; then
    echo "Experiment config missing: ${REPO_DIR}/${OPD_CONFIG_PATH}"
    exit 1
fi

rm -rf "${REPO_DIR}/venvs"
ray stop --force 2>/dev/null || true
sed -i \
    's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' \
    "${REPO_DIR}/nemo_rl/distributed/ray_actor_environment_registry.py"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

DATESTR="$(date +%y%m%d)"
RUN_NAME="opd-skywork7b-to-r1qwen1.5b-reverse-${OPD_RUN_TAG}-step50-16k-${DATESTR}"
mkdir -p "${REPO_DIR}/logs"

echo "Policy:  $POLICY_MODEL"
echo "Teacher: $TEACHER_MODEL"
echo "Config:  $OPD_CONFIG_PATH"
echo "Run:     $RUN_NAME"

cd "$REPO_DIR"
python examples/run_distillation_math.py \
    --config "$OPD_CONFIG_PATH" \
    policy.model_name="$POLICY_MODEL" \
    teacher.model_name="$TEACHER_MODEL" \
    cluster.gpus_per_node=8 \
    policy.train_micro_batch_size=1 \
    policy.train_global_batch_size=128 \
    policy.optimizer.kwargs.lr=1.0e-6 \
    policy.max_total_sequence_length=16384 \
    teacher.max_total_sequence_length=16384 \
    teacher.logprob_batch_size=1 \
    distillation.max_num_epochs=1 \
    distillation.max_num_steps=50 \
    distillation.val_period=5 \
    distillation.val_at_start=true \
    distillation.max_val_samples=1000 \
    data.validation_source=train_holdout \
    data.validation_num_samples=1000 \
    data.validation_seed=42 \
    checkpointing.save_period=5 \
    checkpointing.keep_top_k=10 \
    checkpointing.save_consolidated=false \
    checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
    loss_fn.kl_type=reverse \
    loss_fn.zero_outside_topk=true \
    logger.wandb_enabled=true \
    logger.wandb.name="$RUN_NAME"
