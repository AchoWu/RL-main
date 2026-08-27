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

# Prestarting one worker per CPU (384 here) makes every worker source BASH_ENV
# concurrently; none register and ray.init() blocks forever in RegisterClient.
export RAY_enable_worker_prestart=0
export RAY_prestart_worker_first_driver=0

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
OPD_ZERO_OUTSIDE_TOPK="${OPD_ZERO_OUTSIDE_TOPK:-true}"
OPD_REFERENCE_KL_PENALTY="${OPD_REFERENCE_KL_PENALTY:-}"
OPD_REFERENCE_KL_TYPE="${OPD_REFERENCE_KL_TYPE:-k3}"
# Sequence length / rollout / dataset knobs. Defaults reproduce the 16k
# DeepScaler runs; the DAPO-32k sweep overrides them from its wrapper.
OPD_SEQ_LEN="${OPD_SEQ_LEN:-16384}"
OPD_NUM_GENERATIONS="${OPD_NUM_GENERATIONS:-1}"
OPD_DATASET_NAME="${OPD_DATASET_NAME:-DeepScaler}"
OPD_VAL_NUM_SAMPLES="${OPD_VAL_NUM_SAMPLES:-1000}"
OPD_MAX_VAL_SAMPLES="${OPD_MAX_VAL_SAMPLES:-1000}"
OPD_MAX_NUM_STEPS="${OPD_MAX_NUM_STEPS:-50}"
OPD_VAL_PERIOD="${OPD_VAL_PERIOD:-5}"
OPD_SAVE_PERIOD="${OPD_SAVE_PERIOD:-5}"
OPD_LEN_TAG="${OPD_LEN_TAG:-16k}"
# Extra `key=value` overrides appended verbatim (e.g. data.config_name=en)
OPD_EXTRA_OVERRIDES="${OPD_EXTRA_OVERRIDES:-}"

EXTRA_OVERRIDES=()
if [[ -n "$OPD_EXTRA_OVERRIDES" ]]; then
    read -r -a EXTRA_OVERRIDES <<< "$OPD_EXTRA_OVERRIDES"
fi

REFERENCE_KL_OVERRIDES=()
if [[ -n "$OPD_REFERENCE_KL_PENALTY" ]]; then
    REFERENCE_KL_OVERRIDES+=(
        loss_fn.reference_policy_kl_penalty="$OPD_REFERENCE_KL_PENALTY"
        loss_fn.reference_policy_kl_type="$OPD_REFERENCE_KL_TYPE"
    )
fi

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

# OPD_DATESTR lets a restart reuse the original run's checkpoint dir. Without
# it a next-day restart would get a fresh DATESTR, point at an empty
# checkpoint_dir, and silently retrain from step 0 instead of resuming.
DATESTR="${OPD_DATESTR:-$(date +%y%m%d)}"
RUN_NAME="opd-skywork7b-to-r1qwen1.5b-reverse-${OPD_RUN_TAG}-step${OPD_MAX_NUM_STEPS}-${OPD_LEN_TAG}-${DATESTR}"
mkdir -p "${REPO_DIR}/logs"

echo "Policy:  $POLICY_MODEL"
echo "Teacher: $TEACHER_MODEL"
echo "Config:  $OPD_CONFIG_PATH"
echo "Zero outside top-k: $OPD_ZERO_OUTSIDE_TOPK"
echo "Dataset: $OPD_DATASET_NAME | seq_len=$OPD_SEQ_LEN | rollout=$OPD_NUM_GENERATIONS | val_holdout=$OPD_VAL_NUM_SAMPLES"
if [[ -n "$OPD_REFERENCE_KL_PENALTY" ]]; then
    echo "Reference KL: beta=$OPD_REFERENCE_KL_PENALTY type=$OPD_REFERENCE_KL_TYPE"
fi
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
    policy.max_total_sequence_length="$OPD_SEQ_LEN" \
    teacher.max_total_sequence_length="$OPD_SEQ_LEN" \
    teacher.logprob_batch_size=1 \
    data.dataset_name="$OPD_DATASET_NAME" \
    distillation.num_generations_per_prompt="$OPD_NUM_GENERATIONS" \
    distillation.max_num_epochs=1 \
    distillation.max_num_steps="$OPD_MAX_NUM_STEPS" \
    distillation.val_period="$OPD_VAL_PERIOD" \
    distillation.val_at_start=true \
    distillation.max_val_samples="$OPD_MAX_VAL_SAMPLES" \
    data.validation_source=train_holdout \
    data.validation_num_samples="$OPD_VAL_NUM_SAMPLES" \
    data.validation_seed=42 \
    checkpointing.save_period="$OPD_SAVE_PERIOD" \
    checkpointing.keep_top_k=10 \
    checkpointing.save_consolidated=false \
    checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
    loss_fn.kl_type=reverse \
    loss_fn.zero_outside_topk="$OPD_ZERO_OUTSIDE_TOPK" \
    "${REFERENCE_KL_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}" \
    logger.wandb_enabled=true \
    logger.wandb.name="$RUN_NAME"
