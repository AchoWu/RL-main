#!/usr/bin/env bash
# score term 权重扫描：Skywork-OR1-Math-7B (teacher) → DeepSeek-R1-Distill-Qwen-1.5B (student)
# 只调 loss_fn.score_loss_weight（λ），block_size / block_mode / block_baseline / teacher
# / 学习率 / 随机种子都与 train_bopd.sh 的 sliding-bs8 组完全一致。
#
# L = L_direct + λ * L_score
#   L_direct：逐 token reverse KL，λ 不影响它。
#   L_score ：score-function 修正项 grad log q(y_t) * sg(R_t)，前向值为 0。
# 两项在 reduction 之前就加在一起（loss_functions.py 的 per_token_loss），所以 λ 不改变
# 任何归一化方式，也不改变日志里的 loss（仍是逐 token KL 均值），各组曲线可直接比较。
#
#   - λ=0 时 score 项被完全跳过，梯度与 block_size=1 逐位相同。所以 bs8-λ0 和标准 OPD
#     在同一 seed 下应该给出同一条曲线；留着它当确定性对照，确认 run 之间的差异来自 λ
#     而不是采样噪声。如果只想省机时，可以把 RUN_LAMBDA0=false 跳过。
#   - base 模型诊断（tools/bopd_diagnostics，docs/design-docs/opd.md 3.1.1）给出 bs8 的
#     |E G_score| / |E G_direct| ≈ 0.24，所以各 λ 下 score/direct 的期望梯度比约为
#     λ=0.25→6%、0.5→12%、1.0→24%、1.5→36%。这只是 base 模型上的估计，训练中途会变。
#   - train_global_batch_size 必须等于每步 rollout 数（128）：score 项要求 on-policy 样本。
#     λ=0 时该约束自动放宽（没有 score 项），但这里仍统一用 gbs=128 保证各组可比。
#   - 训练 100 步，每 10 步测评 + 存权重；验证集为 DeepScaleR 留出的 1000 题，每题 1 次。
#   - save_consolidated=false，训练结束后手工转 HF：
#       python examples/converters/convert_dcp_to_hf.py \
#         --config <ckpt>/config.yaml \
#         --dcp-ckpt-path <ckpt>/policy/weights --hf-ckpt-path <out>
#
# 用法：bash train_bopd_lambda.sh                       # OPD 对照 + λ=0/0.25/0.5/1.0/1.5
#       RUN_OPD_BASELINE=false bash train_bopd_lambda.sh   # 已经跑过 OPD，跳过
#       LAMBDAS="0.25 0.5" RUN_OPD_BASELINE=false RUN_LAMBDA0=false bash train_bopd_lambda.sh
#       BLOCK_SIZE=4 bash train_bopd_lambda.sh             # 换 block size 扫同一组 λ
set -euo pipefail

trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python /group/40092/howu/RL-main/test_gpu.py
' EXIT INT TERM

# ====== conda 环境（用 opd，env-3.12 已被清空） ======
source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd
echo "▶ Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# ====== 代理（HF / wandb 都需要，容器内默认没有代理变量） ======
: "${ENV_VENUS_PROXY:=http://star-proxy.oa.com:3128}"
export NO_PROXY=localhost,127.0.0.1,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

# ====== HF 缓存（数据集已预下载到这里；模型走 /dev/shm 本地路径） ======
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# ====== 保存 checkpoint 超时 = 30 分钟 ======
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800

# ====== WandB 在线上传 ======
export WANDB_MODE=online
export WANDB_PROJECT=nemo-distillation

# ====== 让所有 Ray worker 使用当前 conda 环境的 Python，彻底绕过 uv ======
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

rm -rf /group/40092/howu/RL-main/venvs
ray stop --force 2>/dev/null || true

sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' /group/40092/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py

export PYTHONPATH=/group/40092/howu/RL-main:${PYTHONPATH:-}

# ====== 模型路径（内存盘） ======
TEACHER_MODEL="/dev/shm/llms/Skywork-OR1-Math-7B/"
POLICY_MODEL="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
for d in "$TEACHER_MODEL" "$POLICY_MODEL"; do
  if [[ ! -f "${d}config.json" ]]; then
    echo "❌ 模型缺失: ${d}config.json 不存在"; exit 1
  fi
done
echo "▶ Using policy=$POLICY_MODEL  teacher=$TEACHER_MODEL"

# ====== 实验组 ======
BLOCK_SIZE="${BLOCK_SIZE:-8}"
BLOCK_MODE="${BLOCK_MODE:-sliding}"
BLOCK_BASELINE="${BLOCK_BASELINE:-false}"
LAMBDAS="${LAMBDAS:-0.25 0.5 1.0 1.5}"
# 默认跳过：OPD 对照已经跑过（AUC 0.5672），直接复用那次结果即可。设成 true 会重跑一组，
# run 名带 -rerun 后缀，不会覆盖原来的。
RUN_OPD_BASELINE="${RUN_OPD_BASELINE:-false}"
RUN_LAMBDA0="${RUN_LAMBDA0:-true}"

# 每组格式："<标签> <block_size> <block_mode> <block_baseline> <lambda> <run 名>"
PREFIX="bopd-skywork7b-to-r1qwen15b-reverse"
ARMS=()
if [[ "${RUN_OPD_BASELINE}" == "true" ]]; then
    # 名字带 -rerun 后缀，避免覆盖已经跑完的 ...-16k-step100-opd 那组的 ckpt / 日志
    ARMS+=("opd 1 fixed false 1.0 ${PREFIX}-16k-step100-opd-rerun")
fi
if [[ "${RUN_LAMBDA0}" == "true" ]]; then
    ARMS+=("lam0 ${BLOCK_SIZE} ${BLOCK_MODE} ${BLOCK_BASELINE} 0.0 ${PREFIX}-bs${BLOCK_SIZE}-16k-step100-${BLOCK_MODE}-lam0")
fi
for LAM in ${LAMBDAS}; do
    # 0.25 -> lam0p25：用 p 当小数点，lam10 这种写法会被读成 λ=10
    TAG="lam${LAM//./p}"
    ARMS+=("${TAG} ${BLOCK_SIZE} ${BLOCK_MODE} ${BLOCK_BASELINE} ${LAM} ${PREFIX}-bs${BLOCK_SIZE}-16k-step100-${BLOCK_MODE}-${TAG}")
done

mkdir -p /group/40092/howu/RL-main/logs/bopd

export WANDB_INIT_TIMEOUT=300

cd /group/40092/howu/RL-main

SUMMARY=()
for ARM in "${ARMS[@]}"; do
    read -r LABEL ARM_BLOCK_SIZE ARM_BLOCK_MODE ARM_BASELINE ARM_LAMBDA RUN_NAME <<< "${ARM}"
    LOG_FILE="logs/bopd/${RUN_NAME}.log"
    ARM_DESC="${LABEL}: block_size=${ARM_BLOCK_SIZE} block_mode=${ARM_BLOCK_MODE} block_baseline=${ARM_BASELINE} score_loss_weight=${ARM_LAMBDA}"

    echo ""
    echo "============================================================"
    echo "[$(date '+%F %T')] START  ${ARM_DESC}"
    echo "  run      : ${RUN_NAME}"
    echo "  log      : ${LOG_FILE}"
    echo "  ckpt     : checkpoints/distillation-${RUN_NAME}"
    echo "============================================================"

    ray stop --force 2>/dev/null || true

    # 不用 `|| true`：那样会把退出码洗成 0，崩溃和跑完看起来一模一样。
    set +e
    python examples/run_distillation_math.py \
        --config examples/configs/distillation_math.yaml \
        policy.model_name="$POLICY_MODEL" \
        teacher.model_name="$TEACHER_MODEL" \
        cluster.gpus_per_node=8 \
        policy.train_micro_batch_size=1 \
        policy.train_global_batch_size=128 \
        policy.optimizer.kwargs.lr=1.0e-6 \
        policy.max_total_sequence_length=16384 \
        teacher.max_total_sequence_length=16384 \
        teacher.logprob_batch_size=1 \
        teacher.dtensor_cfg.tensor_parallel_size=2 \
        teacher.dtensor_cfg.context_parallel_size=2 \
        distillation.max_num_epochs=1 \
        distillation.max_num_steps=100 \
        distillation.val_period=10 \
        distillation.val_at_start=true \
        distillation.max_val_samples=1024 \
        +data.val_holdout_size=1000 \
        +data.val_repeat=1 \
        checkpointing.save_period=10 \
        checkpointing.keep_top_k=10 \
        checkpointing.save_consolidated=false \
        checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
        logger.wandb_enabled=false \
        loss_fn.kl_type=reverse \
        loss_fn.block_size="${ARM_BLOCK_SIZE}" \
        loss_fn.block_mode="${ARM_BLOCK_MODE}" \
        loss_fn.block_baseline="${ARM_BASELINE}" \
        loss_fn.score_loss_weight="${ARM_LAMBDA}" \
        logger.wandb.name="${RUN_NAME}" 2>&1 | tee "${LOG_FILE}"
    RC=${PIPESTATUS[0]}
    set -e

    if [[ $RC -eq 0 ]]; then
        echo "[$(date '+%F %T')] DONE   ${ARM_DESC} (exit 0)"
    else
        echo "[$(date '+%F %T')] FAILED ${ARM_DESC} (exit ${RC}) — 详见 ${LOG_FILE}"
    fi
    SUMMARY+=("${ARM_DESC}  exit=${RC}  run=${RUN_NAME}")
done

echo ""
echo "============================================================"
echo "实验汇总"
echo "============================================================"
printf '  %s\n' "${SUMMARY[@]}"
