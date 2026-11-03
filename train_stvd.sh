#!/usr/bin/env bash
# OPD 蒸馏：JustRL2 step-900 (teacher) → DeepSeek-R1-Distill-Qwen-1.5B (student)
# 教师是 verl06/recipe/justrl2/run_justrl2_16k.sh 训练 DeepSeek-R1-Distill-Qwen-1.5B
# RL 模型（global_step_900，已转 HF 格式到 /dev/shm/llms/justrl2_step_900/）。
# 损失为平方全变差距离 STVD（loss_fn.kl_type=stvd），其余设置与 train_opd.sh 一致，
# 用于和 train_opd.sh 的 reverse KL 基线对比。配置见 examples/configs/distillation_stvd.yaml。
#   - 16k 序列长度、STVD = (0.5 * sum_i |p_S - p_T|)^2，每位置有界在 [0, 1]
#   - 本脚本串行跑两组对照：loss_fn.zero_outside_topk = false 和 true，
#     run/checkpoint/日志分别带 -ztkfalse / -ztktrue 后缀，其余超参与随机种子完全一致
#   - 训练 distillation.max_num_steps 步；训练前及每 distillation.val_period 步测评、每 checkpointing.save_period 步保存权重
#   - 验证集 AIME 2024/2025/2026，每题采样 8 次（3 × 30 × 8 = 720 条），分年份报 validation/accuracy/aime<年>
#   - checkpoint 保存超时放宽到 30 分钟
#   - save_consolidated=false（nemo_automodel consolidate 的 all_gather_object 会误算尺寸 OOM），
#     训练结束后手工转 HF：
#       python examples/converters/convert_dcp_to_hf.py \
#         --config <ckpt>/config.yaml \
#         --dcp-ckpt-path <ckpt>/policy/weights --hf-ckpt-path <out>
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
# 数据集已缓存，模型是本地目录，可离线；如需重新下载请注释掉这两行
# export HF_HUB_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

# ====== 保存 checkpoint 超时 = 30 分钟 ======
# 分布式集合通信默认 10 分钟，consolidated 保存 7B/1.5B 权重时 rank 间等待易超时
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
# vLLM rollout 侧超时同样放宽，避免长序列生成被误判超时
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800

# ====== WandB 在线上传 ======
export WANDB_MODE=online
export WANDB_PROJECT=nemo-distillation

# ====== 让所有 Ray worker 使用当前 conda 环境的 Python，彻底绕过 uv ======
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# 清理旧的 venv 构建残留（避免 STARTED_ENV_BUILDER 死锁）
rm -rf /group/40092/howu/RL-main/venvs

# 停止残留的 Ray 进程（避免旧 worker 缓存问题）
ray stop --force 2>/dev/null || true

sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' /group/40092/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py
# ====== 结束 ======

export PYTHONPATH=/group/40092/howu/RL-main:${PYTHONPATH:-}

# ====== 模型路径（内存盘） ======
TEACHER_MODEL="/dev/shm/llms/justrl2_step_900/"
POLICY_MODEL="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
for d in "$TEACHER_MODEL" "$POLICY_MODEL"; do
  if [[ ! -f "${d}config.json" ]]; then
    echo "❌ 模型缺失: ${d}config.json 不存在"; exit 1
  fi
done
echo "▶ Using policy=$POLICY_MODEL  teacher=$TEACHER_MODEL"

# 实验名前缀（每组再追加 -ztk<false|true>）
RUN_PREFIX="opd-justrl2step900-to-r1qwen15b-stvd-16k-step50"

# ====== 两组对照实验：loss_fn.zero_outside_topk = false / true ======
# 唯一变量是学生落在 teacher top-k 之外的概率质量是否计入 STVD：
#   false（基线）：teacher/student 都截断到 top-k 并重新归一化，只在窗口内算距离
#   true          ：student 用全词表 log_softmax（不重归一化），窗口外的质量通过
#                   (1 - sum_i q_i) 计入，等于对"top-k 外为 0 的 teacher"算精确 TVD
# 两组共用 distillation.seed=42，其余超参完全一致。8 卡独占，只能串行跑。
ZTK_VARIANTS=(false true)

mkdir -p /group/40092/howu/RL-main/logs

export HTTP_PROXY=http://star-proxy.oa.com:3128
export HTTPS_PROXY=http://star-proxy.oa.com:3128
export WANDB_INIT_TIMEOUT=300

cd /group/40092/howu/RL-main

SUMMARY=()
for ZTK in "${ZTK_VARIANTS[@]}"; do
    RUN_NAME="${RUN_PREFIX}-ztk${ZTK}"
    LOG_FILE="logs/${RUN_NAME}.log"

    echo ""
    echo "============================================================"
    echo "[$(date '+%F %T')] START  zero_outside_topk=${ZTK}"
    echo "  run      : ${RUN_NAME}"
    echo "  log      : ${LOG_FILE}"
    echo "  ckpt     : checkpoints/distillation-${RUN_NAME}"
    echo "============================================================"

    # 上一组可能留下残余 Ray 集群，先清掉再起新的
    ray stop --force 2>/dev/null || true

    # 不用 `|| true`：那样会把退出码洗成 0，崩溃和跑完 50 步看起来一模一样。
    # 这里显式收集退出码，既保证前一组失败时后一组照常继续，又能在末尾看清结果。
    set +e
    python examples/run_distillation_math.py \
        --config examples/configs/distillation_stvd.yaml \
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
        distillation.max_num_steps=50 \
        distillation.val_period=5 \
        distillation.val_at_start=true \
        distillation.max_val_samples=768 \
        data.val_aime_years="[2024,2025,2026]" \
        data.val_repeat=8 \
        checkpointing.save_period=5 \
        checkpointing.keep_top_k=10 \
        checkpointing.save_consolidated=false \
        checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
        logger.wandb_enabled=true \
        loss_fn.kl_type=stvd \
        loss_fn.zero_outside_topk="${ZTK}" \
        logger.wandb.name="${RUN_NAME}" 2>&1 | tee "${LOG_FILE}"
    RC=${PIPESTATUS[0]}
    set -e

    if [[ $RC -eq 0 ]]; then
        echo "[$(date '+%F %T')] DONE   zero_outside_topk=${ZTK} (exit 0)"
    else
        echo "[$(date '+%F %T')] FAILED zero_outside_topk=${ZTK} (exit ${RC}) — 详见 ${LOG_FILE}"
    fi
    SUMMARY+=("zero_outside_topk=${ZTK}  exit=${RC}  run=${RUN_NAME}")
done

echo ""
echo "============================================================"
echo "两组实验汇总"
echo "============================================================"
printf '  %s\n' "${SUMMARY[@]}"

