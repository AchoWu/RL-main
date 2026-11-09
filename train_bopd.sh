#!/usr/bin/env bash
# 块级 OPD 蒸馏：Skywork-OR1-Math-7B (teacher) → DeepSeek-R1-Distill-Qwen-1.5B (student)
# 教师 /dev/shm/llms/Skywork-OR1-Math-7B/（Qwen2.5-7B 系，与 E1/E2 同一个 teacher）。
# 师生 tokenizer 相同但 config vocab_size 不同（152064 vs 151936），差异只在 padding 行，
# 由 check_vocab_equality + _remap_teacher_topk_outside_student_vocab_ 处理。
# 损失从逐 token reverse KL 换成块级 reverse KL（loss_fn.block_size > 1）：在逐 token KL
# 之外多出 grad log q(y_t) * sg(块内后续 token 的 KL 之和) 这一项，即对
# E_{y~q}[块内 sum_t KL~_t] 求精确梯度。KL~_t 是 teacher top-k（这里 k=64）上重新归一化
# 后的 KL，所以只有 top-k 覆盖整个词表时才等于块联合分布的 KL。见 docs/design-docs/opd.md。
#   - 16k 序列长度、reverse KL，块大小由环境变量 BLOCK_SIZE 指定（默认 4，须 >= 2）
#   - BLOCK_BASELINE=true 时 score 项减去只依赖 y_<t 的 baseline（loss_fn.block_baseline，
#     无偏、降方差），run 名追加 -bl；默认 false。只接受 true / false
#   - train_global_batch_size 必须等于每步 rollout 数（128），否则 setup 会 assert 失败
#   - 支持三组对照，除 loss 外超参与随机种子完全一致：
#       opd     ：标准 OPD 基线，逐 token reverse KL（block_size=1），与 BLOCK_SIZE /
#                 BLOCK_BASELINE 无关。扫多个 BLOCK_SIZE 时只需跑一次，之后用
#                 RUN_OPD_BASELINE=false 跳过
#       fixed   ：不重叠块
#       sliding ：fixed 对所有块偏移取平均，后面第 j 个 token 权重 (k-j)/k
#     run/checkpoint/日志分别带 -opd / -fixed / -sliding 后缀
#   - 训练 distillation.max_num_steps 步；训练前及每 distillation.val_period 步测评、每 checkpointing.save_period 步保存权重
#   - 验证集：从 DeepScaleR 训练集（seed=42 打乱后）留出 1000 题并从训练集删除（含重复题），
#     每题采样 1 次（1000 条），报 validation/accuracy
#   - checkpoint 保存超时放宽到 30 分钟
#   - save_consolidated=false（nemo_automodel consolidate 的 all_gather_object 会误算尺寸 OOM），
#     训练结束后手工转 HF：
#       python examples/converters/convert_dcp_to_hf.py \
#         --config <ckpt>/config.yaml \
#         --dcp-ckpt-path <ckpt>/policy/weights --hf-ckpt-path <out>
#
# 用法：bash train_bopd.sh                                   # opd + block_size=4 的 fixed / sliding
#       BLOCK_SIZE=8 RUN_OPD_BASELINE=false bash train_bopd.sh
#       BLOCK_SIZE=8 BLOCK_BASELINE=true RUN_OPD_BASELINE=false bash train_bopd.sh
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
TEACHER_MODEL="/dev/shm/llms/Skywork-OR1-Math-7B/"
POLICY_MODEL="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
for d in "$TEACHER_MODEL" "$POLICY_MODEL"; do
  if [[ ! -f "${d}config.json" ]]; then
    echo "❌ 模型缺失: ${d}config.json 不存在"; exit 1
  fi
done
echo "▶ Using policy=$POLICY_MODEL  teacher=$TEACHER_MODEL"

# ====== 标准 OPD + sliding 块大小扫描 ======
# 先跑标准 OPD baseline（逐 token reverse KL，block_size=1），
# 再跑 sliding：BLOCK_SIZE=2、4、8、16。不跑 fixed，关闭 block_baseline。
# 五组串行运行，其余训练参数保持不变。
BLOCK_SIZES=(2 4 8 16)

# 每组格式：
# "<标签> <block_size> <block_mode> <block_baseline> <run 名>"
# block_size=1 时 block_mode / block_baseline 不起作用，loss 退化为逐 token KL
ARMS=("opd 1 fixed false bopd-skywork7b-to-r1qwen15b-reverse-16k-step100-opd")
for BS in "${BLOCK_SIZES[@]}"; do
    RUN_NAME="bopd-skywork7b-to-r1qwen15b-reverse-bs${BS}-16k-step100-sliding"
    ARMS+=("sliding ${BS} sliding false ${RUN_NAME}")
done

mkdir -p /group/40092/howu/RL-main/logs/bopd

export HTTP_PROXY=http://star-proxy.oa.com:3128
export HTTPS_PROXY=http://star-proxy.oa.com:3128
export WANDB_INIT_TIMEOUT=300

cd /group/40092/howu/RL-main

SUMMARY=()
for ARM in "${ARMS[@]}"; do
    read -r LABEL ARM_BLOCK_SIZE ARM_BLOCK_MODE ARM_BASELINE RUN_NAME <<< "${ARM}"
    LOG_FILE="logs/bopd/${RUN_NAME}.log"
    ARM_DESC="${LABEL}: block_size=${ARM_BLOCK_SIZE} block_mode=${ARM_BLOCK_MODE} block_baseline=${ARM_BASELINE}"

    echo ""
    echo "============================================================"
    echo "[$(date '+%F %T')] START  ${ARM_DESC}"
    echo "  run      : ${RUN_NAME}"
    echo "  log      : ${LOG_FILE}"
    echo "  ckpt     : checkpoints/distillation-${RUN_NAME}"
    echo "============================================================"

    # 上一组可能留下残余 Ray 集群，先清掉再起新的
    ray stop --force 2>/dev/null || true

    # 不用 `|| true`：那样会把退出码洗成 0，崩溃和跑完看起来一模一样。
    # 这里显式收集退出码，既保证前一组失败时后一组照常继续，又能在末尾看清结果。
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
