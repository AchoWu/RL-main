#!/usr/bin/env bash
# DAPO-32k 复跑：把原先 4 组 reverse-OPD 消融换到更难的数据集 + 更长生成长度。
#
# 与之前 (DeepScaler / 16k / rollout=1) 的差异，其余超参一律不变：
#   - 数据集: open-r1/DAPO-Math-17k-Processed (config=en, 14116 条)
#   - 生成长度: 32768 (原 16384) —— 原先 ~48% 的样本被 16k 截断
#   - rollout:  1 (不变，理由见下方 OPD_NUM_GENERATIONS 注释)
#   - 验证集:   从训练集随机抽 500 道 (train_holdout, seed=42)，与训练集严格互斥
# 最终模型质量仍用 AIME-2024 avg@32 评测，以便与已有分数同尺度对比。
#
# 用法:
#   bash run_opd_dapo32k_sweep.sh            # 全部 4 组，串行
#   bash run_opd_dapo32k_sweep.sh seqmean-refkl tokmean-refkl   # 只跑指定组
#   OPD_MAX_NUM_STEPS=2 OPD_SMOKE=1 bash run_opd_dapo32k_sweep.sh seqmean-refkl  # smoke test
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 4 组消融: <短名>|<config>|<zero_outside_topk>|<refkl beta, 空=不启用>|<run tag>
# 顺序 = 从最朴素到最多附加项，便于先拿到 baseline 再看各项增量。
# 全部 zero_outside_topk=false，与之前 16k 那批实验保持一致（旧的 nomask-reverse
# 跑的是 base config 默认值 false，并未显式开启）。
ABLATIONS=(
  "nomask-reverse|examples/configs/distillation_math.yaml|false||dapo32k-nomask-reverse"
  "seqmean|examples/configs/distillation_math_tvd_sequence_balanced.yaml|false||dapo32k-oldkl-sequence-mean"
  "tokmean-refkl|examples/configs/distillation_math_oldkl_reference_kl.yaml|false|0.01|dapo32k-oldkl-token-mean-refkl-b0.01"
  "seqmean-refkl|examples/configs/distillation_math_sequence_balanced_reference_kl.yaml|false|0.01|dapo32k-oldkl-sequence-mean-refkl-b0.01"
)

# DAPO-32k 共同设置
export OPD_DATASET_NAME=DAPOMath17KProcessed
export OPD_SEQ_LEN="${OPD_SEQ_LEN:-32768}"
# rollout 保持 1。OPD 的 loss 是逐 token 对齐 teacher 的 top-k 分布，没有 GRPO 那种
# 组内 mean/std baseline（num_generations_per_prompt 在 distillation.py 里只用于
# repeat_interleave），所以多次采样同一题不带来额外监督信号，只降低 prompt 多样性。
# 且 num_prompts_per_step(128) * rollout 必须等于 train_global_batch_size(128) 才是
# 严格 on-policy：rollout=4 会让一批 rollout 被切成 4 次梯度更新，后 3 次是 off-policy，
# 会污染这批 ref-KL / reduction 的消融对比。
export OPD_NUM_GENERATIONS="${OPD_NUM_GENERATIONS:-1}"
export OPD_VAL_NUM_SAMPLES=500
export OPD_MAX_VAL_SAMPLES=500
# 13616 train 样本 / num_prompts_per_step=128 = 106 步跑满 1 个 epoch (drop_last=True)。
# val_period=10 -> 11 个观测点，val 总耗时约 1.9h/组（每次 500 题 ~10.5min）。
export OPD_MAX_NUM_STEPS="${OPD_MAX_NUM_STEPS:-106}"
export OPD_VAL_PERIOD="${OPD_VAL_PERIOD:-10}"
export OPD_SAVE_PERIOD="${OPD_SAVE_PERIOD:-10}"
export OPD_LEN_TAG="${OPD_LEN_TAG:-dapo32k}"
# `+` prefix: config_name is a new key not present in distillation_math.yaml,
# so Hydra requires append rather than override.
export OPD_EXTRA_OVERRIDES="+data.config_name=en"

# 选择要跑的组（默认全部）
WANTED=("$@")

for entry in "${ABLATIONS[@]}"; do
  # NOTE: the array above must NOT be named GROUPS — bash reserves that as a
  # read-only variable (the caller's group count), so the assignment is
  # silently ignored and every entry would expand to "0".
  IFS='|' read -r short cfg zerotopk beta tag <<< "$entry"

  if [[ ${#WANTED[@]} -gt 0 ]]; then
    match=0
    for w in "${WANTED[@]}"; do
      if [[ "$w" == "$short" ]]; then
        match=1
      fi
    done
    [[ $match -eq 1 ]] || continue
  fi

  echo "=============================================================="
  echo "[$(date '+%F %T')] group=$short  config=$cfg  refkl=${beta:-none}"
  echo "=============================================================="

  # 每组一个独立子 shell，避免 OPD_REFERENCE_KL_PENALTY 在组间泄漏
  (
    export OPD_CONFIG_PATH="$cfg"
    export OPD_RUN_TAG="$tag"
    export OPD_ZERO_OUTSIDE_TOPK="$zerotopk"
    if [[ -n "$beta" ]]; then
      export OPD_REFERENCE_KL_PENALTY="$beta"
      export OPD_REFERENCE_KL_TYPE=k3
    fi
    bash "${SCRIPT_DIR}/train_opd_skywork7b_to_r1qwen15b_matched_runner.sh"
  ) || {
    echo "[FAIL] group=$short exited non-zero" >&2
    [[ "${OPD_SMOKE:-0}" == "1" ]] && exit 1
  }

  echo "[done] group=$short"
done

echo "all requested groups finished"
