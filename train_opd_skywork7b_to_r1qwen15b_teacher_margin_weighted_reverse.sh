#!/usr/bin/env bash
# Old-baseline conditional reverse KL with per-token teacher margin weights.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPD_CONFIG_PATH=examples/configs/distillation_math_teacher_margin_weighted.yaml
export OPD_RUN_TAG=oldkl-teacher-margin-power1
export OPD_ZERO_OUTSIDE_TOPK=false
exec bash "${SCRIPT_DIR}/train_opd_skywork7b_to_r1qwen15b_matched_runner.sh"
