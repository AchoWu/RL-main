#!/usr/bin/env bash
# Old conditional reverse KL with the original s0.05/e0.4/u1.0 TVD curriculum.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPD_CONFIG_PATH=examples/configs/distillation_math_tvd_warmup_low_e04_oldkl.yaml
export OPD_RUN_TAG=oldkl-tvdwarm-low-s0.05-e0.4-u1.0
export OPD_ZERO_OUTSIDE_TOPK=false
exec bash "${SCRIPT_DIR}/train_opd_skywork7b_to_r1qwen15b_matched_runner.sh"
