#!/usr/bin/env bash
# Old-baseline token-mean reverse OPD with only frozen-initial-student ref-KL.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REFERENCE_KL_BETA="${REFERENCE_KL_BETA:-0.01}"
export OPD_CONFIG_PATH=examples/configs/distillation_math_oldkl_reference_kl.yaml
export OPD_RUN_TAG="oldkl-token-mean-refkl-b${REFERENCE_KL_BETA}"
export OPD_ZERO_OUTSIDE_TOPK=false
export OPD_REFERENCE_KL_PENALTY="$REFERENCE_KL_BETA"
export OPD_REFERENCE_KL_TYPE=k3
exec bash "${SCRIPT_DIR}/train_opd_skywork7b_to_r1qwen15b_matched_runner.sh"
