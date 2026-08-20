#!/usr/bin/env bash
# Old-baseline conditional reverse KL on the top 50% confident-disagreement tokens.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPD_CONFIG_PATH=examples/configs/distillation_math_tvd_confident_disagreement.yaml
export OPD_RUN_TAG=oldkl-confident-disagreement-top50
export OPD_ZERO_OUTSIDE_TOPK=false
exec bash "${SCRIPT_DIR}/train_opd_skywork7b_to_r1qwen15b_matched_runner.sh"
