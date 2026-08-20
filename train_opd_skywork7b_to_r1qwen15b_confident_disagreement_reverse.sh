#!/usr/bin/env bash
# Reverse OPD on the top 50% teacher-confident disagreement tokens per sequence.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPD_CONFIG_PATH=examples/configs/distillation_math_tvd_confident_disagreement.yaml
export OPD_RUN_TAG=confident-disagreement-top50
exec bash "${SCRIPT_DIR}/train_opd_skywork7b_to_r1qwen15b_matched_runner.sh"
