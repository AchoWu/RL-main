#!/usr/bin/env bash
# Matched full-token baseline with equal total KL weight per sequence.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPD_CONFIG_PATH=examples/configs/distillation_math_tvd_sequence_balanced.yaml
export OPD_RUN_TAG=matched-full-sequence-mean
exec bash "${SCRIPT_DIR}/train_opd_skywork7b_to_r1qwen15b_matched_runner.sh"
