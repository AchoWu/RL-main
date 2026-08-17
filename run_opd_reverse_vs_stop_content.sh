#!/usr/bin/env bash
# Sequential A/B run. The two child scripts share model paths, dataset split,
# seed, optimizer, 307-step budget, validation cadence, and checkpoint cadence.
# The method arm changes only the factorized stop/content objective and run name.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/2] reverse-KL baseline"
bash "${ROOT_DIR}/train_opd_skywork7b_to_r1qwen15b_reverse.sh"

echo "[2/2] reverse-content + forward-stop KL"
bash "${ROOT_DIR}/train_opd_skywork7b_to_r1qwen15b_stop_content.sh"
