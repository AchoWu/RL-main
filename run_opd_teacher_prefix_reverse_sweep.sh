#!/usr/bin/env bash
# Run the four fixed-prefix experiments sequentially from the same base model.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for prefix_len in 0 256 512 1024; do
  echo "============================================================"
  echo "Running reverse-KL OPD with teacher_prefix_length=${prefix_len}"
  echo "============================================================"
  bash "${ROOT_DIR}/train_opd_skywork7b_to_r1qwen15b_teacher_prefix_reverse.sh" "$prefix_len"
done
