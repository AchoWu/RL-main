#!/usr/bin/env bash
# AIME-2024 avg@32 评测：OPD reverse-oldkl-sequence-mean step_45
set -euo pipefail

source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd

export HF_HOME=/root/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CKPT=/dev/shm/hf_ckpts/oldkl-seqmean-16k-step45
OUT=/group/40092/howu/RL-main/aime2024-opd-oldkl-seqmean-16k-step45_avg32.jsonl

cd /group/40092/howu/RL-main
python vllm_opd_aime.py \
  --model-path "$CKPT" \
  --test-file ./aime_2024.jsonl \
  --output-path "$OUT" \
  --tensor-parallel-size 1 \
  --data-parallel-size 8
