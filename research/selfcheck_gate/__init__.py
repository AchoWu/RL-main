"""Offline utilities for the OPD teacher self-check gate.

Pipeline (see train_opd_skywork7b_to_r1qwen15b_selfcheck_gate_reverse.sh):
  1. probe_tokenizer.py    - pick single-token positive/negative pair
  2. estimate_baseline.py  - global log-ratio baseline b on neutral prefixes
  3. validate_probe_signal.py - go/no-go: inspect s distribution on rollouts
"""
