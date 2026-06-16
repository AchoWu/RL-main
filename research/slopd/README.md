# Phase 0: SLOPD Pre-Implementation Validation

**Goal**: validate the core assumptions of SLOPD *before* writing any production
code. If any of these checks fail, the method design needs to be revisited.

These scripts are **standalone** — they do not depend on NeMo-RL internals and
can run with just a HF / vLLM install on a single GPU.

## Phase 0.1: Segmenter Robustness

**Question**: does the math-aware sentence segmenter produce sensible
segmentations on real CoT outputs from a 1.7B-scale model?

**Pass criteria**:
- 90%+ trajectories produce ≥3 sentences
- median sentence count ≥ 8 on AIME (long CoT)
- pathological sentence rate (too short/long) < 10%

### Files
- `segmenter.py` — math-aware sentence segmenter (LaTeX/decimal-protected).
  Run `python segmenter.py` to execute self-tests.
- `phase0_segmenter_check.py` — runs segmenter over Qwen3-1.7B rollouts on
  AIME 2024 and reports statistics.

### Run on the server

```bash
# From the RL-main project root
cd research/slopd

# Self-test the segmenter (no GPU needed)
python segmenter.py

# Full Phase 0.1 check (needs 1 GPU; ~20-40 min for 30 problems)
python phase0_segmenter_check.py \
    --model_path /group/40143/howu/llms/Qwen3-1.7B \
    --data_path /path/to/aime_2024.jsonl \
    --out_dir ./phase0_results \
    --max_problems 30 \
    --max_new_tokens 4096 \
    --save_rollouts ./phase0_results/rollouts.jsonl
```

### Outputs

After the run:
- `phase0_results/summary.json` — all numerical statistics + pass/fail flags
- `phase0_results/samples.txt` — 5 raw trajectories with their segmentations
  (read this manually to spot-check segmenter quality)
- `phase0_results/histograms.txt` — text-mode histograms of sentence counts
  and lengths
- `phase0_results/rollouts.jsonl` — saved generations (so re-running just
  segmenter changes does not need to re-generate)

### Iterating on the segmenter

If results show bad segmentations (e.g. LaTeX getting split, decimals breaking
sentences), edit `segmenter.py` and re-run with `--cached_rollouts` to skip
generation:

```bash
python phase0_segmenter_check.py \
    --data_path /path/to/aime_2024.jsonl \
    --cached_rollouts ./phase0_results/rollouts.jsonl \
    --out_dir ./phase0_results_v2
```

## Decision gate

After running Phase 0.1, look at `summary.json` → `pass_criteria`:

- All three PASS flags true → proceed to **Phase 0.2** (teacher candidate
  diversity check, separate script TBD).
- Pathological rate is the only failure → tweak segmenter parameters
  (`min_chars`, `max_chars`) or `_LATEX_PATTERNS`.
- Median sentence count is the only failure → AIME outputs are too short;
  may need to increase `max_new_tokens` or add a "show all steps" instruction
  to the prompt.
- 3+ sentences fraction failure is fundamental → segmenter cannot find
  reasonable boundaries; the entire SLOPD method needs to reconsider its
  segmentation strategy (e.g. switch to LLM-based segmenter).
