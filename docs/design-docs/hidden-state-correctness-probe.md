# Hidden-State Correctness Probe

This experiment tests whether a frozen student model linearly encodes the eventual
correctness of its current reasoning trajectory. It is a cheap precursor to a
recoverability-triggered teacher intervention policy.

## Protocol

1. Exclude the existing 1,000-example DeepScaleR validation holdout.
2. Select 1,000 training problems and sample two student trajectories per problem.
3. Label each complete trajectory using the existing `math-verify` evaluator.
4. Select the empty response prefix and seven positions spanning the response. Each
   position snaps to a paragraph or line boundary within 256 tokens when available.
   Prefixes after an answer marker or within 512 tokens of the end are excluded.
5. Run one frozen, teacher-forced forward pass over each complete trajectory and
   extract the last non-whitespace token state at 50%, 75%, 90%, and 100% depth.
6. Split by problem, not by prefix: 70% train, 15% validation, and 15% test.
7. Fit an L2-regularized logistic probe for every layer and select the layer with
   the highest validation ROC-AUC.

Every visited prefix inherits its trajectory's terminal correctness label. This is
a single-sample Monte Carlo target for the student's probability of eventual success.

## Run

Start with a 100-problem smoke test:

```bash
cd /group/40094/jingweidong/RL-main
NUM_PROBLEMS=100 \
OUTPUT_DIR=/group/40094/jingweidong/RL-main/outputs/hidden-probe-smoke \
bash run_hidden_state_correctness_probe.sh
```

Then run the default 1,000-problem experiment:

```bash
nohup bash run_hidden_state_correctness_probe.sh \
  > logs/hidden-state-correctness-probe.log 2>&1 &
```

The generation and extraction phases are sharded across eight GPUs. JSONL generation
and NPZ extraction manifests are resumable. The probe-training phase uses GPU 0.

## Outputs

- `trajectories.shard-*.jsonl`: generated traces, labels, and checkpoint positions.
- `hidden.shard-*.chunk-*.npz`: float16 hidden-state features.
- `extraction-manifest.shard-*.jsonl`: completed extraction chunks.
- `probe_summary.json`: layer, split, calibration, checkpoint, and temporal metrics.
- `probe_predictions.csv`: every prefix score for plotting and further analysis.
- `best_logistic_probe.pt`: probe weights and feature normalization statistics.

The first decision is based on five groups of metrics:

1. `layer_results[*].metrics.test`: held-out trajectory discrimination and calibration.
2. `test_metrics_by_checkpoint`: how early the signal becomes predictive.
3. `test_metrics_by_progress`: performance at 0%, 0-25%, 25-50%, 50-75%, and 75-100%.
4. `test_within_problem_concordance`: correct-vs-wrong discrimination for different
   trajectories sampled from the same held-out problem.
5. `test_temporal_metrics`: whether incorrect traces show larger score drops without
   causing the same trigger rate on correct traces.

High global ROC-AUC alone is insufficient. A useful intervention trigger also needs a
large gap between `incorrect_drop_ge_0.2_rate` and `correct_drop_ge_0.2_rate`.
