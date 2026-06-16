# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NeMo RL** is an open-source, scalable, and efficient post-training library for reinforcement learning on Large Language Models (LLMs) and Vision Language Models (VLMs). It's part of the NVIDIA NeMo Framework and supports RL algorithms including GRPO, GSPO, DAPO, DPO, SFT, and On-policy Distillation across 1-1000+ GPUs.

Key design philosophy: modular abstractions for managing RL Actors (policy models, inference engines, reward environments) from single-GPU prototypes to massive distributed deployments.

## Development Setup

### Environment Setup
```bash
# Clone with submodules (critical - don't skip --recursive)
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl

# Create Python 3.12+ virtual environment using uv
uv venv
# Do NOT activate manually; use `uv run` for all commands

# If submodules weren't initialized:
git submodule update --init --recursive
```

### Common Development Commands

**Install/manage dependencies:**
```bash
uv sync                    # After pyproject.toml changes
uv sync --extra dev        # Linting & type checking
uv sync --extra mcore      # Megatron-Core backend
uv sync --extra vllm       # vLLM generation backend
NRL_FORCE_REBUILD_VENVS=true uv run python examples/run_grpo_math.py  # Force rebuild
```

**Run code:**
```bash
# ALWAYS use `uv run` (not python directly or activated venv)
uv run python examples/run_grpo_math.py
uv run python examples/run_grpo_math.py cluster.gpus_per_node=8 policy.model_name="meta-llama/Llama-3.1-8B-Instruct"
```

**Code quality:**
```bash
uv run --group dev pre-commit run --all-files  # All checks
uv run --group dev ruff check --fix nemo_rl/   # Linting
uv run --group dev ruff format nemo_rl/        # Format
uv run --group dev pyrefly check               # Type checking
```

**Testing:**
```bash
uv run pytest tests/                                          # All tests
uv run pytest tests/unit/algorithms/test_grpo.py             # Specific file
uv run pytest tests/unit/algorithms/test_grpo.py::test_name -v  # Single test
uv run pytest tests/ -k "grpo" -v                            # Pattern match
uv run pytest tests/ --cov=nemo_rl --cov-report=html         # With coverage
uv run pytest tests/unit/                                     # Unit tests only (fast)
uv run pytest -m "not mcore"                                 # Exclude markers
```

**Documentation:**
```bash
cd docs && uv run make html      # Build
cd docs && uv run make livehtml   # Live serve
```
## High-Level Architecture

### Core Design Principles

NeMo RL scales from 1 GPU to 1000+ GPUs using **composable abstractions for RL Actors**:

1. **Resourcing** (`RayVirtualCluster`): GPU/CPU allocation via Ray placement groups
2. **Isolation** (`RayWorkerGroup`): Worker processes with independent environments
3. **Coordination**: Single-process controller orchestrating RL training loop
4. **Communication**: Data via controller, NCCL collectives, or multiprocess queues

Algorithm code remains unchanged across all scales.

### Directory Structure

```
nemo_rl/
├── algorithms/          # GRPO, GSPO, DAPO, DPO, SFT, On-policy Distillation
│   ├── grpo.py          # Main GRPO (~3000 lines, full training loop)
│   ├── loss_functions.py # Token/sequence-level losses
│   ├── reward_functions.py # Reward shaping
│   └── interfaces.py    # LossFunction protocol
│
├── data/                # Datasets & loading
│   ├── datasets/        # AllTaskProcessedDataset implementations
│   ├── processors/      # math_hf_data_processor, etc.
│   └── interfaces.py    # TaskDataSpec, DatumSpec TypedDicts
│
├── distributed/         # Ray-based distribution
│   ├── virtual_cluster.py      # RayVirtualCluster
│   ├── worker_groups.py        # RayWorkerGroup
│   ├── batched_data_dict.py    # Efficient data transport
│   └── ray_actor_environment_registry.py
│
├── models/
│   ├── generation/      # vLLM, Megatron inference
│   │   ├── vllm/
│   │   ├── megatron/
│   │   └── interfaces.py # GenerationInterface
│   │
│   ├── policy/          # Training implementations
│   │   ├── lm_policy.py
│   │   ├── workers/
│   │   └── interfaces.py
│   │
│   ├── dtensor/         # PyTorch FSDP2 backend
│   └── megatron/        # Megatron-Core backend
│
├── environments/        # Reward/evaluation
│   ├── math_environment.py
│   ├── games/
│   └── tools/
│
├── evals/               # Evaluation tools
├── experience/          # Rollout collection
└── utils/               # Logging, checkpointing, config

examples/
├── configs/
│   ├── grpo_math_1B.yaml              # DTensor (reference)
│   ├── grpo_math_1B_megatron.yaml     # Megatron backend
│   ├── dpo.yaml, sft.yaml
│   └── recipes/                       # Production configs
└── run_grpo_math.py, run_dpo.py, run_sft.py
```

### Configuration System

NeMo RL uses **TypedDict + YAML + Hydra**:

- **YAML is single source of truth** for defaults
- TypedDict provides type hints (e.g., `GRPOMasterConfig`, `PolicyConfig`)
- Use `typing.NotRequired[...]` for optional keys
- **NO code defaults** (except alpha features) - define in YAML
- Exemplar configs in `examples/configs/*.yaml` document all values
- Recipe configs in `examples/configs/recipes/` are functional snapshots

**Key TypedDicts:** `MasterConfig`, `PolicyConfig`, `VllmConfig`, `GenerationConfig`, `DataConfig`

**FORBIDDEN:**
```python
grpo_config.get("num_prompts_per_step", 32)  # Use YAML
policy_config.get("model_name", "llama-1b")  # Use YAML
```

**CORRECT:**
```python
num_prompts = grpo_config["num_prompts_per_step"]
if "milestones" in scheduler_cfg:
    # Use milestones
```

See: `CODING_GUIDELINES.md` (Configuration Defaults section)

### Training Backends (auto-selected from config)

#### DTensor (PyTorch FSDP2)
- Good for: Models ~70B
- Advantages: Pure PyTorch, no weight conversion, native HF checkpoints
- Enable: `policy.dtensor_cfg.enabled=True`
- Reference: `examples/configs/grpo_math_1B.yaml`

#### Megatron
- Good for: Models >100B, 6D parallelism (TP/PP/CP/SP/EP/FSDP)
- Advantages: Optimized for massive models, seamless training→inference
- Enable: `policy.megatron_cfg.enabled=True`
- Reference: `examples/configs/grpo_math_1B_megatron.yaml`
- Note: Megatron takes precedence if both enabled

**Checkpoints:**
- Input: Always Hugging Face format
- Megatron: Converts HF→Megatron once, caches (controlled by `NRL_MEGATRON_CHECKPOINT_DIR`, `HF_HOME`, `~/.cache/huggingface/nemo_rl`)
- Output: PyTorch DCP, convert back to HF with `examples/converters/convert_dcp_to_hf.py`

See: `docs/design-docs/training-backends.md`

### Generation Backends (pluggable via `GenerationInterface`)

#### vLLM (recommended)
- Default in most examples
- High-throughput, memory-efficient
- Weight updates via IPC handles
- Config: `policy.generation.vllm_cfg`

#### Megatron
- Native Megatron inference, no weight conversion
- Seamless with Megatron training
- Config: `policy.generation.megatron_cfg`

See: `docs/design-docs/generation.md`

### Supported Algorithms

- **GRPO/GSPO**: Group Relative Policy Optimization
- **DAPO**: Decoupled Clip + Dynamic Sampling (extended GRPO with token-level loss, reward shaping)
- **DPO**: Direct Preference Optimization
- **SFT**: Supervised Fine-Tuning
- **On-policy Distillation**: Student on-policy learning from teacher via KL logit alignment

### RL Training Loop (Simplified GRPO)

```python
for batch in dataloader:
    batch.repeat_interleave(num_generations_per_prompt)  # 1
    generations = policy_generation.generate(batch)      # 2
    rewards = environment.step(generations)              # 3
    logprobs = policy.get_logprobs(generations)          # 4
    reference_logprobs = policy.get_reference_logprobs(generations)
    training_data = calculate_grpo_training_data(...)    # 5
    policy.train(generations, logprobs, training_data, loss_fn)  # 6
```

See: `nemo_rl/algorithms/grpo.py` for full implementation (checkpointing, validation, async rollouts).

### Distributed Execution

- **Controller** (main process): Orchestrates loop, manages data
- **Workers** (Ray actors): Independent processes, isolated environments
- **Communication**: Via `RayWorkerGroup.run_all_workers_multiple_data()`
- **Resources**: `RayVirtualCluster` divides GPUs into bundles; `RayWorkerGroup` creates actors on bundles
- **Colocation**: Multiple workgroups (policy + generation) share GPUs in-turn

See: `docs/design-docs/design-and-philosophy.md`

## Coding Standards & Patterns

### Style & Linting
- Python 3.12+ required
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- 4-space indentation
- `snake_case` functions/files, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- Google-style docstrings (Sphinx autodoc2)
- Pre-commit enforces: ruff format, ruff lint, pyrefly type check, TOML format

### Documentation (from CONTRIBUTING.md)
- **All new key features need docs** (design doc or guide)
- Explain motivation, approach, examples, implementation details
- Ensures user adoption and developer extensibility
- Place in `docs/guides/` or `docs/design-docs/`

### Git Workflow (from CONTRIBUTING.md)
- Sign off commits: `git commit -s` (Developer Certificate of Origin required)
- Feature branches: `git switch -c my-feature`
- PR requirements: pre-commit passes, tests pass, docs linked, no code defaults

### Configuration (from CODING_GUIDELINES.md)
1. Define TypedDict configs with field documentation
2. Exemplar YAML in `examples/configs/` with documented defaults
3. Use `typing.NotRequired[...]` for optional fields
4. Test compliance in `tests/unit/test_config_validation.py`
5. No code-level defaults for required config

### Testing
- **Unit tests** (`tests/unit/`): Fast, no GPU
- **Functional tests** (`tests/functional/`): GPU required
- **Markers**: `@pytest.mark.mcore`, `@pytest.mark.vllm`, `@pytest.mark.hf_gated`

## Environment Variables

### Essential
```bash
export HF_TOKEN=<token>                # Gated models
export HF_HOME=<path>                  # HF cache
export WANDB_API_KEY=<key>             # Weights & Biases
export HF_DATASETS_CACHE=<path>        # Dataset cache
huggingface-cli login                  # Llama models
```

### Megatron Backend
```bash
export NRL_MEGATRON_CHECKPOINT_DIR=/path/to/mcore/checkpoints
# Or HF_HOME: ~/.cache/huggingface/nemo_rl (default if HF_HOME not set)
```

### Debugging
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64  # Reduce fragmentation
export NRL_FORCE_REBUILD_VENVS=true    # Force rebuild
export NRL_CONTAINER=1                 # In container
export NRL_IGNORE_VERSION_MISMATCH=1   # Skip fingerprint check
```

## Important Files

- `pyproject.toml`: Dependencies, workspace config for submodules
- `CONTRIBUTING.md`: PR workflow, DCO requirement
- `CODING_GUIDELINES.md`: Style, config philosophy
- `.pre-commit-config.yaml`: Ruff, pyrefly, TOML, config validation
- `docs/design-docs/design-and-philosophy.md`: RL Actor abstraction, RayVirtualCluster, TypedDict patterns
- `docs/design-docs/training-backends.md`: DTensor vs Megatron, checkpoints
- `docs/design-docs/generation.md`: GenerationInterface, vLLM, Megatron inference

## Common Workflows

### Adding New Algorithm
1. Loss function: `nemo_rl/algorithms/loss_functions.py` (implement `LossFunction`)
2. Training loop: `nemo_rl/algorithms/my_algorithm.py` (follow `grpo.py` pattern)
3. Example: `examples/run_my_algorithm.py` + YAML config
4. Config TypedDicts with YAML exemplar
5. Tests: `tests/unit/algorithms/test_my_algorithm.py`
6. Docs: `docs/guides/my_algorithm.md` (motivation, approach, usage)

### Debugging Multi-GPU
1. Check `PYTORCH_CUDA_ALLOC_CONF` (memory fragmentation)
2. `uv run pytest tests/ --durations=15` (slow tests)
3. nsys profiling: `docs/nsys-profiling.md`
4. Ray logs: `/tmp/ray/session_*/logs/` or `~/.ray/session_*/logs/`

### Running Experiments
- Single GPU: `uv run python examples/run_grpo_math.py`
- Multi-GPU: `uv run python examples/run_grpo_math.py cluster.gpus_per_node=8`
- Multi-node: Use `ray.sub` SLURM script (requires SLURM + Docker)
- Custom config: `uv run python examples/run_grpo_math.py --config examples/configs/recipes/llm/grpo_qwen.yaml`

## Submodules & Dependencies

Workspace management via `uv` for submodule integration:

- `3rdparty/Megatron-LM-workspace/Megatron-LM`: Megatron training/inference
- `3rdparty/Automodel-workspace/Automodel`: Foundation for DTensor
- `3rdparty/Megatron-Bridge-workspace`: Checkpoint conversion
- `3rdparty/Gym-workspace`: RL environments (optional)

After clone, init submodules:
```bash
git submodule update --init --recursive
git config submodule.recurse true  # Auto-update after branch changes
```

After submodule changes, force rebuild:
```bash
NRL_FORCE_REBUILD_VENVS=true uv run python <script>
```

## Quick Reference

| Task | Command |
|------|---------|
| GRPO single GPU | `uv run python examples/run_grpo_math.py` |
| GRPO 8 GPUs | `uv run python examples/run_grpo_math.py cluster.gpus_per_node=8` |
| GRPO Megatron | `uv run python examples/run_grpo_math.py --config examples/configs/grpo_math_1B_megatron.yaml` |
| DPO | `uv run python examples/run_dpo.py` |
| SFT | `uv run python examples/run_sft.py` |
| Tests | `uv run pytest tests/unit/` |
| Lint | `uv run --group dev pre-commit run --all-files` |
| Type check | `uv run --group dev pyrefly check` |
| Build docs | `cd docs && uv run make html` |
| Convert checkpoint | `uv run python examples/converters/convert_dcp_to_hf.py --config <config.yaml> --dcp-ckpt-path <path> --hf-ckpt-path <out>` |
