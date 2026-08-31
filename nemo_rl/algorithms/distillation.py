# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.
# limitations under the License.
import copy
import math
import os
import warnings
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import _should_use_async_rollouts, refit_policy_generation
from nemo_rl.algorithms.loss_functions import (
    DistillationLossConfig,
    DistillationLossDataDict,
    DistillationLossFn,
    OADLossFn,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class ProgressiveTeacherBlocksConfig(TypedDict):
    enabled: bool
    block_size: int
    steps_per_stage: int


class DistillationConfig(TypedDict):
    # Training configuration
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_rollout_turns: int  # for multi-turn rollouts. Math Environments just have 1 turn (answering the question)
    max_num_steps: int  # maximum number of steps to train for
    max_num_epochs: int  # maximum number of epochs to train for
    val_batch_size: int
    val_period: int
    val_at_start: bool
    max_val_samples: int
    topk_logits_k: int
    seed: int
    teacher_prefix_length: NotRequired[int]
    progressive_teacher_blocks: NotRequired[ProgressiveTeacherBlocksConfig]


class DistillationSaveState(TypedDict):
    total_steps: int  # Track total number of steps across all epochs
    current_epoch: int  # Track current epoch
    current_step: int  # Track step within current epoch
    val_reward: NotRequired[
        float
    ]  # Can be any metric. Setted to 'accuracy' by default in validation.
    consumed_samples: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training


def _default_distillation_save_state() -> DistillationSaveState:
    return {
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "val_reward": -99999999.0,  # Aligned with GRPO
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


def _resolve_progressive_teacher_block(
    distillation_config: DistillationConfig, global_step: int
) -> tuple[int, Optional[int], int]:
    """Return teacher-prefix length, student block cap, and curriculum stage."""
    progressive_cfg = distillation_config.get("progressive_teacher_blocks")
    if not progressive_cfg or not progressive_cfg.get("enabled", False):
        return int(distillation_config.get("teacher_prefix_length", 0)), None, 0

    block_size = int(progressive_cfg["block_size"])
    steps_per_stage = int(progressive_cfg["steps_per_stage"])
    if block_size <= 0:
        raise ValueError("progressive_teacher_blocks.block_size must be greater than 0")
    if steps_per_stage <= 0:
        raise ValueError(
            "progressive_teacher_blocks.steps_per_stage must be greater than 0"
        )
    if global_step < 0:
        raise ValueError("global_step must be greater than or equal to 0")

    stage = global_step // steps_per_stage
    return stage * block_size, block_size, stage


def _resolve_tvd_gate_threshold(
    gate_cfg: Optional[dict],
    global_step: int,
    max_num_steps: int,
) -> tuple[str, float]:
    """Return (mode, threshold) for the TVD gate at the current step.

    - mode="none"   ⇒ threshold=0.0 (gate is a no-op; callers should skip work)
    - mode="fixed"  ⇒ constant scalar from cfg["threshold"]
    - mode="warmup" ⇒ S-shaped cosine anneal from start_threshold at step 0
      to end_threshold at global_step / max_num_steps == warmup_until_frac,
      then stays at end_threshold. Slope is 0 at both endpoints — no jump
      when the gate "opens up" at the end of warmup.
    - mode="top_fraction" ⇒ the returned scalar is the fixed per-sequence
      keep fraction used by score-based ranking inside DistillationLossFn.

    Config-shape validation (required keys per mode, valid mode names) lives
    in `DistillationLossFn.__init__` — this resolver assumes it's been
    handed a well-formed config and uses direct dict access. Kept as a pure
    function (no torch dep) so it can be unit-tested standalone.
    """
    if gate_cfg is None:
        return "none", 0.0
    mode = gate_cfg.get("mode", "none")
    if mode == "none":
        return "none", 0.0
    if mode == "fixed":
        tau = float(gate_cfg["threshold"])
        return "fixed", max(min(tau, 1.0), 0.0)
    if mode == "warmup":
        start = float(gate_cfg["start_threshold"])
        end = float(gate_cfg["end_threshold"])
        until_frac = float(gate_cfg["warmup_until_frac"])
        start = max(min(start, 1.0), 0.0)
        end = max(min(end, 1.0), 0.0)
        until_frac = max(min(until_frac, 1.0), 0.0)
        if until_frac <= 0.0:
            return "warmup", end
        frac = 0.0 if max_num_steps <= 0 else min(global_step / max_num_steps, 1.0)
        progress = min(frac / until_frac, 1.0)
        curve = 0.5 * (1.0 - math.cos(math.pi * progress))
        tau = start + (end - start) * curve
        return "warmup", max(min(tau, 1.0), 0.0)
    if mode == "top_fraction":
        keep_fraction = float(gate_cfg["keep_fraction"])
        return "top_fraction", max(min(keep_fraction, 1.0), 0.0)
    raise ValueError(f"Unknown tvd_gate mode: {mode!r}")


class MasterConfig(TypedDict):
    """Main configuration structure."""

    policy: PolicyConfig  # Student model configuration
    teacher: PolicyConfig  # Teacher model configuration
    loss_fn: DistillationLossConfig  # Loss function configuration
    env: dict[str, Any]  # Environment configuration
    data: DataConfig  # Data configuration
    distillation: DistillationConfig  # Distillation configuration
    logger: LoggerConfig  # Logger configuration
    cluster: ClusterConfig  # Cluster configuration
    checkpointing: CheckpointingConfig  # Checkpointing configuration


# ===============================================================================
# Setup & Initialization
# ===============================================================================
def check_vocab_equality(
    tokenizer: TokenizerType, student_model_name: str, teacher_model_name: str
) -> None:
    """Check if the vocab of the tokenizer (student) and the teacher tokenizer are equal."""
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

    skip_hint = "Set NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true to skip this check."

    # 1) Exact token->id mapping equality
    vocab_a = tokenizer.get_vocab()
    vocab_b = teacher_tokenizer.get_vocab()
    assert vocab_a == vocab_b, (
        f"Token->ID mapping differs between student and teacher. {skip_hint}"
    )

    # 2) Size consistency (sanity checks)
    assert len(tokenizer) == len(teacher_tokenizer), (
        f"Effective vocab sizes differ between student and teacher. {skip_hint}"
    )

    # 3) Chech model.config.vocab_size to guarantee the last dimension of the logits is the same
    student_config = AutoConfig.from_pretrained(student_model_name)
    teacher_config = AutoConfig.from_pretrained(teacher_model_name)
    assert student_config.vocab_size == teacher_config.vocab_size, (
        f"Model config vocab sizes differ between student and teacher. {skip_hint}"
    )


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,  # student_policy
    ColocatablePolicyInterface,  # teacher_policy
    Optional[GenerationInterface],  # student_generation
    Optional[GenerationInterface],  # teacher_generation
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    DistillationLossFn | OADLossFn,
    Logger,
    CheckpointManager,
    DistillationSaveState,
    MasterConfig,
]:
    """Main entry point for distillation algorithm.

    Returns:
        tuple of student_policy, teacher_policy, student_generation, teacher_generation,
        train_dataloader, val_dataloader,
        loss_fn, logger, checkpointer, distillation_save_state, master_config
    """
    # Extract configuration
    policy_config = master_config["policy"]
    teacher_config = master_config["teacher"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    distillation_config = master_config["distillation"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for distillation"
    )

    teacher_prefix_length = int(distillation_config.get("teacher_prefix_length", 0))
    if teacher_prefix_length < 0:
        raise ValueError(
            "distillation.teacher_prefix_length must be greater than or equal to 0"
        )
    progressive_cfg = distillation_config.get("progressive_teacher_blocks")
    progressive_enabled = bool(
        progressive_cfg is not None and progressive_cfg.get("enabled", False)
    )
    max_teacher_prefix_length = teacher_prefix_length
    if progressive_enabled:
        assert progressive_cfg is not None
        if teacher_prefix_length != 0:
            raise ValueError(
                "teacher_prefix_length must be 0 when progressive_teacher_blocks "
                "is enabled"
            )
        _, block_size, _ = _resolve_progressive_teacher_block(
            distillation_config, global_step=0
        )
        assert block_size is not None
        steps_per_stage = int(progressive_cfg["steps_per_stage"])
        max_stage = max(0, (distillation_config["max_num_steps"] - 1) // steps_per_stage)
        max_teacher_prefix_length = max_stage * block_size
        if generation_config.get("vllm_cfg", {}).get("async_engine", False):
            raise ValueError(
                "progressive_teacher_blocks currently requires synchronous vLLM "
                "generation"
            )

    if max_teacher_prefix_length > 0:
        if generation_config["backend"] != "vllm":
            raise ValueError("Teacher-prefix OPD currently requires the vLLM backend")
        if not generation_config["colocated"]["enabled"]:
            raise ValueError(
                "Teacher-prefix OPD currently requires colocated vLLM generation"
            )

    # Disallow SP + packing for dtensor path
    for cfg, who in ((policy_config, "student"), (teacher_config, "teacher")):
        # DTensor sequence parallel is supported; ensure CP and SP are not enabled together
        # This incompatibility is enforced in DTensor workers during initialization.
        # Additionally, SP may not be compatible with sequence packing for some models.
        # Refer to https://github.com/NVIDIA-NeMo/RL/issues/1178 for more details.
        # Therefore, we disable SP + packing for distillation.
        dtensor_enabled = cfg["dtensor_cfg"]["enabled"]
        sequence_packing_enabled = (
            "sequence_packing" in cfg and cfg["sequence_packing"]["enabled"]
        )
        sequence_parallel_enabled = (
            "sequence_parallel" in cfg["dtensor_cfg"]
            and cfg["dtensor_cfg"]["sequence_parallel"]
        )

        if dtensor_enabled and sequence_packing_enabled and sequence_parallel_enabled:
            raise AssertionError(
                f"Distillation does not support DTensor sequence parallel + sequence packing ({who} policy). "
                "Please refer to https://github.com/NVIDIA-NeMo/RL/issues/1178 for more details."
            )

    # Set random seed
    set_seed(distillation_config["seed"])

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    distillation_save_state: Optional[DistillationSaveState] = cast(
        Optional[DistillationSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if distillation_save_state is None:
        distillation_save_state = _default_distillation_save_state()

    # ==========================
    #           Data
    # ==========================
    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=distillation_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
    )

    if last_checkpoint_path:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(
        f"  ✓ Training dataloader loaded with {len(train_dataset)} samples", flush=True
    )

    # Load validation dataset if provided
    val_dataloader: Optional[StatefulDataLoader] = None
    # If validation is enabled, load the validation dataloader
    if distillation_config["val_period"] > 0 or distillation_config["val_at_start"]:
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distillation_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )
        print(
            f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples",
            flush=True,
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...", flush=True)
    colocated_inference = generation_config["colocated"]["enabled"]

    if colocated_inference:
        cluster = RayVirtualCluster(
            name="distillation_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
            * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=(
                1
                if generation_config["backend"] == "megatron"
                else (4 if max_teacher_prefix_length > 0 else 3)
            ),
        )
        train_cluster = cluster
        inference_cluster = cluster
        print(
            f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes",
            flush=True,
        )
    else:
        assert generation_config["backend"] != "megatron", (
            "Non-colocated inference is not supported for Megatron generation backends. "
            "Please use vLLM backend for generation."
        )

        # train resources will be updated through overall and inference resources below
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        # validate and configure resources
        if cluster_config["num_nodes"] == 1:
            assert (
                inference_gpus_per_node is not None and inference_gpus_per_node > 0
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set to a value > 0 "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.colocated.resources.num_nodes must be 1 or set to null "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            inference_nodes = 1
            train_gpus_per_node -= inference_gpus_per_node
        else:
            assert inference_nodes > 0, (
                "policy.generation.colocated.resources.num_nodes must be > 0 "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is not None
                and inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set and equal to cluster.gpus_per_node "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got inference_gpus_per_node={inference_gpus_per_node}, cluster.gpus_per_node={cluster_config['gpus_per_node']}."
            )
            train_nodes -= inference_nodes

        # create clusters
        train_cluster = RayVirtualCluster(
            name="distillation_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        inference_cluster = RayVirtualCluster(
            name="distillation_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        print(
            f"  ✓ Separate clusters created: train={train_nodes}x{train_gpus_per_node}GPUs, inference={inference_nodes}x{inference_gpus_per_node}GPUs",
            flush=True,
        )

    # ==========================
    #      Teacher Policy
    # ==========================
    print("\n▶ Setting up teacher policy...", flush=True)
    # Checkpoint paths
    weights_path = None
    optimizer_path = None

    if not bool(os.getenv("NRL_SKIP_DISTILLATION_TOKENIZER_CHECK", False)):
        check_vocab_equality(
            tokenizer, policy_config["model_name"], teacher_config["model_name"]
        )

    if "megatron_cfg" in teacher_config and teacher_config["megatron_cfg"]["enabled"]:
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        teacher_config["megatron_cfg"]["train_iters"] = total_train_iters

    teacher_policy = Policy(
        name_prefix="teacher",
        cluster=train_cluster,
        config=teacher_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=False,
        init_reference_model=False,
    )
    teacher_policy.offload_after_refit()

    # ==========================
    #    Student Generation Interface
    # ==========================
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "megatron":
        student_generation = None
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        if "vllm_cfg" in generation_config:
            ## make vllm hf overrides match the training policy
            generation_config["vllm_cfg"]["hf_overrides"] = policy_config.get(
                "hf_config_overrides", {}
            )
        student_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        student_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}",
            flush=True,
        )

    # ==========================
    #    Teacher Prefix Generation
    # ==========================
    teacher_generation: Optional[GenerationInterface] = None
    if max_teacher_prefix_length > 0:
        teacher_generation_config = copy.deepcopy(teacher_config["generation"])
        assert teacher_generation_config is not None
        teacher_generation_config["model_name"] = teacher_config["model_name"]
        teacher_generation_config["max_new_tokens"] = max_teacher_prefix_length
        teacher_generation_config = configure_generation_config(
            teacher_generation_config, tokenizer, is_eval=True
        )
        if "vllm_cfg" in teacher_generation_config:
            teacher_generation_config["vllm_cfg"]["hf_overrides"] = (
                teacher_config.get("hf_config_overrides", {})
            )
        teacher_generation = VllmGeneration(
            cluster=inference_cluster,
            config=cast(VllmConfig, teacher_generation_config),
            name_prefix="teacher_prefix_vllm",
        )
        teacher_generation.finish_generation()
        print(
            "  ✓ Using teacher vLLM for teacher-prefix generation "
            f"(max {max_teacher_prefix_length} tokens)",
            flush=True,
        )

    # ==========================
    #      Student Policy
    # ==========================
    print("\n▶ Setting up student policy...", flush=True)

    # Checkpoint paths
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

    ema_anchor_cfg = loss_config.get("ema_anchor") or {}
    ema_anchor_enabled = bool(ema_anchor_cfg.get("enabled", False))
    ema_anchor_mu = float(ema_anchor_cfg.get("mu", 0.999))
    ema_anchor_kl_weight = float(ema_anchor_cfg.get("kl_weight", 0.0))

    if ema_anchor_enabled:
        # EMA anchor plumbing (update_reference_ema / get_reference_topk_logits)
        # is currently only implemented on DTensorPolicyWorkerV2. Fail fast
        # here rather than 20+ minutes into training with a Ray AttributeError.
        student_dtensor_cfg = policy_config.get("dtensor_cfg", {})
        student_uses_v2 = bool(
            student_dtensor_cfg.get("enabled", False)
            and student_dtensor_cfg.get("_v2", False)
        )
        student_uses_megatron = bool(
            policy_config.get("megatron_cfg", {}).get("enabled", False)
        )
        if student_uses_megatron or not student_uses_v2:
            raise ValueError(
                "loss_fn.ema_anchor.enabled=true requires the student policy "
                "to run on DTensorPolicyWorkerV2 "
                "(policy.dtensor_cfg.enabled=true and policy.dtensor_cfg._v2=true). "
                "The EMA worker methods are not implemented on the v1 DTensor "
                "worker or the Megatron worker."
            )

    student_policy = Policy(
        name_prefix="student",
        cluster=train_cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=(
            float(loss_config.get("reference_policy_kl_penalty", 0.0)) > 0.0
            or ema_anchor_enabled
        ),
    )

    if student_generation is not None:
        state_dict_info = student_policy.prepare_refit_info()
        student_generation.prepare_refit_info(state_dict_info)

    # if it is not colocated inference, initialize collective communication for update weights
    if not colocated_inference:
        ip, port = train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication", flush=True)
        train_world_size = train_cluster.world_size()
        # inference cluster + head node of the train cluster
        world_size = train_world_size + inference_nodes * inference_gpus_per_node
        # init collective
        futures_train = student_policy.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        futures_inference = student_generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )  # type: ignore
        # wait for all futures to complete
        ray.get(futures_train + futures_inference)

    # Dispatch loss function: default "kl" (back-compat); "oad" enables
    # Overlap-Aligned Distillation (see BASIC_OAD_PROPOSAL.md).
    loss_type = loss_config.get("type", "kl")
    if loss_type == "kl":
        loss_fn = DistillationLossFn(loss_config)
    elif loss_type == "oad":
        loss_fn = OADLossFn(loss_config.get("oad", {}))
    else:
        raise ValueError(
            f"Unknown loss_fn.type: {loss_type!r}. Expected one of: 'kl', 'oad'."
        )

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n", flush=True)

    return (
        student_policy,
        teacher_policy,
        student_generation,
        teacher_generation,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_save_state,
        master_config,
    )


# ===============================================================================
# Training & Validation
# ===============================================================================


def _generate_teacher_prefixes(
    teacher_generation: GenerationInterface,
    batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    requested_length: int,
) -> tuple[BatchedDataDict[DatumSpec], dict[str, float]]:
    """Append teacher-generated assistant prefixes without environment interaction."""
    flat_messages, input_lengths = batched_message_log_to_flat_message(
        batch["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
    )
    generation_input = BatchedDataDict[GenerationDatumSpec](
        {
            "input_ids": flat_messages["token_ids"],
            "input_lengths": input_lengths,
            "stop_strings": batch.get("stop_strings", [None] * batch.size),
            "max_new_tokens": torch.full(
                (batch.size,), requested_length, dtype=torch.long
            ),
        }
    )
    generation_input.update(flat_messages.get_multimodal_dict(as_tensors=False))
    if "vllm_content" in batch:
        generation_input["vllm_content"] = batch["vllm_content"]
    if "vllm_images" in batch:
        generation_input["vllm_images"] = batch["vllm_images"]
    if "vllm_videos" in batch:
        generation_input["vllm_videos"] = batch["vllm_videos"]

    outputs = teacher_generation.generate(generation_input, greedy=False)
    prefix_ids: list[torch.Tensor] = []
    for i, input_length in enumerate(input_lengths.tolist()):
        total_length = int(outputs["unpadded_sequence_lengths"][i].item())
        prefix_ids.append(outputs["output_ids"][i, input_length:total_length])

    prefix_texts = tokenizer.batch_decode(prefix_ids, skip_special_tokens=True)
    for message_log, text, token_ids in zip(
        batch["message_log"], prefix_texts, prefix_ids
    ):
        message_log.append(
            {
                "role": "assistant",
                "content": text,
                "token_ids": token_ids,
                # This prefix is context only. The first student suffix token remains
                # unmasked because masks are aligned to target token ownership.
                "token_loss_mask": torch.zeros_like(token_ids),
            }
        )

    actual_lengths = torch.tensor([len(ids) for ids in prefix_ids], dtype=torch.float32)
    reached_requested = actual_lengths >= requested_length
    eos_token_id = tokenizer.eos_token_id
    ended_with_eos = torch.tensor(
        [
            bool(len(ids) > 0 and eos_token_id is not None and ids[-1] == eos_token_id)
            for ids in prefix_ids
        ],
        dtype=torch.float32,
    )
    metrics = {
        "teacher_prefix_requested_tokens": float(requested_length),
        "teacher_prefix_mean_tokens": actual_lengths.mean().item(),
        "teacher_prefix_reached_requested_rate": reached_requested.float()
        .mean()
        .item(),
        "teacher_prefix_ended_with_eos_rate": ended_with_eos.mean().item(),
    }
    return batch, metrics


def _add_distillation_loss_masks(message_logs: list[list[dict[str, Any]]]) -> None:
    """Mask context and teacher prefixes while training on student targets."""
    for message_log in message_logs:
        for message in message_log:
            if "token_loss_mask" in message:
                continue
            if message["role"] == "assistant":
                message["token_loss_mask"] = torch.ones_like(message["token_ids"])
            else:
                message["token_loss_mask"] = torch.zeros_like(message["token_ids"])


def distillation_train(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    student_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: DistillationLossFn | OADLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: DistillationSaveState,
    master_config: MasterConfig,
    teacher_generation: Optional[GenerationInterface] = None,
) -> None:
    """Run Distillation training algorithm."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    # If student_generation is None, use the student_policy as the generation interface (megatron framework backend)
    if student_generation is None:
        student_generation = student_policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert student_generation is not None  # for mypy type check

    # common config/state items
    current_epoch = distillation_save_state["current_epoch"]  # current epoch
    current_step = distillation_save_state[
        "current_step"
    ]  # current step within current epoch
    total_steps = distillation_save_state[
        "total_steps"
    ]  # total number of steps across all epochs
    consumed_samples = distillation_save_state["consumed_samples"]
    total_valid_tokens = distillation_save_state["total_valid_tokens"]
    val_period = master_config["distillation"]["val_period"]
    val_at_start = master_config["distillation"]["val_at_start"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    max_epochs = master_config["distillation"][
        "max_num_epochs"
    ]  # max number of epochs to train for
    max_steps = master_config["distillation"][
        "max_num_steps"
    ]  # max number of steps to train for

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...", flush=True)
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(
                student_policy, student_generation, colocated_inference
            )
            POLICY_GENERATION_STALE = False
        else:
            student_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            student_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=total_steps,
            master_config=master_config,
            logger=logger,
        )
        student_generation.finish_generation()
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    # Run distillation training (multi-epoch until reaching max_num_steps or max_num_epochs)
    batch: BatchedDataDict[DatumSpec]

    while total_steps < max_steps and current_epoch < max_epochs:
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}",
            flush=True,
        )

        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(student_policy, total_steps + 1)
            if student_policy != student_generation:
                maybe_gpu_profile_step(student_generation, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # Prepare batch
                print("▶ Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    # Repeat batch items
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(
                            master_config["distillation"]["num_generations_per_prompt"]
                        )
                    )

                # Generate responses - this updates the LLMMessageLogType in repeated_batch
                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            student_policy,
                            student_generation,
                            colocated_inference,
                            timer=timer,
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()

                (
                    current_teacher_prefix_length,
                    student_block_size,
                    curriculum_stage,
                ) = _resolve_progressive_teacher_block(
                    master_config["distillation"], global_step=total_steps
                )
                progressive_cfg = master_config["distillation"].get(
                    "progressive_teacher_blocks"
                )
                progressive_enabled = bool(
                    progressive_cfg is not None
                    and progressive_cfg.get("enabled", False)
                )
                teacher_prefix_metrics: dict[str, float] = {}
                if progressive_enabled:
                    assert student_block_size is not None
                    teacher_prefix_metrics = {
                        "curriculum_stage": float(curriculum_stage),
                        "curriculum_teacher_prefix_tokens": float(
                            current_teacher_prefix_length
                        ),
                        "curriculum_student_block_tokens": float(student_block_size),
                        "teacher_prefix_requested_tokens": float(
                            current_teacher_prefix_length
                        ),
                        "teacher_prefix_mean_tokens": 0.0,
                        "teacher_prefix_reached_requested_rate": 1.0,
                        "teacher_prefix_ended_with_eos_rate": 0.0,
                    }
                if (
                    teacher_generation is not None
                    and current_teacher_prefix_length > 0
                ):
                    # Student refit above leaves the training policy offloaded. Sleep
                    # student vLLM while the fixed teacher vLLM owns the GPU memory.
                    student_generation.finish_generation()
                    with timer.time("teacher_prefix_generation"):
                        teacher_generation.prepare_for_generation()
                        try:
                            repeated_batch, generated_prefix_metrics = (
                                _generate_teacher_prefixes(
                                    teacher_generation,
                                    repeated_batch,
                                    tokenizer,
                                    requested_length=current_teacher_prefix_length,
                                )
                            )
                            teacher_prefix_metrics.update(generated_prefix_metrics)
                        finally:
                            teacher_generation.finish_generation()
                    student_generation.prepare_for_generation()

                with timer.time("generation"):
                    # Use async rollouts if vLLM async engine is enabled
                    if _should_use_async_rollouts(master_config):
                        (
                            repeated_batch,
                            rollout_metrics,
                        ) = run_async_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["distillation"][
                                "max_rollout_turns"
                            ],
                            greedy=False,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["distillation"][
                                "max_rollout_turns"
                            ],
                            greedy=False,
                            max_new_tokens_per_turn=student_block_size,
                        )
                    student_generation.finish_generation()

                with timer.time("data_processing"):
                    # Explicit masks on teacher prefixes are preserved; newly
                    # generated student assistant targets are unmasked.
                    _add_distillation_loss_masks(repeated_batch["message_log"])

                    # Convert updated LLMMessageLogType to FlatMessagesType for training
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    # Create training data from flattened messages
                    train_data = BatchedDataDict[DistillationLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    # this will be mini-batched inside the policy, so maintain the packed multimodal structure
                    train_data.update(
                        flat_messages.get_multimodal_dict(as_tensors=False)
                    )
                    train_data.to("cpu")

                reference_kl_enabled = (
                    isinstance(loss_fn, DistillationLossFn)
                    and loss_fn.reference_policy_kl_penalty > 0.0
                )
                ema_anchor_enabled_run = (
                    isinstance(loss_fn, DistillationLossFn)
                    and getattr(loss_fn, "ema_anchor_enabled", False)
                    and float(getattr(loss_fn, "ema_anchor_kl_weight", 0.0)) > 0.0
                )
                if reference_kl_enabled or ema_anchor_enabled_run:
                    print(
                        "▶ Preparing student for reference/EMA inference...",
                        flush=True,
                    )
                    with timer.time("reference_logprob_inference_prep"):
                        student_policy.prepare_for_lp_inference()
                    try:
                        if reference_kl_enabled:
                            print("▶ Computing reference logprobs...", flush=True)
                            with timer.time("reference_logprob_inference"):
                                reference_output = (
                                    student_policy.get_reference_policy_logprobs(train_data)
                                )
                                train_data["reference_policy_logprobs"] = reference_output[
                                    "reference_logprobs"
                                ]
                        if ema_anchor_enabled_run:
                            print("▶ Computing EMA top-k logits...", flush=True)
                            with timer.time("ema_anchor_topk_inference"):
                                ema_topk = student_policy.get_reference_topk_logits(
                                    train_data,
                                    k=master_config["distillation"]["topk_logits_k"],
                                )
                                train_data["ema_topk_logits"] = ema_topk["topk_logits"]
                                train_data["ema_topk_indices"] = ema_topk["topk_indices"]
                    finally:
                        # Teacher and student share the training cluster. Release
                        # the student weights before loading the teacher.
                        student_policy.offload_after_refit()

                print("▶ Preparing for teacher logprob inference...", flush=True)
                with timer.time("teacher_logprob_inference_prep"):
                    teacher_policy.prepare_for_lp_inference()

                print("▶ Computing teacher logprobs...", flush=True)
                with timer.time("teacher_logprob_inference"):
                    teacher_topk = teacher_policy.get_topk_logits(
                        train_data, k=master_config["distillation"]["topk_logits_k"]
                    )
                    train_data["teacher_topk_logits"] = teacher_topk["topk_logits"]
                    train_data["teacher_topk_indices"] = teacher_topk["topk_indices"]
                    # Path B for OAD: workers that compute exact full-vocab logsumexp
                    # expose it as `logsumexp`. We pass it through if present so
                    # OADLossFn can avoid the Path A approximation. Workers that
                    # don't populate it (e.g. current Megatron path) will simply
                    # not set this key, and OADLossFn will surface a clear error.
                    if "logsumexp" in teacher_topk:
                        train_data["teacher_logsumexp"] = teacher_topk["logsumexp"]
                    if "eos_logits" in teacher_topk:
                        train_data["teacher_eos_logits"] = teacher_topk["eos_logits"]

                print("▶ Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    teacher_policy.offload_after_refit()
                    student_policy.prepare_for_training()  # set model train and reload optim to GPU
                    POLICY_GENERATION_STALE = True

                print("▶ Training policy...", flush=True)
                tvd_gate_threshold_current: Optional[float] = None
                tvd_gate_target_keep_frac: Optional[float] = None
                with timer.time("policy_training"):
                    # Stamp the TVD gate state on the loss function BEFORE handing
                    # it to Ray workers. `loss_fn` is picklable and shipped as a
                    # common kwarg every step, so state stamped here is what each
                    # worker sees for this step. Skip the stamp entirely when the
                    # loss has no gate config so baseline runs' pickled loss image
                    # stays byte-identical to before this feature.
                    if isinstance(loss_fn, DistillationLossFn) and getattr(
                        loss_fn, "tvd_gate_cfg", None
                    ) is not None:
                        gate_cfg = loss_fn.tvd_gate_cfg
                        mode, tau = _resolve_tvd_gate_threshold(
                            gate_cfg,
                            global_step=total_steps,
                            max_num_steps=master_config["distillation"][
                                "max_num_steps"
                            ],
                        )
                        loss_fn._tvd_gate_state = {
                            "tau": tau if mode != "none" else float("-inf"),
                            "mode": mode,
                        }
                        # Log τ here — NOT via the loss.metrics pipeline. The
                        # worker unconditionally divides every loss-fn metric by
                        # num_global_batches (dtensor_policy_worker.py:804), which
                        # would silently rescale τ. Threshold is a per-step
                        # scalar, not a per-microbatch sum, so it belongs to the
                        # outer training-loop metrics dict.
                        if mode == "top_fraction":
                            tvd_gate_target_keep_frac = tau
                        elif mode != "none":
                            tvd_gate_threshold_current = tau
                    # nemo_rl/models/policy/workers/dtensor_policy_worker.py 506
                    train_results = student_policy.train(train_data, loss_fn)

                # EMA-of-student: after the optimizer step, blend the just-updated
                # weights into the reference (EMA) buffer. Cheap CPU-side mix; the
                # buffer is used as the top-k source in the NEXT step's loss.
                if ema_anchor_enabled_run:
                    with timer.time("ema_anchor_update"):
                        student_policy.update_reference_ema(
                            mu=float(loss_fn.ema_anchor_mu)
                        )

                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                # Run periodic validation and always validate the final model.
                should_validate = val_period > 0 and (
                    (total_steps + 1) % val_period == 0 or is_last_step
                )
                if should_validate:
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            student_policy, student_generation, colocated_inference
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        student_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                        logger=logger,
                    )
                    student_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "mean_prompt_length": repeated_batch["length"].numpy(),
                    "total_num_tokens": input_lengths.numpy(),
                }
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {
                        "lr",
                        "wd",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    else:
                        # IMPORTANT: dtensor_policy_worker_v2.py:863 already
                        # pre-divides every loss-fn metric by num_global_batches
                        # before appending to `all_mb_metrics`, so summing the
                        # microbatch values here recovers the true cross-mb mean.
                        # Using np.mean instead would compound that pre-division
                        # and shrink probability metrics by ~num_microbatches.
                        metrics[k] = np.sum(v).item()

                # TVD gate diagnostics.
                # Loss emits base/kept/tvd as raw sums. After the worker's
                # pre-divide + this loop's np.sum they represent true totals
                # scaled by 1 (perfectly recovered) for the SAME num_global_batches
                # (which is why we don't report the absolute counts — the pipeline
                # only preserves ratios, not integers). Ratios are invariant.
                #
                # Threshold is logged directly from the outer stamp (bypasses
                # the pre-divide pipeline entirely).
                if tvd_gate_threshold_current is not None:
                    metrics["tvd_gate_threshold_current"] = tvd_gate_threshold_current
                if tvd_gate_target_keep_frac is not None:
                    metrics["tvd_gate_target_keep_frac"] = tvd_gate_target_keep_frac
                if "tvd_gate_base_tokens_sum" in metrics:
                    _base = metrics.pop("tvd_gate_base_tokens_sum")
                    _kept = metrics.pop("tvd_gate_kept_tokens_sum")
                    _tvd_sum = metrics.pop("tvd_topk_sum")
                    if _base > 0:
                        metrics["tvd_gate_kept_frac"] = _kept / _base
                        metrics["tvd_topk_mean"] = _tvd_sum / _base
                        if "tvd_teacher_margin_sum" in metrics:
                            metrics["tvd_teacher_margin_mean"] = metrics.pop(
                                "tvd_teacher_margin_sum"
                            ) / _base
                            metrics["tvd_confident_disagreement_mean"] = metrics.pop(
                                "tvd_confident_disagreement_sum"
                            ) / _base
                            metrics[
                                "tvd_selected_confident_disagreement_mean"
                            ] = metrics.pop(
                                "tvd_selected_confident_disagreement_sum"
                            ) / max(_kept, 1.0)
                    else:
                        # Schema-stable NaN so downstream analysis pipelines
                        # don't see the metric silently disappear on rare
                        # empty-mask steps.
                        metrics["tvd_gate_kept_frac"] = float("nan")
                        metrics["tvd_topk_mean"] = float("nan")
                        if "tvd_teacher_margin_sum" in metrics:
                            metrics.pop("tvd_teacher_margin_sum")
                            metrics.pop("tvd_confident_disagreement_sum")
                            metrics.pop("tvd_selected_confident_disagreement_sum")
                            metrics["tvd_teacher_margin_mean"] = float("nan")
                            metrics["tvd_confident_disagreement_mean"] = float("nan")
                            metrics[
                                "tvd_selected_confident_disagreement_mean"
                            ] = float("nan")
                if "teacher_margin_base_tokens_sum" in metrics:
                    _base = metrics.pop("teacher_margin_base_tokens_sum")
                    _margin_sum = metrics.pop("teacher_margin_sum")
                    _weight_sum = metrics.pop("teacher_margin_weight_sum")
                    _weight_sq_sum = metrics.pop("teacher_margin_weight_sq_sum")
                    if _base > 0:
                        metrics["teacher_margin_mean"] = _margin_sum / _base
                        metrics["teacher_margin_weight_mean"] = _weight_sum / _base
                        metrics["teacher_margin_weight_rms"] = math.sqrt(
                            _weight_sq_sum / _base
                        )
                        metrics["teacher_margin_weight_ess_frac"] = (
                            _weight_sum * _weight_sum
                        ) / max(_base * _weight_sq_sum, 1.0e-12)
                    else:
                        metrics["teacher_margin_mean"] = float("nan")
                        metrics["teacher_margin_weight_mean"] = float("nan")
                        metrics["teacher_margin_weight_rms"] = float("nan")
                        metrics["teacher_margin_weight_ess_frac"] = float("nan")
                metrics.update(rollout_metrics)
                metrics.update(teacher_prefix_metrics)
                # EMA anchor constants: log outside the loss-fn metrics pipeline
                # so they don't get sliced by num_global_batches.
                if ema_anchor_enabled_run:
                    metrics["ema_anchor_mu"] = float(loss_fn.ema_anchor_mu)
                    metrics["ema_anchor_kl_weight"] = float(
                        loss_fn.ema_anchor_kl_weight
                    )
                total_valid_tokens += metrics["global_valid_toks"]

                ## Checkpointing
                consumed_samples += master_config["distillation"][
                    "num_prompts_per_step"
                ]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                # +1 because total_steps is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    student_policy.prepare_for_training()

                    distillation_save_state["current_epoch"] = current_epoch
                    distillation_save_state["current_step"] = current_step + 1
                    distillation_save_state["total_steps"] = total_steps + 1
                    distillation_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        distillation_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in distillation_save_state:
                        del distillation_save_state["val_reward"]
                    distillation_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                            f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                            f" e.g. 'val_reward --> 'val:accuracy'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in distillation_save_state:
                                del distillation_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            distillation_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, distillation_save_state, master_config
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            # Log training data
            log_data = {"content": flat_messages["content"]}
            log_data["input_lengths"] = input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps + 1}.jsonl"
            )

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore

            print("\n📊 Training Results:")

            print(f"  • Loss: {metrics['loss']:.4f}")
            print(
                f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
            )

            # OAD-specific metrics (only present when loss_fn.type=oad).
            # OADLossFn returns per-mb *sums* + a token count rather than
            # already-averaged values, so the final average is computed here
            # from sum / count. This is invariant to the worker's mb / dp /
            # global-batch aggregation and shows real [0, 1] probabilities.
            if "oad_token_count" in metrics:
                token_count = metrics["oad_token_count"]
                if token_count > 0:
                    acc = metrics["oad_acceptance_sum"] / token_count
                    teacher_mass = (
                        metrics["oad_teacher_topk_mass_sum"] / token_count
                    )
                    student_mass = (
                        metrics["oad_student_mass_on_teacher_topk_sum"]
                        / token_count
                    )
                    active_pos = (
                        metrics["oad_active_grad_position_sum"] / token_count
                    )
                    active_tok = (
                        metrics["oad_active_grad_token_sum"] / token_count
                    )
                    print("  • OAD metrics:")
                    print(f"      - acceptance_rate_mean_pathB: {acc:.4f}")
                    print(
                        f"      - acceptance_rate_min_pathB:  "
                        f"{metrics['oad_min_accept_pathB']:.4f}"
                    )
                    print(f"      - tvd_mean_pathB:             {1.0 - acc:.4f}")
                    print(f"      - teacher_topk_mass:          {teacher_mass:.4f}")
                    print(
                        f"      - student_mass_on_teacher_topk:{student_mass:.4f}"
                    )
                    print(
                        f"      - active_grad_ratio_position_pathB:{active_pos:.4f}"
                    )
                    print(
                        f"      - active_grad_ratio_token_pathB:   {active_tok:.4f}"
                    )
            if "total_flops" in train_results:
                total_tflops = (
                    train_results["total_flops"]
                    / timing_metrics["policy_training"]
                    / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)",
                    flush=True,
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%",
                        flush=True,
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops

            print("\n⏱️  Timing:", flush=True)
            # Display total time first, separately
            total_time = timing_metrics.get("total_step_time", 0)

            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            metrics.update(
                {
                    "tokens_per_sec_per_gpu": metrics["total_num_tokens"]
                    / total_time
                    / total_num_gpus
                }
            )

            print(f"  • Total step time: {total_time:.2f}s", flush=True)

            # Display all other timing metrics
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= max_steps:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        # End of epoch
        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch


def _message_token_count(message: dict[str, Any]) -> int:
    """Return the number of token ids stored on a message."""
    token_ids = message.get("token_ids")
    if token_ids is None:
        return 0
    if torch.is_tensor(token_ids):
        return int(token_ids.numel())
    return len(token_ids)


def _last_token_id(message: dict[str, Any]) -> Optional[int]:
    """Return the final token id from a message, if one exists."""
    token_ids = message.get("token_ids")
    if token_ids is None or len(token_ids) == 0:
        return None
    last_token = token_ids[-1]
    return int(last_token.item()) if torch.is_tensor(last_token) else int(last_token)


def _validation_response_stats(
    message_log: list[dict[str, Any]],
    tokenizer,
    master_config: MasterConfig,
) -> tuple[int, int, bool, bool]:
    """Collect prompt/response length and termination stats for one rollout."""
    prompt_length = sum(
        _message_token_count(message)
        for message in message_log
        if message["role"] not in {"assistant", "environment"}
    )
    assistant_messages = [
        message for message in message_log if message["role"] == "assistant"
    ]
    response_length = sum(_message_token_count(message) for message in assistant_messages)

    eos_token_id = tokenizer.eos_token_id
    final_token_id = (
        _last_token_id(assistant_messages[-1]) if assistant_messages else None
    )
    ended_with_eos = eos_token_id is not None and final_token_id == eos_token_id

    max_sequence_length = master_config["policy"]["max_total_sequence_length"]
    max_new_tokens = master_config["policy"]["generation"].get(
        "max_new_tokens", max_sequence_length
    )
    available_sequence_tokens = max(max_sequence_length - prompt_length, 0)
    response_token_budget = min(max_new_tokens, available_sequence_tokens)
    exhausted_token_budget = (
        response_token_budget > 0 and response_length >= response_token_budget
    )

    return prompt_length, response_length, ended_with_eos, exhausted_token_budget


def _mean_or_nan(values: list[int]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _summarize_validation_metrics(
    rewards: list[float],
    prompt_lengths: list[int],
    response_lengths: list[int],
    ended_with_eos: list[bool],
    exhausted_token_budget: list[bool],
    rollout_truncated: list[bool],
) -> dict[str, Any]:
    """Build scalar validation metrics suitable for WandB/TensorBoard."""
    num_samples = len(rewards)
    if num_samples == 0:
        return {
            "accuracy": 0.0,
            "avg_length": 0.0,
            "num_samples": 0,
        }

    reward_array = np.asarray(rewards, dtype=np.float64)
    response_length_array = np.asarray(response_lengths, dtype=np.float64)
    accuracy = float(reward_array.mean())
    accuracy_stderr = math.sqrt(accuracy * (1.0 - accuracy) / num_samples)
    ci95_half_width = 1.96 * accuracy_stderr

    correct_lengths = [
        length for length, reward in zip(response_lengths, rewards) if reward > 0.5
    ]
    incorrect_lengths = [
        length for length, reward in zip(response_lengths, rewards) if reward <= 0.5
    ]

    return {
        # Keep the original names for checkpoint selection and existing dashboards.
        "accuracy": accuracy,
        "avg_length": float(response_length_array.mean()),
        "num_samples": num_samples,
        "num_correct": int((reward_array > 0.5).sum()),
        "num_incorrect": int((reward_array <= 0.5).sum()),
        "accuracy_stderr": accuracy_stderr,
        "accuracy_ci95_low": max(0.0, accuracy - ci95_half_width),
        "accuracy_ci95_high": min(1.0, accuracy + ci95_half_width),
        "prompt_length_mean": float(np.mean(prompt_lengths)),
        "response_length_mean": float(response_length_array.mean()),
        "response_length_std": float(response_length_array.std()),
        "response_length_min": int(response_length_array.min()),
        "response_length_p50": float(np.quantile(response_length_array, 0.50)),
        "response_length_p90": float(np.quantile(response_length_array, 0.90)),
        "response_length_p95": float(np.quantile(response_length_array, 0.95)),
        "response_length_p99": float(np.quantile(response_length_array, 0.99)),
        "response_length_max": int(response_length_array.max()),
        "correct_response_length_mean": _mean_or_nan(correct_lengths),
        "incorrect_response_length_mean": _mean_or_nan(incorrect_lengths),
        "eos_termination_rate": float(np.mean(ended_with_eos)),
        "token_budget_exhaustion_rate": float(np.mean(exhausted_token_budget)),
        "rollout_truncation_rate": float(np.mean(rollout_truncated)),
    }


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
    logger: Optional[Logger] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation", flush=True)
        return {}, {}

    if val_task_to_env is None:
        print(
            "  ⚠️ No validation task to environment mapping provided, skipping validation",
            flush=True,
        )
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...", flush=True)

        total_rewards: list[float] = []
        prompt_lengths: list[int] = []
        response_lengths: list[int] = []
        ended_with_eos: list[bool] = []
        exhausted_token_budget: list[bool] = []
        rollout_truncated: list[bool] = []
        all_message_logs = []  # Collect all message logs

        max_val_samples = master_config["distillation"]["max_val_samples"]
        for val_batch in val_dataloader:
            remaining_samples = max_val_samples - len(total_rewards)
            if remaining_samples <= 0:
                break
            if val_batch.size > remaining_samples:
                val_batch = val_batch.select_indices(list(range(remaining_samples)))

            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            # Use async rollouts if vLLM async engine is enabled
            if _should_use_async_rollouts(master_config):
                val_batch, _gen_metrics = run_async_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["distillation"][
                        "max_rollout_turns"
                    ],
                    greedy=False,
                )
            else:
                val_batch, _gen_metrics = run_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["distillation"][
                        "max_rollout_turns"
                    ],
                    greedy=False,
                )
            rewards = val_batch["total_reward"]

            total_rewards.extend(rewards.tolist())
            rollout_truncated.extend(val_batch["truncated"].bool().tolist())

            for message_log in val_batch["message_log"]:
                prompt_length, response_length, eos_ended, budget_exhausted = (
                    _validation_response_stats(message_log, tokenizer, master_config)
                )
                prompt_lengths.append(prompt_length)
                response_lengths.append(response_length)
                ended_with_eos.append(eos_ended)
                exhausted_token_budget.append(budget_exhausted)

            # Collect message logs for later display
            to_env = [
                get_keys_from_message_log(
                    val_batch["message_log"][i], ["role", "content"]
                )
                for i in range(len(val_batch["message_log"]))
            ]

            all_message_logs.extend(to_env)

        val_metrics = _summarize_validation_metrics(
            total_rewards,
            prompt_lengths,
            response_lengths,
            ended_with_eos,
            exhausted_token_budget,
            rollout_truncated,
        )

        if logger is not None and response_lengths:
            logger.log_histogram(
                response_lengths,
                step,
                "validation/response_length_histogram",
                commit=False,
            )
            correct_lengths = [
                length
                for length, reward in zip(response_lengths, total_rewards)
                if reward > 0.5
            ]
            incorrect_lengths = [
                length
                for length, reward in zip(response_lengths, total_rewards)
                if reward <= 0.5
            ]
            if correct_lengths:
                logger.log_histogram(
                    correct_lengths,
                    step,
                    "validation/correct_response_length_histogram",
                    commit=False,
                )
            if incorrect_lengths:
                logger.log_histogram(
                    incorrect_lengths,
                    step,
                    "validation/incorrect_response_length_histogram",
                    commit=False,
                )

        # Print sample conversations only once at the end of validation
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...", flush=True)

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    accuracy = val_metrics["accuracy"]
    avg_length = val_metrics["avg_length"]
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    if val_metrics["num_samples"] > 0:
        print(
            "    • EOS termination / token-budget exhaustion / rollout truncation: "
            f"{val_metrics['eos_termination_rate']:.2%} / "
            f"{val_metrics['token_budget_exhaustion_rate']:.2%} / "
            f"{val_metrics['rollout_truncation_rate']:.2%}"
        )
    print(f"    • Samples processed: {len(total_rewards)}", flush=True)

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s", flush=True)

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics
