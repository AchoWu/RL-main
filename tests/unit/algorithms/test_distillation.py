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
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

import nemo_rl.algorithms.distillation as distil_mod
from nemo_rl.algorithms.distillation import (
    _add_distillation_loss_masks,
    _default_distillation_save_state,
    _generate_teacher_prefixes,
    _resolve_progressive_teacher_block,
    _resolve_tvd_gate_threshold,
    _summarize_validation_metrics,
    check_vocab_equality,
    distillation_train,
    validate,
)
from nemo_rl.algorithms.loss_functions import DistillationLossFn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_teacher_prefix_generation_appends_masked_context():
    batch = BatchedDataDict[DatumSpec](
        {
            "message_log": [
                [
                    {
                        "token_ids": torch.tensor([1, 2]),
                        "role": "user",
                        "content": "problem",
                    }
                ]
            ],
            "loss_multiplier": torch.tensor([1.0]),
        }
    )
    teacher_generation = MagicMock()
    teacher_generation.generate.return_value = BatchedDataDict(
        {
            "output_ids": torch.tensor([[1, 2, 10, 11]]),
            "generation_lengths": torch.tensor([2]),
            "unpadded_sequence_lengths": torch.tensor([4]),
            "logprobs": torch.zeros(1, 4),
        }
    )
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.batch_decode.return_value = ["teacher prefix"]

    result, metrics = _generate_teacher_prefixes(
        teacher_generation, batch, tokenizer, requested_length=2
    )

    prefix = result["message_log"][0][-1]
    assert prefix["token_ids"].tolist() == [10, 11]
    assert prefix["token_loss_mask"].tolist() == [0, 0]
    assert metrics["teacher_prefix_mean_tokens"] == 2.0
    assert metrics["teacher_prefix_reached_requested_rate"] == 1.0
    generation_input = teacher_generation.generate.call_args.args[0]
    assert generation_input["max_new_tokens"].tolist() == [2]


@pytest.mark.parametrize(
    ("global_step", "expected_prefix", "expected_stage"),
    [
        (0, 0, 0),
        (24, 0, 0),
        (25, 256, 1),
        (49, 256, 1),
        (50, 512, 2),
        (74, 512, 2),
        (75, 768, 3),
        (99, 768, 3),
    ],
)
def test_progressive_teacher_block_schedule(
    global_step, expected_prefix, expected_stage
):
    config = {
        "teacher_prefix_length": 0,
        "progressive_teacher_blocks": {
            "enabled": True,
            "block_size": 256,
            "steps_per_stage": 25,
        },
    }

    prefix_length, student_block_size, stage = _resolve_progressive_teacher_block(
        config, global_step
    )

    assert prefix_length == expected_prefix
    assert student_block_size == 256
    assert stage == expected_stage


def test_progressive_teacher_block_disabled_preserves_fixed_prefix():
    config = {
        "teacher_prefix_length": 512,
        "progressive_teacher_blocks": {
            "enabled": False,
            "block_size": 256,
            "steps_per_stage": 25,
        },
    }

    assert _resolve_progressive_teacher_block(config, 75) == (512, None, 0)


def test_distillation_masks_preserve_teacher_prefix_and_unmask_student_suffix():
    message_logs = [
        [
            {"role": "user", "token_ids": torch.tensor([1, 2])},
            {
                "role": "assistant",
                "token_ids": torch.tensor([3, 4]),
                "token_loss_mask": torch.tensor([0, 0]),
            },
            {"role": "assistant", "token_ids": torch.tensor([5, 6])},
        ]
    ]

    _add_distillation_loss_masks(message_logs)

    assert message_logs[0][0]["token_loss_mask"].tolist() == [0, 0]
    assert message_logs[0][1]["token_loss_mask"].tolist() == [0, 0]
    assert message_logs[0][2]["token_loss_mask"].tolist() == [1, 1]

    flat, _ = batched_message_log_to_flat_message(
        message_logs, pad_value_dict={"token_ids": 0}
    )
    # DistillationLossFn slices token_mask[:, 1:]. Both student targets,
    # including the first token after the teacher prefix, remain unmasked.
    assert flat["token_loss_mask"][:, 1:].tolist() == [[0, 0, 0, 1, 1]]


@pytest.fixture
def mock_components():
    # Create mock components
    student_policy = MagicMock()
    student_policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {"global_valid_toks": [10]},
    }
    # Add generate method since student_generation will be set to student_policy
    student_policy.generate.return_value = {
        "output_ids": torch.randint(0, 8, (2, 10)),
        "generation_lengths": torch.tensor([5, 7]),
        "unpadded_sequence_lengths": torch.tensor([8, 10]),
        "logprobs": torch.randn(2, 10, 8),
    }

    teacher_policy = MagicMock()
    teacher_policy.get_topk_logits.return_value = {
        "topk_logits": torch.randn(2, 10, 64),
        "topk_indices": torch.randint(0, 8, (2, 10, 64)),
    }

    # Set student_generation to None to avoid Ray-related refit issues
    # This makes NEED_REFIT = False, so refit_policy_generation won't be called
    student_generation = None

    # Create a proper message log structure with token_ids (similar to SFT)
    # Use BatchedDataDict instead of regular dict to support repeat_interleave
    mock_batch = BatchedDataDict[DatumSpec](
        {
            "message_log": [
                [
                    {
                        "token_ids": torch.tensor([1, 2, 3]),
                        "role": "user",
                        "content": "What is 1+1?",
                    },
                    {
                        "token_ids": torch.tensor([4, 5, 6]),
                        "role": "assistant",
                        "content": "The answer is 2.",
                    },
                ]
            ],
            "loss_multiplier": torch.tensor(
                [1.0]
            ),  # Make it 1D tensor for batch dimension
            "task_name": ["math"],
            "extra_env_info": [{}],
            "length": torch.tensor([6]),  # Make it 1D tensor for batch dimension
            "idx": torch.tensor([0]),  # Make it 1D tensor for batch dimension
        }
    )

    # Create mock dataloader with 10 batches that can be iterated multiple times
    train_dataloader = MagicMock(spec=StatefulDataLoader)

    def train_iter(self):
        return iter([mock_batch] * 10)

    train_dataloader.__iter__ = train_iter
    train_dataloader.__len__ = MagicMock(return_value=10)

    val_dataloader = MagicMock(spec=StatefulDataLoader)

    def val_iter(self):
        return iter([mock_batch] * 10)

    val_dataloader.__iter__ = val_iter
    val_dataloader.__len__ = MagicMock(return_value=10)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    loss_fn = DistillationLossFn(
        {
            "kl_type": "forward",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
        }
    )

    logger = MagicMock()
    checkpointer = MagicMock()

    # Create mock environments
    task_to_env = {"math": MagicMock()}
    val_task_to_env = {"math": MagicMock()}

    # Create mock master config
    master_config = {
        "distillation": {
            "max_num_steps": 5,
            "max_num_epochs": 10,
            "val_period": 100,
            "val_batch_size": 1,
            "val_at_start": False,
            "max_val_samples": 10,
            "topk_logits_k": 64,
            "num_prompts_per_step": 1,
            "num_generations_per_prompt": 1,
            "max_rollout_turns": 0,  # No environment interaction needed for distillation
            "seed": 42,
        },
        "policy": {
            "train_global_batch_size": 1,
            "make_sequence_length_divisible_by": 8,
            "max_total_sequence_length": 2048,
            "generation": {
                "colocated": {
                    "enabled": False,
                },
            },
        },
        "teacher": {
            "model_name": "test-teacher",
        },
        "loss_fn": {
            "kl_type": "forward",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
        },
        "data": {
            "dataset_name": "test_dataset",
        },
        "logger": {
            "num_val_samples_to_print": 5,
        },
        "cluster": {
            "num_nodes": 1,
            "gpus_per_node": 2,
        },
        "checkpointing": {
            "enabled": False,
            "checkpoint_must_save_by": None,
            "save_period": 10,
            "metric_name": None,
        },
    }

    return {
        "student_policy": student_policy,
        "teacher_policy": teacher_policy,
        "student_generation": student_generation,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer": tokenizer,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "task_to_env": task_to_env,
        "val_task_to_env": val_task_to_env,
        "master_config": master_config,
    }


def test_distillation_train_max_steps(mock_components):
    """Test that training terminates correctly when maximum steps are reached."""
    mock_components["master_config"]["distillation"]["max_num_steps"] = 5

    distillation_save_state = _default_distillation_save_state()

    # Run training. The final step must be validated even though it is not a
    # multiple of val_period.
    with patch.object(
        distil_mod,
        "validate",
        return_value=({"accuracy": 0.5}, {}),
    ) as mock_validate:
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            distillation_save_state,
            mock_components["master_config"],
        )

    assert mock_components["student_policy"].train.call_count == 5
    assert mock_validate.call_count == 1
    assert mock_validate.call_args.kwargs["step"] == 5


def test_exit_on_timeout(mock_components, capsys):
    """Test that training loop exits when timeout is reached"""
    # Set max steps to large number
    mock_components["master_config"]["distillation"]["max_num_steps"] = 100

    distillation_save_state = _default_distillation_save_state()

    # Mock TimeoutChecker to return False for first 7 checks, then True (timeout)
    with patch("nemo_rl.algorithms.distillation.TimeoutChecker") as mock_timeout_class:
        mock_timeout_instance = MagicMock()
        # Create a side_effect that returns False 7 times, then True
        check_results = [False] * 7 + [True]
        mock_timeout_instance.check_save.side_effect = check_results
        mock_timeout_class.return_value = mock_timeout_instance

        # Run training
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            distillation_save_state,
            mock_components["master_config"],
        )

        # Verify training stopped at 8 steps (when check_save returned True)
        assert mock_components["student_policy"].train.call_count == 8

        # Verify the timeout message was printed and training actually stopped
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split("\n")

        # Find the timeout message
        timeout_line_idx = None
        for i, line in enumerate(output_lines):
            if "Timeout has been reached, stopping training early" in line:
                timeout_line_idx = i
                break

        assert timeout_line_idx is not None, "Timeout message not found in output"

        # For distillation, verify we don't see more step messages after timeout
        remaining_lines = output_lines[timeout_line_idx:]
        for line in remaining_lines:
            # Distillation doesn't have epochs, but check for step markers
            assert not line.startswith("Step ") or "Step 8" in line, (
                f"Training continued after timeout: {line}"
            )


def test_validate_function(mock_components):
    """Test independent validation function to ensure validation logic correctness."""
    mock_components["master_config"]["distillation"]["max_val_samples"] = 3
    # Run validation
    val_metrics, validation_timings = validate(
        mock_components["student_generation"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["val_task_to_env"],
        step=0,
        master_config=mock_components["master_config"],
    )

    # Verify validation results
    assert isinstance(val_metrics, dict)
    assert isinstance(validation_timings, dict)
    assert val_metrics["num_samples"] == 3
    assert "response_length_p95" in val_metrics
    assert "eos_termination_rate" in val_metrics
    # For distillation, we don't need environment interaction since max_rollout_turns=0
    # The validation focuses on generation and teacher-student knowledge transfer
    # Note: validate() function itself doesn't call logger.log_metrics - that's done by the caller


def test_summarize_validation_metrics_uses_per_sample_lengths():
    metrics = _summarize_validation_metrics(
        rewards=[1.0, 0.0, 1.0],
        prompt_lengths=[10, 20, 30],
        response_lengths=[10, 20, 60],
        ended_with_eos=[True, True, False],
        exhausted_token_budget=[False, False, True],
        rollout_truncated=[False, False, True],
    )

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["avg_length"] == pytest.approx(30.0)
    assert metrics["correct_response_length_mean"] == pytest.approx(35.0)
    assert metrics["incorrect_response_length_mean"] == pytest.approx(20.0)
    assert metrics["eos_termination_rate"] == pytest.approx(2 / 3)
    assert metrics["token_budget_exhaustion_rate"] == pytest.approx(1 / 3)


def test_check_vocab_equality_pass(monkeypatch):
    student_tokenizer = MagicMock()
    student_tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
    student_tokenizer.__len__.return_value = 2

    teacher_tokenizer = MagicMock()
    teacher_tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
    teacher_tokenizer.__len__.return_value = 2

    student_config = MagicMock()
    student_config.vocab_size = 2
    teacher_config = MagicMock()
    teacher_config.vocab_size = 2

    monkeypatch.setattr(
        distil_mod.AutoTokenizer,
        "from_pretrained",
        lambda name: teacher_tokenizer,
    )
    monkeypatch.setattr(
        distil_mod.AutoConfig,
        "from_pretrained",
        lambda name: student_config if name == "student-model" else teacher_config,
    )

    # Should not raise
    check_vocab_equality(student_tokenizer, "student-model", "teacher-model")


def test_check_vocab_equality_vocab_mismatch_raises(monkeypatch):
    student_tokenizer = MagicMock()
    student_tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
    student_tokenizer.__len__.return_value = 2

    teacher_tokenizer = MagicMock()
    teacher_tokenizer.get_vocab.return_value = {"a": 0, "c": 2}
    teacher_tokenizer.__len__.return_value = 2

    student_config = MagicMock()
    student_config.vocab_size = 2
    teacher_config = MagicMock()
    teacher_config.vocab_size = 2

    monkeypatch.setattr(
        distil_mod.AutoTokenizer,
        "from_pretrained",
        lambda name: teacher_tokenizer,
    )
    monkeypatch.setattr(
        distil_mod.AutoConfig,
        "from_pretrained",
        lambda name: student_config if name == "student-model" else teacher_config,
    )

    with pytest.raises(AssertionError):
        check_vocab_equality(student_tokenizer, "student-model", "teacher-model")


def test_check_vocab_equality_length_mismatch_raises(monkeypatch):
    # Same vocab mapping but different __len__ values
    vocab = {"a": 0, "b": 1}
    student_tokenizer = MagicMock()
    student_tokenizer.get_vocab.return_value = vocab
    student_tokenizer.__len__.return_value = 2

    teacher_tokenizer = MagicMock()
    teacher_tokenizer.get_vocab.return_value = vocab
    teacher_tokenizer.__len__.return_value = 3

    student_config = MagicMock()
    student_config.vocab_size = 2
    teacher_config = MagicMock()
    teacher_config.vocab_size = 2

    monkeypatch.setattr(
        distil_mod.AutoTokenizer,
        "from_pretrained",
        lambda name: teacher_tokenizer,
    )
    monkeypatch.setattr(
        distil_mod.AutoConfig,
        "from_pretrained",
        lambda name: student_config if name == "student-model" else teacher_config,
    )

    with pytest.raises(AssertionError):
        check_vocab_equality(student_tokenizer, "student-model", "teacher-model")


def test_check_vocab_equality_config_vocab_size_mismatch_raises(monkeypatch):
    vocab = {"a": 0, "b": 1}
    student_tokenizer = MagicMock()
    student_tokenizer.get_vocab.return_value = vocab
    student_tokenizer.__len__.return_value = 2

    teacher_tokenizer = MagicMock()
    teacher_tokenizer.get_vocab.return_value = vocab
    teacher_tokenizer.__len__.return_value = 2

    student_config = MagicMock()
    student_config.vocab_size = 2
    teacher_config = MagicMock()
    teacher_config.vocab_size = 3

    monkeypatch.setattr(
        distil_mod.AutoTokenizer,
        "from_pretrained",
        lambda name: teacher_tokenizer,
    )
    monkeypatch.setattr(
        distil_mod.AutoConfig,
        "from_pretrained",
        lambda name: student_config if name == "student-model" else teacher_config,
    )

    with pytest.raises(AssertionError):
        check_vocab_equality(student_tokenizer, "student-model", "teacher-model")


def test_noncolocated_inference_requires_explicit_gpus_per_node_single_node():
    """Test that non-colocated inference requires explicit gpus_per_node when cluster.num_nodes=1."""
    from unittest.mock import MagicMock, patch

    from nemo_rl.algorithms.distillation import setup

    # Create minimal config with non-colocated inference but gpus_per_node=None
    master_config = {
        "policy": {
            "generation": {
                "backend": "vllm",
                "colocated": {
                    "enabled": False,  # Non-colocated
                    "resources": {
                        "gpus_per_node": None,  # This should trigger error
                        "num_nodes": None,
                    },
                },
            },
            "dtensor_cfg": {
                "enabled": False,
            },
        },
        "teacher": {
            "dtensor_cfg": {
                "enabled": False,
            },
        },
        "loss_fn": {},
        "distillation": {
            "seed": 42,
            "topk_logits_k": 64,
            "num_prompts_per_step": 1,  # Config extraction requires this key
            "val_period": 0,  # Config extraction requires this key
            "val_at_start": False,  # Config extraction requires this key
        },
        "data": {"shuffle": False},
        "logger": {},  # Config extraction requires this key
        "checkpointing": {},  # Config extraction requires this key
        "cluster": {
            "num_nodes": 1,  # Single node
            "gpus_per_node": 8,
        },
    }

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)

    # Mock everything we don't need to test
    with (
        patch("nemo_rl.algorithms.distillation.Logger") as mock_logger,
        patch("nemo_rl.algorithms.distillation.CheckpointManager") as mock_checkpointer,
        patch("nemo_rl.algorithms.distillation.StatefulDataLoader"),
        pytest.raises(
            AssertionError,
            match="policy.generation.colocated.resources.gpus_per_node must be explicitly set",
        ),
    ):
        # Configure mocks to skip checkpoint loading
        mock_checkpointer.return_value.get_latest_checkpoint_path.return_value = None
        setup(master_config, tokenizer, dataset, None)


def test_distillation_setup_non_colocated_smoke(monkeypatch):
    """Smoke test: calling setup with a non-colocated config should succeed."""
    from unittest.mock import MagicMock, patch

    import nemo_rl.algorithms.distillation as distil_mod

    # Single node cluster; inference uses a subset of GPUs on same node
    master_config = {
        "policy": {
            "generation": {
                "backend": "vllm",
                "colocated": {
                    "enabled": False,
                    "resources": {
                        "gpus_per_node": 8,  # inference on 8 GPU
                        "num_nodes": 1,
                    },
                },
            },
            "dtensor_cfg": {
                "enabled": False,
            },
            "model_name": "test-policy",
        },
        "teacher": {
            "model_name": "test-teacher",
            "dtensor_cfg": {
                "enabled": False,
            },
        },
        "loss_fn": {
            "kl_type": "forward",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
        },
        "distillation": {
            "seed": 42,
            "topk_logits_k": 64,
            "num_prompts_per_step": 1,
            "max_num_epochs": 10,
            "max_num_steps": 100,
            "val_period": 0,
            "val_at_start": False,
        },
        "data": {"shuffle": False},
        "logger": {},
        "checkpointing": {},
        "cluster": {"num_nodes": 2, "gpus_per_node": 8},
    }

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=1)

    # Skip tokenizer/vocab equality check inside setup
    monkeypatch.setenv("NRL_SKIP_DISTILLATION_TOKENIZER_CHECK", "1")

    ip_port = ("127.0.0.1", 12345)

    class DummyCluster:
        def __init__(self, *args, **kwargs):
            pass

        def world_size(self):
            return 1

        def get_master_address_and_port(self):
            return ip_port

    class DummyPolicy:
        def __init__(self, *args, **kwargs):
            pass

        def prepare_refit_info(self):
            return {}

        def offload_after_refit(self):
            return None

        def init_collective(self, *args, **kwargs):
            return [MagicMock()]

    class DummyVllmGeneration:
        def __init__(self, *args, **kwargs):
            pass

        def finish_generation(self):
            return None

        def prepare_refit_info(self, *args, **kwargs):
            return None

        def init_collective(self, *args, **kwargs):
            return [MagicMock()]

    with (
        patch.object(distil_mod, "RayVirtualCluster", DummyCluster),
        patch.object(distil_mod, "Logger"),
        patch.object(distil_mod, "CheckpointManager") as mock_ckpt_mgr,
        patch.object(distil_mod, "StatefulDataLoader"),
        patch.object(distil_mod, "Policy", DummyPolicy),
        patch.object(distil_mod, "VllmGeneration", DummyVllmGeneration),
        patch.object(distil_mod, "ray") as mock_ray,
    ):
        mock_ckpt_mgr.return_value.get_latest_checkpoint_path.return_value = None
        mock_ray.get = MagicMock(return_value=None)

        # Should not raise
        result = distil_mod.setup(master_config, tokenizer, dataset, None)

        # Basic shape check of returned tuple
        assert isinstance(result, tuple)


def test_noncolocated_inference_requires_explicit_gpus_per_node_multi_node():
    """Test that non-colocated inference requires explicit gpus_per_node when cluster.num_nodes>1."""
    from unittest.mock import MagicMock, patch

    from nemo_rl.algorithms.distillation import setup

    # Create minimal config with non-colocated inference but gpus_per_node=None
    master_config = {
        "policy": {
            "generation": {
                "backend": "vllm",
                "colocated": {
                    "enabled": False,  # Non-colocated
                    "resources": {
                        "gpus_per_node": None,  # This should trigger error
                        "num_nodes": 1,  # Use 1 node for inference
                    },
                },
            },
            "dtensor_cfg": {
                "enabled": False,
            },
        },
        "teacher": {
            "dtensor_cfg": {
                "enabled": False,
            },
        },
        "loss_fn": {},
        "distillation": {
            "seed": 42,
            "topk_logits_k": 64,
            "max_num_epochs": 10,
            "max_num_steps": 100,
            "num_prompts_per_step": 1,  # Config extraction requires this key
            "val_period": 0,  # Config extraction requires this key
            "val_at_start": False,  # Config extraction requires this key
        },
        "data": {"shuffle": False},
        "logger": {},  # Config extraction requires this key
        "checkpointing": {},  # Config extraction requires this key
        "cluster": {
            "num_nodes": 2,  # Multi-node
            "gpus_per_node": 8,
        },
    }

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)

    # Mock everything we don't need to test
    with (
        patch("nemo_rl.algorithms.distillation.Logger") as mock_logger,
        patch("nemo_rl.algorithms.distillation.CheckpointManager") as mock_checkpointer,
        patch("nemo_rl.algorithms.distillation.StatefulDataLoader"),
        pytest.raises(
            AssertionError,
            match="policy.generation.colocated.resources.gpus_per_node must be explicitly set",
        ),
    ):
        # Configure mocks to skip checkpoint loading
        mock_checkpointer.return_value.get_latest_checkpoint_path.return_value = None
        setup(master_config, tokenizer, dataset, None)


# ---------------------------------------------------------------------------
# TVD gate: threshold resolver + gate math sanity checks.
# ---------------------------------------------------------------------------


class TestResolveTvdGateThreshold:
    def test_none_when_no_config(self):
        mode, tau = _resolve_tvd_gate_threshold(None, 0, 100)
        assert mode == "none" and tau == 0.0

    def test_none_when_mode_none(self):
        mode, tau = _resolve_tvd_gate_threshold({"mode": "none"}, 5, 100)
        assert mode == "none" and tau == 0.0

    def test_fixed_returns_scalar(self):
        mode, tau = _resolve_tvd_gate_threshold(
            {"mode": "fixed", "threshold": 0.3}, 0, 100
        )
        assert mode == "fixed" and tau == 0.3

    def test_fixed_clamps(self):
        _, tau_hi = _resolve_tvd_gate_threshold(
            {"mode": "fixed", "threshold": 1.5}, 0, 100
        )
        _, tau_lo = _resolve_tvd_gate_threshold(
            {"mode": "fixed", "threshold": -0.1}, 0, 100
        )
        assert tau_hi == 1.0 and tau_lo == 0.0

    def test_warmup_endpoints(self):
        cfg = {
            "mode": "warmup",
            "start_threshold": 0.8,
            "end_threshold": 0.1,
            "warmup_until_frac": 0.3,
        }
        _, tau_0 = _resolve_tvd_gate_threshold(cfg, 0, 1000)
        _, tau_end = _resolve_tvd_gate_threshold(cfg, 300, 1000)
        _, tau_after = _resolve_tvd_gate_threshold(cfg, 700, 1000)
        assert tau_0 == pytest.approx(0.8, abs=1e-9)
        assert tau_end == pytest.approx(0.1, abs=1e-9)
        assert tau_after == pytest.approx(0.1, abs=1e-9)

    def test_warmup_midpoint(self):
        # progress=0.5 -> curve=0.5 -> τ = start + (end-start)*0.5
        cfg = {
            "mode": "warmup",
            "start_threshold": 0.8,
            "end_threshold": 0.1,
            "warmup_until_frac": 0.3,
        }
        _, tau_mid = _resolve_tvd_gate_threshold(cfg, 150, 1000)
        assert tau_mid == pytest.approx(0.45, abs=1e-9)

    def test_warmup_monotone_decreasing(self):
        cfg = {
            "mode": "warmup",
            "start_threshold": 0.8,
            "end_threshold": 0.1,
            "warmup_until_frac": 0.3,
        }
        prev = float("inf")
        for step in range(0, 301, 10):
            _, tau = _resolve_tvd_gate_threshold(cfg, step, 1000)
            assert tau <= prev + 1e-9
            prev = tau

    def test_warmup_increasing(self):
        cfg = {
            "mode": "warmup",
            "start_threshold": 0.1,
            "end_threshold": 0.8,
            "warmup_until_frac": 0.3,
        }
        _, tau_0 = _resolve_tvd_gate_threshold(cfg, 0, 1000)
        _, tau_end = _resolve_tvd_gate_threshold(cfg, 300, 1000)
        assert tau_0 == pytest.approx(0.1, abs=1e-9)
        assert tau_end == pytest.approx(0.8, abs=1e-9)

    def test_warmup_zero_until_frac(self):
        # Degenerate: no warmup, stay at end_threshold immediately.
        cfg = {
            "mode": "warmup",
            "start_threshold": 0.8,
            "end_threshold": 0.1,
            "warmup_until_frac": 0.0,
        }
        _, tau = _resolve_tvd_gate_threshold(cfg, 0, 1000)
        assert tau == pytest.approx(0.1, abs=1e-9)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown tvd_gate mode"):
            _resolve_tvd_gate_threshold({"mode": "bogus"}, 0, 100)


class TestTvdGateMath:
    """Sanity checks on the min-overlap → TVD identity and threshold semantics.

    These don't touch the loss function directly; they verify the arithmetic
    the gate relies on (torch.minimum + strict-> comparison) matches intent.
    """

    def test_identity_gives_zero_tvd(self):
        # Any distribution vs itself: min(p,p).sum() == 1, TVD == 0.
        torch.manual_seed(0)
        p = torch.softmax(torch.randn(2, 5, 10), dim=-1)
        tvd = 1.0 - torch.minimum(p, p).sum(-1)
        assert tvd.max().item() < 1e-6

    def test_disjoint_gives_tvd_one(self):
        p_S = torch.zeros(10)
        p_S[0] = 1.0
        p_T = torch.zeros(10)
        p_T[-1] = 1.0
        tvd = 1.0 - torch.minimum(p_S, p_T).sum()
        assert tvd.item() == pytest.approx(1.0, abs=1e-9)

    def test_half_overlap(self):
        # Both distributions concentrated on {0,1} vs {0,2} → overlap 0.5.
        p_S = torch.zeros(10)
        p_S[0] = 0.5
        p_S[1] = 0.5
        p_T = torch.zeros(10)
        p_T[0] = 0.5
        p_T[2] = 0.5
        tvd = 1.0 - torch.minimum(p_S, p_T).sum()
        assert tvd.item() == pytest.approx(0.5, abs=1e-9)

    def test_gate_direction_keeps_high_tvd(self):
        # Threshold semantics: keep iff tvd > τ (direction B).
        tvd = torch.tensor([0.1, 0.4, 0.6, 0.9])
        gate = (tvd > 0.5).float()
        assert gate.tolist() == [0.0, 0.0, 1.0, 1.0]

    def test_gate_all_out_at_tau_one(self):
        # τ = 1 gates everything out (tvd is bounded by 1).
        tvd = torch.tensor([0.1, 0.5, 0.9, 1.0])
        gate = (tvd > 1.0).float()
        assert gate.sum().item() == 0

    def test_gate_all_in_at_tau_neg(self):
        # τ = -1 keeps everything.
        tvd = torch.tensor([0.0, 0.4, 0.9])
        gate = (tvd > -1.0).float()
        assert gate.sum().item() == 3


class TestTvdGateInitValidation:
    """Fail-fast at DistillationLossFn.__init__ when the gate is misconfigured.

    Users copying the exemplar yaml commonly drop fields they don't use
    (e.g. keep only `mode`+`threshold` for fixed) and later flip
    `mode='warmup'`. Validation lives in __init__ so the failure surfaces
    at controller construction time, BEFORE Ray workers spin up and teacher
    rollouts run — the resolver itself uses direct dict access on the same
    keys, so a misconfig would also fail there, but by that point the run
    has already burned expensive setup.
    """

    def _kl_base(self):
        return {
            "kl_type": "forward",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": True,
        }

    def test_fixed_missing_threshold(self):
        cfg = self._kl_base()
        cfg["tvd_gate"] = {"mode": "fixed"}
        with pytest.raises(AssertionError, match="threshold"):
            DistillationLossFn(cfg)

    def test_warmup_missing_start_threshold(self):
        cfg = self._kl_base()
        cfg["tvd_gate"] = {
            "mode": "warmup",
            "end_threshold": 0.1,
            "warmup_until_frac": 0.3,
        }
        with pytest.raises(AssertionError, match="start_threshold"):
            DistillationLossFn(cfg)

    def test_warmup_missing_all_keys(self):
        cfg = self._kl_base()
        cfg["tvd_gate"] = {"mode": "warmup"}
        with pytest.raises(AssertionError, match="start_threshold"):
            DistillationLossFn(cfg)

    def test_requires_zero_outside_topk(self):
        cfg = self._kl_base()
        cfg["zero_outside_topk"] = False
        cfg["tvd_gate"] = {"mode": "fixed", "threshold": 0.3}
        with pytest.raises(AssertionError, match="zero_outside_topk"):
            DistillationLossFn(cfg)

    def test_unknown_mode_raises(self):
        cfg = self._kl_base()
        cfg["tvd_gate"] = {"mode": "bogus"}
        with pytest.raises(ValueError, match="Unknown tvd_gate.mode"):
            DistillationLossFn(cfg)

    def test_mode_none_accepted_without_extra_keys(self):
        # Baseline / opt-out: mode=none should require no other keys.
        cfg = self._kl_base()
        cfg["tvd_gate"] = {"mode": "none"}
        loss = DistillationLossFn(cfg)
        assert loss._tvd_gate_state["mode"] == "none"

    def test_no_gate_cfg_at_all(self):
        # Absent tvd_gate key = feature completely disabled, same as mode=none.
        loss = DistillationLossFn(self._kl_base())
        assert loss._tvd_gate_state["mode"] == "none"

    def test_direction_defaults_to_high(self):
        # Legacy configs (no `direction` key) must resolve to "high" so every
        # experiment that predates this feature keeps its exact prior semantics.
        cfg = self._kl_base()
        cfg["tvd_gate"] = {"mode": "fixed", "threshold": 0.3}
        loss = DistillationLossFn(cfg)
        assert loss.tvd_gate_direction == "high"

    def test_direction_low_accepted(self):
        cfg = self._kl_base()
        cfg["tvd_gate"] = {"mode": "fixed", "threshold": 0.3, "direction": "low"}
        loss = DistillationLossFn(cfg)
        assert loss.tvd_gate_direction == "low"

    def test_unknown_direction_raises(self):
        cfg = self._kl_base()
        cfg["tvd_gate"] = {
            "mode": "fixed",
            "threshold": 0.3,
            "direction": "sideways",
        }
        with pytest.raises(ValueError, match="Unknown tvd_gate.direction"):
            DistillationLossFn(cfg)

    def test_direction_none_when_mode_none(self):
        # mode=none should short-circuit direction validation — a stray
        # `direction=bogus` on a disabled gate is not the user's problem.
        cfg = self._kl_base()
        cfg["tvd_gate"] = {"mode": "none", "direction": "sideways"}
        loss = DistillationLossFn(cfg)
        assert loss._tvd_gate_state["mode"] == "none"


class TestTvdGateDirectionMath:
    """Direction semantics: 'high' keeps tvd > τ, 'low' keeps tvd < τ.

    These verify the boolean comparison the gate uses, not the loss function
    end-to-end (which would need a full teacher/student mock).
    """

    def test_low_direction_keeps_similar_tokens(self):
        # Curriculum start: only near-agreement tokens (low TVD) survive.
        tvd = torch.tensor([0.01, 0.2, 0.5, 0.9])
        # τ = 0.1 — only the 0.01 token passes.
        gate = (tvd < 0.1).float()
        assert gate.tolist() == [1.0, 0.0, 0.0, 0.0]

    def test_low_direction_at_tau_one_keeps_all(self):
        # τ = 1.0 admits every token because TVD is clamped to [0, 1].
        # (Strict < means tvd == 1.0 exactly is dropped, but such positions
        # are vanishingly rare — only when student and teacher are literally
        # disjoint on the top-k support.)
        tvd = torch.tensor([0.0, 0.4, 0.9])
        gate = (tvd < 1.0).float()
        assert gate.sum().item() == 3

    def test_low_direction_at_tau_zero_keeps_none(self):
        # τ = 0 keeps nothing (TVD is bounded below by 0, strict <).
        tvd = torch.tensor([0.0, 0.1, 0.5])
        gate = (tvd < 0.0).float()
        assert gate.sum().item() == 0

    def test_high_and_low_are_complementary_off_boundary(self):
        # Away from the boundary (tvd != τ), every token is kept by exactly
        # one direction. Together they cover the full base mask — useful as
        # a sanity check that the two directions really are "either side".
        tvd = torch.tensor([0.1, 0.3, 0.7, 0.95])
        tau = 0.5
        high = (tvd > tau).float()
        low = (tvd < tau).float()
        # No tvd equals τ here, so high + low == 1 everywhere.
        assert torch.equal(high + low, torch.ones_like(high))
