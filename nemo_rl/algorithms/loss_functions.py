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
import math
from typing import Any, NotRequired, Optional, TypedDict, TypeVar

import torch
import torch.distributed

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.algorithms.utils import calculate_kl, masked_mean
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    ChunkedDistributedEntropy,
    ChunkedDistributedGatherLogprob,
    _get_tokens_on_this_cp_rank,
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    from_parallel_logits_to_logprobs,
    gather_logits_at_global_indices,
    get_logprobs_from_vocab_parallel_logits,
    vocab_cp_logsumexp,
)

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class ClippedPGLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    reference_policy_kl_type: str
    kl_input_clamp_value: float | None
    kl_output_clamp_value: float | None
    ratio_clip_min: float
    ratio_clip_max: float
    # Dual-clipping value (should be >1 if enabled; usually set to 3 empirically). None to disable.
    ratio_clip_c: float | None
    use_on_policy_kl_approximation: bool
    use_importance_sampling_correction: bool
    truncated_importance_sampling_ratio: float | None
    token_level_loss: bool
    # If True, apply the off-policy importance-sampling correction at the
    # sequence level (one weight per generated sample), as in GSPO.
    # If False (default), correction is applied at the token level as in the
    # original GRPO paper.
    sequence_level_importance_ratios: NotRequired[bool]
    disable_ppo_ratio: NotRequired[bool]
    # If True, force the ratio to 1.0 for truly on-policy behavior,
    # eliminating any importance sampling effects.
    # NOTE: This should only be used when doing exactly one update per rollout
    # (i.e., num_prompts_per_step * num_generations_per_prompt == train_global_batch_size)
    force_on_policy_ratio: NotRequired[bool]


class ClippedPGLossDataDict(TypedDict):
    """Required keys for the Clipped Policy Gradient loss function."""

    input_ids: torch.Tensor
    advantages: torch.Tensor
    prev_logprobs: torch.Tensor
    generation_logprobs: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    __extra__: Any


class ClippedPGLossFn(LossFunction):
    """Generalized Clipped Policy Gradient loss function w/ KL regularization.

    This implements:

    - PPO (Clipped) - https://arxiv.org/abs/1707.06347
    - GRPO - https://arxiv.org/abs/2402.03300
    - REINFORCE/RLOO (set disable_ppo_ratio = True and ignores ratio_clip_min/ratio_clip_max) - https://arxiv.org/abs/2402.14740
    - GSPO (set sequence_level_importance_ratios = True and token_level_loss = False) - https://arxiv.org/abs/2507.18071
    - Truly on-policy (set force_on_policy_ratio = True to force ratio = 1.0, requires one update per rollout)

    Formula:
    L(θ) = E_t [ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ] - β * KL(π_θ || π_ref)

    where:
    - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio
    - A_t is the advantage estimate
    - ε is the clip parameter (ratio_clip_min/ratio_clip_max)
        - As proposed in the DAPO paper (https://arxiv.org/pdf/2503.14476),
          we allow setting a distinct minimum and maximum value for the clip parameter (set to the same value for PPO/GRPO/etc.)
            - ratio_clip_min: minimum value for the clip parameter
            - ratio_clip_max: maximum value for the clip parameter
    - β is the KL penalty coefficient (reference_policy_kl_penalty)
    - KL(π_θ || π_ref) is the KL divergence between the current policy and reference policy (Schulman Approx.)

    For REINFORCE/RLOO (when disable_ppo_ratio=True), the formula simplifies to:
    L(θ) = E_t [ π_θ(a_t|s_t) * A_t ] - β * KL(π_θ || π_ref)

    Also supports "Dual-Clipping" from https://arxiv.org/pdf/1912.09729, which
    imposes an additional upper bound on the probability ratio when advantages are negative.
    This prevents excessive policy updates. $rA << 0$ -> $cA$(clipped)
    The loss function is modified to the following when A_t < 0:
    L(θ) = E_t [ max(min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t), c * A_t) ] - β * KL(π_θ || π_ref)

    where:
    - c is the dual-clip parameter (ratio_clip_c), which must be greater than 1 and is
      usually set as 3 empirically.

    Due to potential numerical instability, we cast the logits to float32 before computing the loss.
    """

    def __init__(self, cfg: ClippedPGLossConfig):
        self.ratio_clip_min = cfg["ratio_clip_min"]
        self.ratio_clip_max = cfg["ratio_clip_max"]
        self.ratio_clip_c = cfg["ratio_clip_c"]  # set to None to disable dual-clipping
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.reference_policy_kl_type = cfg["reference_policy_kl_type"]
        self.kl_input_clamp_value = cfg["kl_input_clamp_value"]
        self.kl_output_clamp_value = cfg["kl_output_clamp_value"]
        self.disable_ppo_ratio = cfg.get("disable_ppo_ratio", False)
        self.force_on_policy_ratio = cfg.get(
            "force_on_policy_ratio", False
        )  # Force ratio to 1.0
        self.use_on_policy_kl_approximation = cfg["use_on_policy_kl_approximation"]
        self.use_importance_sampling_correction = cfg[
            "use_importance_sampling_correction"
        ]
        self.truncated_importance_sampling_ratio = cfg[
            "truncated_importance_sampling_ratio"
        ]
        # Whether to compute importance weights per-sequence instead of per-token.
        self.sequence_level_importance_ratios = cfg.get(
            "sequence_level_importance_ratios",
            False,
        )
        self.loss_type = (
            LossType.TOKEN_LEVEL if cfg["token_level_loss"] else LossType.SEQUENCE_LEVEL
        )
        if self.sequence_level_importance_ratios:
            assert self.loss_type == LossType.SEQUENCE_LEVEL, (
                "sequence-level importance sampling (e.g. GSPO) is mutually exclusive with token-level loss"
            )
        if self.truncated_importance_sampling_ratio is not None:
            assert self.use_importance_sampling_correction, (
                "truncated_importance_sampling_ratio is only supported when use_importance_sampling_correction is True"
            )
            assert self.truncated_importance_sampling_ratio > 0, (
                "truncated_importance_sampling_ratio should be positive"
            )

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[ClippedPGLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Clipped Policy Gradient RL loss function."""
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        advantages = data["advantages"][:, 1:]
        prev_logprobs = data["prev_logprobs"][:, 1:]
        generation_logprobs = data["generation_logprobs"][:, 1:]
        reference_policy_logprobs = data["reference_policy_logprobs"][:, 1:]
        seq_index = data.get("seq_index", None)

        mask = token_mask * sample_mask.unsqueeze(-1)

        # token_mult_prob_error
        # See more details and other metrics in docs/guides/grpo.md#metrics
        lp_error = torch.abs(generation_logprobs - prev_logprobs)  # noqa: F841  (precommit ignore for now)
        # average over all tokens in the microbatch
        mult_prob_error = masked_mean(
            torch.exp(lp_error * mask),
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        # gen-kl: kl(P_gen || P_train)
        # where log_ratio = prev_logprobs - generation_logprobs
        gen_kl_error = calculate_kl(
            logprobs=generation_logprobs,
            logprobs_reference=prev_logprobs,
            kl_type=self.reference_policy_kl_type,
            input_clamp_value=None,
            output_clamp_value=None,
        )
        gen_kl_error = masked_mean(
            gen_kl_error,
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        # policy-kl: kl(P_train || P_gen)
        # where log_ratio = generation_logprobs - prev_logprobs
        policy_kl_error = calculate_kl(
            logprobs=prev_logprobs,
            logprobs_reference=generation_logprobs,
            kl_type=self.reference_policy_kl_type,
            input_clamp_value=None,
            output_clamp_value=None,
        )
        policy_kl_error = masked_mean(
            policy_kl_error,
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        # Jensen-Shannon divergence
        # M = 0.5 * (P_train + P_gen)
        # JSD = 0.5 * KL(P_train || M) + 0.5 * KL(P_gen || M)
        log_mixture = torch.log(
            0.5 * torch.exp(prev_logprobs) + 0.5 * torch.exp(generation_logprobs)
        )
        # KL(P_train || M)
        kl_prev_to_mixture = (
            torch.exp(prev_logprobs - log_mixture) - (prev_logprobs - log_mixture) - 1
        )

        # KL(P_gen || M)
        kl_gen_to_mixture = (
            torch.exp(generation_logprobs - log_mixture)
            - (generation_logprobs - log_mixture)
            - 1
        )

        js_divergence_error = masked_mean(
            0.5 * kl_prev_to_mixture + 0.5 * kl_gen_to_mixture,
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        next_token_logits = next_token_logits.to(torch.float32)

        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            curr_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            # slice off to the correct length to remove potential CP padding
            curr_logprobs = curr_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            curr_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_token_logits_wo_last = next_token_logits[
                :, :-1
            ]  # Remove last position's logits
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits_wo_last, dim=-1
            )
            next_tokens = data["input_ids"][:, 1:].cuda()  # Skip first token
            curr_logprobs = next_token_logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        # Calculate KL regularization.
        if self.reference_policy_kl_penalty != 0:
            if self.use_on_policy_kl_approximation:
                # See: docs/guides/grpo.md#on-policy-kl-approximation
                kl_importance_weights = torch.exp(
                    curr_logprobs - generation_logprobs
                ).detach()
                kl_importance_weights = torch.nan_to_num(
                    kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
                )
            else:
                kl_importance_weights = torch.ones_like(curr_logprobs)
            kl = (
                kl_importance_weights
                * self.reference_policy_kl_penalty
                * calculate_kl(
                    logprobs=curr_logprobs,
                    logprobs_reference=reference_policy_logprobs,
                    kl_type=self.reference_policy_kl_type,
                    input_clamp_value=self.kl_input_clamp_value,
                    output_clamp_value=self.kl_output_clamp_value,
                )
            )
            if self.loss_type == LossType.TOKEN_LEVEL:
                kl = masked_mean(
                    kl, mask, global_normalization_factor=global_valid_toks
                )
            else:
                kl = masked_mean(
                    masked_mean(kl, token_mask, dim=-1),
                    sample_mask,
                    global_normalization_factor=global_valid_seqs,
                )
        else:
            kl = torch.tensor(0.0)

        # Calculate clipped loss function if ppo ratio is enabled.
        if self.force_on_policy_ratio:
            # Force ratio to 1.0 for truly on-policy behavior
            # Use curr_logprobs twice so ratio=1 but gradients still flow
            log_ratios = curr_logprobs - curr_logprobs.detach()
            ratios = log_ratios.exp()  # = exp(0) = 1.0, but depends on curr_logprobs
            ratios_clamped = ratios
        elif not self.disable_ppo_ratio:
            log_ratios = curr_logprobs - prev_logprobs
            if self.sequence_level_importance_ratios:
                seq_log_ratio_mean = masked_mean(
                    log_ratios,
                    token_mask,
                    dim=-1,
                ).unsqueeze(-1)
                seq_ratio = seq_log_ratio_mean.exp()
                ratios = seq_ratio.repeat(1, advantages.shape[1])
            else:
                ratios = log_ratios.exp()
            ratios_clamped = ratios.clamp(
                1.0 - self.ratio_clip_min, 1.0 + self.ratio_clip_max
            )
        else:
            ratios = curr_logprobs
            ratios_clamped = curr_logprobs

        loss1 = -advantages * ratios
        loss2 = -advantages * ratios_clamped

        # Determine which value to use for clipping (max for pessimistic estimate)
        clip_loss = torch.max(loss1, loss2)

        # Dual-clipping see https://arxiv.org/pdf/1912.09729
        if self.ratio_clip_c is not None:
            assert self.ratio_clip_c > 1, (
                f"ratio_clip_c must exceed 1 representing a lower bound of the ratios, got {self.ratio_clip_c}."
            )
            loss3 = -advantages * self.ratio_clip_c
            clip_loss = torch.where(
                advantages < 0, torch.min(clip_loss, loss3), clip_loss
            )

        # -------------------------------------------------------------
        # Off-policy (actor) importance-sampling correction
        # -------------------------------------------------------------
        # See: docs/guides/grpo.md#importance-sampling-correction
        if self.sequence_level_importance_ratios:
            # importance weight w_i = exp(Σ_t (log π_actor − log π_behaviour))
            seq_lp_diff = ((prev_logprobs - generation_logprobs) * mask).sum(dim=-1)
            actor_importance_weights = torch.exp(seq_lp_diff).detach()
            actor_importance_weights = torch.nan_to_num(
                actor_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
            )
            # Broadcast to token dimension so we can reuse existing reduction
            actor_importance_weights_expanded = actor_importance_weights.unsqueeze(-1)
        else:
            # Token-level correction
            actor_importance_weights_expanded = torch.exp(
                prev_logprobs - generation_logprobs
            )
            actor_importance_weights_expanded = torch.nan_to_num(
                actor_importance_weights_expanded, nan=0.0, posinf=0.0, neginf=0.0
            )
        # TIS see https://fengyao.notion.site/off-policy-rl
        if self.truncated_importance_sampling_ratio is not None:
            actor_importance_weights_expanded = torch.clamp(
                actor_importance_weights_expanded,
                max=self.truncated_importance_sampling_ratio,
            )
        actor_importance_weights = actor_importance_weights_expanded
        del actor_importance_weights_expanded
        if self.use_importance_sampling_correction:
            importance_weights_to_use = actor_importance_weights
        else:
            importance_weights_to_use = torch.ones_like(prev_logprobs)

        if self.loss_type == LossType.TOKEN_LEVEL:
            actor_loss = masked_mean(
                importance_weights_to_use * clip_loss,
                mask,
                global_normalization_factor=global_valid_toks,
            )
        else:
            actor_loss = masked_mean(
                masked_mean(
                    importance_weights_to_use * clip_loss,
                    token_mask,
                    dim=-1,
                ),
                sample_mask,
                global_normalization_factor=global_valid_seqs,
            )

        # Metric: sampling importance ratio (mean over samples)
        # See: docs/guides/grpo.md#sampling-importance-ratio
        if self.sequence_level_importance_ratios:
            sample_importance_ratio = masked_mean(
                actor_importance_weights,
                sample_mask,
                global_normalization_factor=global_valid_seqs,
            )
        else:
            sample_importance_ratio = masked_mean(
                actor_importance_weights,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        # Approximating entropy as E_{s ~ \pi_{gen}(s)}[-(\pi_{curr}/\pi_{gen})log(\pi_{curr}(s))]
        # See more details and other metrics in docs/guides/grpo.md#metrics
        with torch.no_grad():
            seq_entropy_approx = -masked_mean(
                torch.exp(curr_logprobs - generation_logprobs) * curr_logprobs,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        loss = actor_loss + kl
        with torch.no_grad():
            probs_ratio = masked_mean(
                ratios.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            probs_ratio_clamped = masked_mean(
                ratios_clamped.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()

            # Calculate min/max values for ratios (only for valid tokens)
            masked_ratios = ratios.detach()[mask.bool()]
            masked_ratios_clamped = ratios_clamped.detach()[mask.bool()]

            # Handle edge case where there might be no valid tokens
            if masked_ratios.numel() > 0:
                probs_ratio_min = masked_ratios.min().item()
                probs_ratio_max = masked_ratios.max().item()
                probs_ratio_clamped_min = masked_ratios_clamped.min().item()
                probs_ratio_clamped_max = masked_ratios_clamped.max().item()
            else:
                probs_ratio_min = float("inf")
                probs_ratio_max = float("-inf")
                probs_ratio_clamped_min = float("inf")
                probs_ratio_clamped_max = float("-inf")

        # If you provided a global_valid_{seqs/toks}, all metrics here are globally normalized
        # by either sequence or token count, depending on particular metric.
        # To get the true metric, you'll need to sum over the microbatch.
        return (
            loss,
            {
                "loss": loss.item(),
                "probs_ratio": probs_ratio,
                "probs_ratio_clamped": probs_ratio_clamped,
                "probs_ratio_min": probs_ratio_min,
                "probs_ratio_max": probs_ratio_max,
                "probs_ratio_clamped_min": probs_ratio_clamped_min,
                "probs_ratio_clamped_max": probs_ratio_clamped_max,
                "kl_penalty": kl.item() / self.reference_policy_kl_penalty if kl else 0,
                "token_mult_prob_error": mult_prob_error,
                "gen_kl_error": gen_kl_error,
                "policy_kl_error": policy_kl_error,
                "js_divergence_error": js_divergence_error,
                "sampling_importance_ratio": sample_importance_ratio.item(),
                "num_valid_samples": sample_mask.sum().item(),
                "approx_entropy": seq_entropy_approx.item(),
            },
        )


class NLLLoss(LossFunction):
    """Negative Log Likelihood Loss function."""

    loss_type = LossType.TOKEN_LEVEL

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: Tensor | None,
        global_valid_toks: Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        dpo_loss: bool = False,
        dpo_average_log_probs: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # logits shape: [batch_size, seq_len, vocab_size]
        # Get the next token logits for each position
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        mask = token_mask * sample_mask.unsqueeze(-1)
        seq_index = data.get("seq_index", None)

        next_token_logits = next_token_logits.to(torch.float32)

        # Gather the logprobs for the actual next tokens
        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            token_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            # slice off to the correct length to remove potential CP padding
            token_logprobs = token_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_tokens = data["input_ids"][:, 1:].cuda()  # Skip first token
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )
            logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits
            token_logprobs = logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        if dpo_loss:
            ## shape: [batch_size]
            num_unmasked_tokens = torch.sum(mask, -1)
            ## multiply by sample_mask to zero out invalid samples
            loss = -torch.sum(token_logprobs * mask, dim=-1)
            if dpo_average_log_probs:
                loss = loss / num_unmasked_tokens.clamp(min=1)
        else:
            ## single scalar loss
            ## scale by the total number of tokens in the batch
            loss = -masked_mean(
                token_logprobs,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        return loss, {
            "loss": loss.item() if loss.ndim == 0 else loss,
            "num_unmasked_tokens": mask.sum().item(),
            "num_valid_samples": sample_mask.sum().item(),
        }


class PreferenceLossDataDict(TypedDict):
    """Required keys for the preference loss function."""

    input_ids: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class PreferenceLoss(LossFunction):
    """Preference Loss function.

    Optimizes the model to prefer chosen responses over rejected ones

    The preference loss is computed as:
    L_pref(θ) = -E[log(σ(β * (r_chosen - r_rejected)))]

    where:
    - σ is the sigmoid function
    - β is a scaling factor (ex: `reference_policy_kl_penalty` in DPO)
    - r_chosen and r_rejected are the rewards for chosen and rejected responses

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing:
            - The preference loss value
            - A dictionary with metrics including:
                - loss: Preference loss
                - accuracy: Fraction of examples where chosen response has higher reward
    """

    def __init__(self):
        self.loss_type = LossType.SEQUENCE_LEVEL

    def split_output_tensor(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        # tensor is of shape (2*micro_batch_size,)
        return tensor[::2], tensor[1::2]

    def _preference_loss(
        self,
        rewards: Tensor,
        sample_mask: Tensor,
        global_valid_seqs: Tensor,
        beta: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards)
        rewards_delta = rewards_chosen - rewards_rejected

        per_sample_loss = (
            -torch.nn.functional.logsigmoid(beta * rewards_delta) * sample_mask[::2]
        )  ## zero out invalid samples

        ## divide by 2 because each preference example corresponds to 2 samples (chosen, rejected)
        return (
            masked_mean(
                per_sample_loss,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_chosen > rewards_rejected,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_chosen,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_rejected,
                sample_mask[1::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
        )

    def __call__(
        self,
        rewards: Tensor,
        data: BatchedDataDict[PreferenceLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sample_mask = data["sample_mask"]

        rewards = rewards.squeeze(-1)

        (
            preference_loss,
            accuracy,
            rewards_chosen_mean,
            rewards_rejected_mean,
        ) = self._preference_loss(rewards, sample_mask, global_valid_seqs)

        ## divide by 2 because we're summing over (chosen, rejected) pairs
        num_valid_samples = sample_mask.sum() / 2

        return preference_loss, {
            "loss": preference_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "num_valid_samples": num_valid_samples.item(),
        }


class DPOLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    preference_loss_weight: float
    sft_loss_weight: float
    preference_average_log_probs: bool
    sft_average_log_probs: bool


class DPOLossDataDict(TypedDict):
    """Required keys for the DPO loss function."""

    input_ids: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class DPOLossFn(PreferenceLoss):
    """Direct Preference Optimization (DPO) loss function.

    This loss function implements the DPO algorithm as described in:
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    (https://arxiv.org/abs/2305.18290)

    The loss combines two main components:
    1. Preference Loss: Optimizes the model to prefer chosen responses over rejected ones
    2. SFT Loss (optional): Auxiliary supervised fine-tuning loss on chosen responses

    The total loss is computed as:
    L(θ) = w_p * L_pref(θ) + w_s * L_sft(θ)

    where:
    - w_p is the preference_loss_weight
    - w_s is the sft_loss_weight
    - L_pref(θ) is the preference loss term
    - L_sft(θ) is the supervised fine-tuning loss term

    The preference loss term is computed as:
    L_pref(θ) = -E[log(σ(β * (r_chosen - r_rejected)))]

    where:
    - σ is the sigmoid function
    - β is the reference_policy_kl_penalty
    - r_chosen and r_rejected are the rewards for chosen and rejected responses
    - The rewards are computed as the sum of log probability differences between
      the current policy and reference policy

    If preference_average_log_probs is True, the rewards are averaged over tokens:
    r = (1/n) * Σ_t (log π_θ(a_t|s_t) - log π_ref(a_t|s_t))

    Otherwise, the rewards are summed over tokens.

    The SFT loss term is a standard negative log likelihood loss on the chosen responses.
    If sft_average_log_probs is True, the loss is averaged over tokens.

    Args:
        cfg (DPOLossConfig): Configuration dictionary containing:
            - reference_policy_kl_penalty (float): Strength of the KL penalty term (β)
            - preference_loss_weight (float): Weight for the preference loss term (w_p)
            - sft_loss_weight (float): Weight for the SFT loss term (w_s)
            - preference_average_log_probs (bool): Whether to average log probs across tokens in preference loss
            - sft_average_log_probs (bool): Whether to average log probs across tokens in SFT loss

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing:
            - The total loss value
            - A dictionary with metrics including:
                - loss: Total loss value
                - sft_loss: SFT loss component
                - preference_loss: Preference loss component
                - accuracy: Fraction of examples where chosen response has higher reward
    """

    def __init__(self, cfg: DPOLossConfig):
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.preference_loss_weight = cfg["preference_loss_weight"]
        self.sft_loss_weight = cfg["sft_loss_weight"]
        self.preference_average_log_probs = cfg["preference_average_log_probs"]
        self.sft_average_log_probs = cfg["sft_average_log_probs"]
        self.sft_loss = NLLLoss()

        self.loss_type = LossType.SEQUENCE_LEVEL

    def _dpo_loss(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ## TODO(@ashors): there's some duplicate code here with the NLLLoss function. We should refactor
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        seq_index = data.get("seq_index", None)

        next_token_logits = next_token_logits.to(torch.float32)
        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            token_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            # slice off to the correct length to remove potential CP padding
            token_logprobs = token_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_tokens = data["input_ids"][:, 1:].cuda()  # Skip first token
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )
            logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits
            token_logprobs = logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        ref_logprobs = data["reference_policy_logprobs"][:, :-1]

        diff = (token_logprobs - ref_logprobs) * token_mask

        rewards = diff.sum(-1)
        if self.preference_average_log_probs:
            rewards = rewards / token_mask.sum(-1).clamp(min=1)

        return self._preference_loss(
            rewards, sample_mask, global_valid_seqs, self.reference_policy_kl_penalty
        )

    # TODO a cleaner typing fix would be required (probably that DPOLossFn should not inherit from PreferenceLoss)
    def __call__(  # type: ignore
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sft_loss_chosen = torch.tensor(0.0)
        if self.sft_loss_weight > 0:
            assert global_valid_toks is not None, (
                "global_valid_toks must be provided for SFT loss"
            )
            sft_loss, _ = self.sft_loss(
                next_token_logits,
                data,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,  ## unused because sft loss returned is at the sample level
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
                dpo_loss=True,
                dpo_average_log_probs=self.sft_average_log_probs,
            )
            sft_loss_chosen, sft_loss_rejected = self.split_output_tensor(sft_loss)
            sft_loss_chosen = masked_mean(
                sft_loss_chosen,
                data["sample_mask"][::2],
                global_normalization_factor=global_valid_seqs / 2,
            )

        (
            preference_loss,
            accuracy,
            rewards_chosen_mean,
            rewards_rejected_mean,
        ) = self._dpo_loss(
            next_token_logits,
            data,
            global_valid_seqs,
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
        )

        dpo_loss = (
            self.sft_loss_weight * sft_loss_chosen
            + self.preference_loss_weight * preference_loss
        )

        ## divide by 2 because we're summing over (chosen, rejected) pairs
        num_valid_samples = data["sample_mask"].sum() / 2

        return dpo_loss, {
            "loss": dpo_loss.item(),
            "sft_loss": sft_loss_chosen.item(),
            "preference_loss": preference_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "num_valid_samples": num_valid_samples.item(),
        }


class SequencePackingLossWrapper:
    def __init__(
        self,
        loss_fn: LossFunction,
        cu_seqlens_q: Tensor,
        cu_seqlens_q_padded: Optional[Tensor] = None,
    ):
        self.loss_fn = loss_fn
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_q_padded = cu_seqlens_q_padded

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: Tensor | None,
        global_valid_toks: Tensor | None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Wraps a loss function to handle sequence packing by doing one sequence at a time to avoid excessive padding."""
        unpadded_cu_seqlens = self.cu_seqlens_q
        unpadded_seq_lengths = self.cu_seqlens_q[1:] - self.cu_seqlens_q[:-1]
        if self.cu_seqlens_q_padded is not None:
            padded_cu_seqlens = self.cu_seqlens_q_padded
            padded_seq_lengths = (
                self.cu_seqlens_q_padded[1:] - self.cu_seqlens_q_padded[:-1]
            )
        else:
            padded_cu_seqlens = unpadded_cu_seqlens
            padded_seq_lengths = unpadded_seq_lengths
        seq_starts = padded_cu_seqlens[:-1]
        seq_ends = padded_cu_seqlens[1:]

        loss_accum = 0
        metrics_accum = {}
        for seq_idx in range(len(seq_starts)):
            seq_start = seq_starts[seq_idx].item()
            seq_end = seq_ends[seq_idx].item()

            # get sequence and unpad all 'data' tensors. The data dict is a BatchedDataDict of unpacked tensors
            seq_data = data.slice(seq_idx, seq_idx + 1)
            unpadded_seq_data = {}
            for k, v in seq_data.items():
                if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[1] > 1:
                    unpadded_seq_data[k] = v[:, : unpadded_seq_lengths[seq_idx]]
                else:
                    unpadded_seq_data[k] = v

            # get next_token_logits
            cp_size = (
                1
                if context_parallel_group is None
                else torch.distributed.get_world_size(context_parallel_group)
            )
            logit_start = seq_start // cp_size
            logit_end = (seq_start + padded_seq_lengths[seq_idx]) // cp_size
            logit_length = logit_end - logit_start
            next_token_logits_slice = next_token_logits.narrow(
                1, logit_start, logit_length
            )

            loss, metrics = self.loss_fn(
                next_token_logits_slice,
                unpadded_seq_data,
                global_valid_seqs,
                global_valid_toks,
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
            )
            loss_accum += loss
            for k, v in metrics.items():
                if k not in metrics_accum:
                    if k in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                        metrics_accum[k] = float("inf")
                    elif k in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                        metrics_accum[k] = float("-inf")
                    else:
                        metrics_accum[k] = 0

                val = v.item() if isinstance(v, torch.Tensor) and v.ndim == 0 else v

                # Skip inf/-inf sentinel values (from sequences with no valid tokens)
                if k in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                    if not math.isinf(val):
                        metrics_accum[k] = min(metrics_accum[k], val)
                elif k in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                    if not math.isinf(val):
                        metrics_accum[k] = max(metrics_accum[k], val)
                else:
                    metrics_accum[k] += val

        return loss_accum, metrics_accum


def _top_fraction_mask(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    keep_fraction: float,
) -> torch.Tensor:
    """Keep the highest-scoring fraction independently in each sequence."""
    valid = valid_mask.bool()
    valid_counts = valid.sum(dim=-1)
    keep_counts = torch.ceil(valid_counts.to(torch.float32) * keep_fraction).to(
        torch.long
    )

    ranked_indices = torch.argsort(
        scores.masked_fill(~valid, float("-inf")), dim=-1, descending=True
    )
    ranks = torch.empty_like(ranked_indices)
    sequence_positions = torch.arange(
        scores.shape[-1], device=scores.device, dtype=ranked_indices.dtype
    ).expand_as(ranked_indices)
    ranks.scatter_(dim=-1, index=ranked_indices, src=sequence_positions)
    return (ranks < keep_counts.unsqueeze(-1)) & valid


def _sequence_balanced_mean(
    values: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    global_valid_seqs: torch.Tensor,
) -> torch.Tensor:
    """Give every valid sequence equal total weight regardless of its length."""
    per_sequence_mean = masked_mean(values, token_mask, dim=-1)
    return masked_mean(
        per_sequence_mean,
        sample_mask,
        global_normalization_factor=global_valid_seqs,
    )


def _mean_normalized_token_weights(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize non-negative scores to mean one within each sequence."""
    valid = valid_mask.to(dtype=scores.dtype)
    valid_counts = valid.sum(dim=-1, keepdim=True)
    score_means = (scores * valid).sum(dim=-1, keepdim=True) / valid_counts.clamp_min(
        1.0
    )
    normalized = scores / score_means.clamp_min(eps)
    # A completely flat zero-margin sequence carries no confidence signal.
    # Fall back to baseline weights instead of silently dropping the sequence.
    normalized = torch.where(
        score_means > eps,
        normalized,
        torch.ones_like(normalized),
    )
    return normalized * valid


class TvdGateConfig(TypedDict, total=False):
    """Config for TVD-based token gating on top of the KL distillation loss.

    Gates the KL loss based on the truncated total-variation distance
    TVD_topk = 1 - sum_y min(p_S(y), p_T(y)) over the teacher's top-k support
    as the per-token disagreement signal. This is the OAD `acceptance`
    quantity turned into a *filter* rather than a loss target — direct OAD
    optimization was observed to collapse into repetition; using TVD only
    as a gate leaves the KL loss shape intact.

    Directions (which tokens PASS the gate):
        - "high" (default): keep iff TVD > τ.  Learn the tokens where
                            student and teacher disagree the most first.
                            Sweep τ 0→1: 0 keeps everything, 1 keeps nothing.
        - "low":            keep iff TVD < τ.  Curriculum-style: start by
                            learning tokens the student already almost
                            matches the teacher on, then gradually open up
                            to the harder / more-disagreeing tokens as τ
                            grows toward 1.

    Modes:
        - "none":   no gating (baseline; token_mask is untouched).
        - "fixed":  constant scalar τ from cfg["threshold"].
        - "warmup": τ anneals from `start_threshold` at step 0 to
                    `end_threshold` at global_step / max_num_steps ==
                    warmup_until_frac, then stays constant. S-shaped cosine
                    curve (matches prefix_length_warmup cosine mode).
        - "top_fraction": rank valid positions independently in every sequence
                          and keep a fixed fraction with the largest score.

    Scores for mode="top_fraction":
        - "confident_disagreement": TVD * (teacher top-1 prob - top-2 prob).
          This selects positions where teacher and student disagree while the
          teacher has a clear preferred next token.

    Reference: opd-improvements-proposal.md (future §), BASIC_OAD_PROPOSAL.md.
    """

    mode: str  # "none" | "fixed" | "warmup" | "top_fraction"
    direction: str  # "high" (default) | "low"
    threshold: float  # used when mode == "fixed"; interpretation depends on direction
    start_threshold: float  # used when mode == "warmup"; τ at step 0
    end_threshold: (
        float  # used when mode == "warmup"; τ at (and after) warmup_until_frac
    )
    warmup_until_frac: float  # fraction of max_num_steps to finish annealing
    score: str  # used when mode == "top_fraction"
    keep_fraction: float  # used when mode == "top_fraction"; in (0, 1]


class TeacherMarginWeightConfig(TypedDict, total=False):
    """Continuous token weights based on the teacher's top-1/top-2 margin."""

    enabled: bool
    power: float
    eps: float


class StopContentConfig(TypedDict):
    """Configuration for factorized content and stopping distillation."""

    enabled: bool
    eos_token_id: int
    stop_kl_type: str  # "forward" | "reverse"
    stop_kl_weight: float
    probability_eps: float


class EmaAnchorConfig(TypedDict, total=False):
    """Config for EMA-of-student anchor regularization.

    Adds `kl_weight * KL(p_S || p_EMA)` on the EMA's top-k support, where
    the EMA copy of the student weights is updated in-place each step as
    `theta_ema <- mu * theta_ema + (1 - mu) * theta_student` (starting from
    the initial student weights). Acts as a soft trust region "in time":
    the student is only lightly penalized for deviating from its recent
    trailing average, not from a frozen initial reference. Complementary to
    the main teacher KL: teacher pulls toward a distant target, EMA anchor
    dampens step-to-step drift.

    Physically reuses the same `reference_model_state_dict` slot as
    `reference_policy_kl_penalty` — the two features are mutually exclusive.
    """

    enabled: bool
    mu: float  # EMA decay; typical 0.995-0.999
    kl_weight: float  # lambda on KL(p_S || p_EMA)


class DistillationLossConfig(TypedDict):
    kl_type: str
    mixed_kl_weight: float
    zero_outside_topk: bool
    reduction: NotRequired[str]  # "token_mean" (default) | "sequence_mean"
    reference_policy_kl_penalty: NotRequired[float]
    reference_policy_kl_type: NotRequired[str]  # "k1" | "k2" | "k3"
    reference_policy_kl_input_clamp: NotRequired[float | None]
    reference_policy_kl_output_clamp: NotRequired[float | None]
    tvd_gate: NotRequired[TvdGateConfig]
    teacher_margin_weight: NotRequired[TeacherMarginWeightConfig]
    stop_content: NotRequired[StopContentConfig]
    ema_anchor: NotRequired[EmaAnchorConfig]
    # If set, drops these token ids from the loss mask at their *target*
    # positions (i.e. positions where the *next* token equals one of these
    # ids). Motivation: on-policy rollouts that stop early on wrong answers
    # put an EOS at a spot where the teacher — being a strong reasoner —
    # prefers "wait/but/however" continuations. Training on these positions
    # suppresses p_S(EOS), so the student stops emitting EOS and rollouts
    # grow until they hit max_length. Common values: DeepSeek-R1-Distill
    # tokenizer uses eos_token_id=151643.
    mask_eos_positions: NotRequired[list[int]]


class DistillationLossDataDict(TypedDict):
    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    teacher_topk_logits: torch.Tensor
    teacher_topk_indices: torch.Tensor
    seq_index: NotRequired[torch.Tensor]
    # Log-probability assigned to each input token by the frozen initial
    # student. Position 0 follows the policy API convention and is always 0.
    reference_policy_logprobs: NotRequired[torch.Tensor]
    # Optional: exact full-vocab logsumexp of the teacher, needed by the TVD
    # gate to convert teacher_topk_logits into true global probabilities.
    # Populated by the DTensor teacher worker (see Path B plumbing).
    teacher_logsumexp: NotRequired[torch.Tensor]
    # Exact teacher EOS logit at every position. Required by stop-content
    # factorization even when EOS is not among the teacher's top-k tokens.
    teacher_eos_logits: NotRequired[torch.Tensor]
    # Top-k logits/indices of the student EMA copy, on the EMA's OWN top-k
    # support. Populated by the student policy when ema_anchor is enabled.
    ema_topk_logits: NotRequired[torch.Tensor]
    ema_topk_indices: NotRequired[torch.Tensor]


class DistillationLossFn(LossFunction):
    """Distillation loss function."""

    def __init__(self, cfg: DistillationLossConfig):
        self.kl_type = cfg["kl_type"]
        self.mixed_kl_weight = cfg["mixed_kl_weight"]
        self.zero_outside_topk = cfg["zero_outside_topk"]
        self.reduction = str(cfg.get("reduction", "token_mean"))
        self.reference_policy_kl_penalty = float(
            cfg.get("reference_policy_kl_penalty", 0.0)
        )
        self.reference_policy_kl_type = str(cfg.get("reference_policy_kl_type", "k3"))
        self.reference_policy_kl_input_clamp = cfg.get(
            "reference_policy_kl_input_clamp", 20.0
        )
        self.reference_policy_kl_output_clamp = cfg.get(
            "reference_policy_kl_output_clamp", 10.0
        )
        self.log_infinitesimal = -100
        self.loss_type = LossType.TOKEN_LEVEL

        assert self.kl_type in ["forward", "reverse", "mixed"], "Invalid KL type"
        assert self.mixed_kl_weight >= 0 and self.mixed_kl_weight <= 1, (
            "Invalid mixed KL weight"
        )
        if self.reduction not in ("token_mean", "sequence_mean"):
            raise ValueError(
                f"Unknown loss_fn.reduction={self.reduction!r}. "
                "Expected one of: 'token_mean', 'sequence_mean'."
            )
        if self.reference_policy_kl_penalty < 0.0:
            raise ValueError(
                "loss_fn.reference_policy_kl_penalty must be non-negative."
            )
        if self.reference_policy_kl_type not in ("k1", "k2", "k3"):
            raise ValueError(
                "loss_fn.reference_policy_kl_type must be one of: 'k1', 'k2', 'k3'."
            )

        # Optional EMA-of-student anchor. Adds KL(p_S || p_EMA) on the EMA's
        # own top-k support. Physically shares the reference_model_state_dict
        # slot with reference-policy KL, so the two are mutually exclusive.
        ema_cfg = cfg.get("ema_anchor")
        self.ema_anchor_enabled = bool(
            ema_cfg is not None and ema_cfg.get("enabled", False)
        )
        self.ema_anchor_mu = 0.999
        self.ema_anchor_kl_weight = 0.0
        if self.ema_anchor_enabled:
            assert ema_cfg is not None
            self.ema_anchor_mu = float(ema_cfg.get("mu", 0.999))
            self.ema_anchor_kl_weight = float(ema_cfg.get("kl_weight", 0.0))
            if not (0.0 <= self.ema_anchor_mu < 1.0):
                raise ValueError("loss_fn.ema_anchor.mu must be in [0, 1).")
            if self.ema_anchor_kl_weight < 0.0:
                raise ValueError(
                    "loss_fn.ema_anchor.kl_weight must be non-negative."
                )
            if self.reference_policy_kl_penalty > 0.0:
                raise ValueError(
                    "ema_anchor and reference_policy_kl_penalty are mutually "
                    "exclusive (both reuse the reference_model_state_dict slot)."
                )

        self.stop_content_cfg = cfg.get("stop_content")
        self.stop_content_enabled = bool(
            self.stop_content_cfg is not None
            and self.stop_content_cfg.get("enabled", False)
        )
        self.stop_content_eos_token_id = -1
        self.stop_content_stop_kl_type = "reverse"
        self.stop_content_stop_kl_weight = 1.0
        self.stop_content_probability_eps = 1.0e-7
        if self.stop_content_enabled:
            assert self.stop_content_cfg is not None
            for required_key in (
                "eos_token_id",
                "stop_kl_type",
                "stop_kl_weight",
                "probability_eps",
            ):
                assert required_key in self.stop_content_cfg, (
                    f"loss_fn.stop_content.enabled=true requires {required_key!r}."
                )
            assert self.kl_type == "reverse", (
                "stop-content factorization requires loss_fn.kl_type='reverse' "
                "for the conditional content term."
            )
            self.stop_content_eos_token_id = int(self.stop_content_cfg["eos_token_id"])
            self.stop_content_stop_kl_type = str(self.stop_content_cfg["stop_kl_type"])
            self.stop_content_stop_kl_weight = float(
                self.stop_content_cfg["stop_kl_weight"]
            )
            self.stop_content_probability_eps = float(
                self.stop_content_cfg["probability_eps"]
            )
            assert self.stop_content_stop_kl_type in ("forward", "reverse"), (
                "loss_fn.stop_content.stop_kl_type must be 'forward' or 'reverse'."
            )
            assert self.stop_content_eos_token_id >= 0, (
                "loss_fn.stop_content.eos_token_id must be non-negative."
            )
            assert self.stop_content_stop_kl_weight >= 0.0, (
                "loss_fn.stop_content.stop_kl_weight must be non-negative."
            )
            assert 0.0 < self.stop_content_probability_eps < 0.5, (
                "loss_fn.stop_content.probability_eps must be in (0, 0.5)."
            )

        # Optional list of token ids to drop from the loss mask at target
        # positions. See DistillationLossConfig.mask_eos_positions docstring.
        raw_mask_eos = cfg.get("mask_eos_positions")
        if raw_mask_eos is None:
            self.mask_eos_positions: list[int] = []
        else:
            self.mask_eos_positions = [int(x) for x in raw_mask_eos]

        self.teacher_margin_weight_cfg = cfg.get("teacher_margin_weight")
        self.teacher_margin_weight_enabled = bool(
            self.teacher_margin_weight_cfg is not None
            and self.teacher_margin_weight_cfg.get("enabled", False)
        )
        self.teacher_margin_weight_power = 1.0
        self.teacher_margin_weight_eps = 1.0e-8
        if self.teacher_margin_weight_enabled:
            assert self.teacher_margin_weight_cfg is not None
            self.teacher_margin_weight_power = float(
                self.teacher_margin_weight_cfg.get("power", 1.0)
            )
            self.teacher_margin_weight_eps = float(
                self.teacher_margin_weight_cfg.get("eps", 1.0e-8)
            )
            assert self.teacher_margin_weight_power > 0.0, (
                "loss_fn.teacher_margin_weight.power must be positive."
            )
            assert self.teacher_margin_weight_eps > 0.0, (
                "loss_fn.teacher_margin_weight.eps must be positive."
            )

        # TVD gate: uses the OAD acceptance quantity as a *filter* on which
        # tokens contribute to the KL loss (not as a loss target itself). The
        # main training loop stamps `_tvd_gate_state` before each train call
        # with the current global_step / max_num_steps + resolved threshold.
        # When mode == "none" (or no gate cfg at all) the gate is off — the
        # loss is byte-identical to a baseline run.
        self.tvd_gate_cfg = cfg.get("tvd_gate")
        # Direction is a stable-per-run choice, not a per-step state, so we
        # freeze it at __init__ time from cfg (default "high" preserves the
        # semantics of every experiment prior to this feature).
        self.tvd_gate_direction: str = (
            self.tvd_gate_cfg.get("direction", "high")
            if self.tvd_gate_cfg is not None
            else "high"
        )
        self._tvd_gate_state: dict[str, Any] = {
            "tau": float("-inf"),
            "mode": "none",
        }

        # A gate needs true global student probabilities on the teacher's
        # top-k support. Keep that scoring path independent from the KL's
        # zero_outside_topk choice: a gate may score with global probabilities
        # while the loss retains the baseline top-k conditional KL.
        gate_mode = (
            self.tvd_gate_cfg.get("mode", "none")
            if self.tvd_gate_cfg is not None
            else "none"
        )
        self.tvd_gate_config_mode = gate_mode
        if self.teacher_margin_weight_enabled and gate_mode != "none":
            raise ValueError(
                "teacher_margin_weight and tvd_gate cannot be enabled together. "
                "Run confidence weighting as an isolated baseline ablation."
            )
        if self.ema_anchor_enabled and gate_mode != "none":
            # Gate-active main loss is an *unnormalized* token-loss sum
            # (worker divides by kept-token count later). EMA-anchor's
            # masked_mean is already normalized, so summing the two would
            # rescale EMA-anchor's gradient by 1/kept_tokens. Refuse the
            # combination until we route EMA through the same delayed
            # normalization path.
            raise ValueError(
                "ema_anchor and tvd_gate cannot be enabled together: the "
                "gate defers loss normalization to the worker, which would "
                "double-normalize the EMA-anchor term. Disable one."
            )
        # A TVD-gated loss is accumulated as an unnormalized token-loss sum.
        # The DTensor worker sees every microbatch and DP shard, so it can divide
        # the accumulated gradients by the exact global number of kept tokens
        # once, immediately before clipping/stepping. Normalizing here would only
        # see one local microbatch and would weight microbatches/ranks incorrectly.
        self.normalize_by_kept_tokens = (
            gate_mode != "none" and self.reduction == "token_mean"
        )
        if self.reference_policy_kl_penalty > 0.0 and gate_mode != "none":
            raise ValueError(
                "reference-policy KL is not yet compatible with tvd_gate: the "
                "gate's delayed kept-token normalization would also rescale the "
                "ungated reference penalty. Run it without a TVD gate."
            )
        if gate_mode != "none":
            # Reject unknown modes up-front so users see the full valid list.
            if gate_mode not in ("fixed", "warmup", "top_fraction"):
                raise ValueError(
                    f"Unknown tvd_gate.mode={gate_mode!r}. "
                    "Expected one of: 'none', 'fixed', 'warmup', 'top_fraction'."
                )
            if gate_mode != "top_fraction" and self.tvd_gate_direction not in (
                "high",
                "low",
            ):
                raise ValueError(
                    f"Unknown tvd_gate.direction={self.tvd_gate_direction!r}. "
                    "Expected one of: 'high', 'low'."
                )
            if gate_mode == "fixed":
                assert "threshold" in self.tvd_gate_cfg, (
                    "tvd_gate.mode='fixed' requires 'threshold' in config."
                )
            elif gate_mode == "warmup":
                for required_key in (
                    "start_threshold",
                    "end_threshold",
                    "warmup_until_frac",
                ):
                    assert required_key in self.tvd_gate_cfg, (
                        f"tvd_gate.mode='warmup' requires {required_key!r} in config."
                    )
            else:  # gate_mode == "top_fraction"
                assert self.tvd_gate_cfg.get("score") == "confident_disagreement", (
                    "loss_fn.tvd_gate.mode='top_fraction' currently requires "
                    "score='confident_disagreement'."
                )
                assert "keep_fraction" in self.tvd_gate_cfg, (
                    "loss_fn.tvd_gate.mode='top_fraction' requires "
                    "'keep_fraction' in config."
                )
                keep_fraction = float(self.tvd_gate_cfg["keep_fraction"])
                assert 0.0 < keep_fraction <= 1.0, (
                    "loss_fn.tvd_gate.keep_fraction must be in (0, 1]."
                )

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: DistillationLossDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute distillation loss between teacher and student logits."""
        # Basic shapes
        input_ids = data["input_ids"]
        batch_size = input_ids.shape[0]

        # CP support: get CP group and size
        cp_group = context_parallel_group
        cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)

        # Ensure float32 for stability (match other losses)
        next_token_logits = next_token_logits.to(torch.float32)
        current_token_logprobs: Optional[torch.Tensor] = None
        if self.reference_policy_kl_penalty > 0.0:
            if "reference_policy_logprobs" not in data:
                raise KeyError(
                    "reference_policy_kl_penalty > 0 requires "
                    "`reference_policy_logprobs` in train_data."
                )
            if vocab_parallel_group is not None:
                assert vocab_parallel_rank is not None
                current_token_logprobs = from_parallel_logits_to_logprobs(
                    next_token_logits,
                    input_ids,
                    vocab_start_index=(
                        vocab_parallel_rank * next_token_logits.shape[-1]
                    ),
                    vocab_end_index=(
                        (vocab_parallel_rank + 1) * next_token_logits.shape[-1]
                    ),
                    tp_group=vocab_parallel_group,
                    inference_only=False,
                    cp_group=context_parallel_group,
                )
                current_token_logprobs = current_token_logprobs[
                    :, : input_ids.shape[1] - 1
                ]
            elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
                current_token_logprobs = get_logprobs_from_vocab_parallel_logits(
                    next_token_logits,
                    input_ids,
                    seq_index=data.get("seq_index"),
                )
            else:
                current_logits = next_token_logits[:, :-1]
                next_tokens = input_ids[:, 1:].to(current_logits.device)
                current_token_logprobs = (
                    torch.nn.functional.log_softmax(current_logits, dim=-1)
                    .gather(dim=-1, index=next_tokens.unsqueeze(-1))
                    .squeeze(-1)
                )
        per_token_kl = None
        # Preferred truncated-KL path: teacher provides top-k support per position
        teacher_topk_logits = data["teacher_topk_logits"]  # [B, S, k]
        teacher_topk_indices = data["teacher_topk_indices"]  # [B, S, k]

        if teacher_topk_indices.shape[-1] <= 0:
            raise ValueError(
                f"topk must be positive, got {teacher_topk_indices.shape[-1]}. "
                "topk=0 is not supported as it would result in empty tensor operations."
            )
        if self.stop_content_enabled and teacher_topk_indices.shape[-1] < 2:
            raise ValueError(
                "stop-content factorization requires topk >= 2 so the conditional "
                "content support remains non-empty when EOS is in the top-k."
            )

        # Determine processing path and setup variables
        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            V_local = int(next_token_logits.shape[-1])
            vocab_start_index = vocab_parallel_rank * V_local
            vocab_end_index = (vocab_parallel_rank + 1) * V_local
            parallel_group = vocab_parallel_group
            logits_tensor = next_token_logits
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            device_mesh = next_token_logits.device_mesh
            tp_group = device_mesh.get_group("tp")
            tp_rank = tp_group.rank()
            local_student_logits = next_token_logits.to_local()
            V_local = int(local_student_logits.shape[-1])
            vocab_start_index = tp_rank * V_local
            vocab_end_index = (tp_rank + 1) * V_local
            parallel_group = tp_group
            logits_tensor = local_student_logits
            teacher_topk_indices = teacher_topk_indices.to(local_student_logits.device)
            # For DTensor, derive CP group/size from the device mesh to ensure CP-aware alignment
            if (
                device_mesh.mesh_dim_names is not None
                and "cp" in device_mesh.mesh_dim_names
            ):
                cp_group = device_mesh.get_group("cp")
                cp_size = cp_group.size()
            else:
                cp_group = None
                cp_size = 1
        else:
            parallel_group = None
            logits_tensor = next_token_logits

        # EOS is frequently outside the teacher's top-k support. Gather its
        # exact full-vocabulary student log-probability separately so the stop
        # decision does not inherit the top-k approximation used for content.
        student_eos_logprobs: Optional[torch.Tensor] = None
        if self.stop_content_enabled:
            eos_indices = torch.full(
                (*teacher_topk_indices.shape[:2], 1),
                self.stop_content_eos_token_id,
                dtype=torch.long,
                device=teacher_topk_indices.device,
            )
            if parallel_group is not None:
                eos_indices_local = eos_indices
                eos_pad_len = 0
                if cp_size > 1:
                    eos_pad_len = (
                        logits_tensor.shape[1] * cp_size - eos_indices_local.shape[1]
                    )
                    if eos_pad_len > 0:
                        eos_indices_local = torch.nn.functional.pad(
                            eos_indices_local, (0, 0, 0, eos_pad_len), value=0
                        )
                    cp_rank = torch.distributed.get_rank(cp_group)
                    eos_indices_local = _get_tokens_on_this_cp_rank(
                        eos_indices_local, cp_rank, cp_size, seq_dim=1
                    )

                eos_chunk_size = max(1, min(int(logits_tensor.shape[1]), 1024))
                student_eos_logprobs = ChunkedDistributedGatherLogprob.apply(  # type: ignore
                    logits_tensor,
                    eos_indices_local,
                    vocab_start_index,
                    vocab_end_index,
                    eos_chunk_size,
                    parallel_group,
                    False,
                )
                if cp_size > 1:
                    student_eos_logprobs = allgather_cp_sharded_tensor(
                        student_eos_logprobs, cp_group, seq_dim=1
                    )
                    if eos_pad_len > 0:
                        student_eos_logprobs = student_eos_logprobs[:, :-eos_pad_len, :]
            else:
                student_full_logprobs = torch.nn.functional.log_softmax(
                    logits_tensor, dim=-1
                )
                student_eos_logprobs = student_full_logprobs.gather(
                    dim=-1, index=eos_indices.to(student_full_logprobs.device)
                )

        # TVD/confident-disagreement scoring requires probabilities normalized
        # over the full vocabulary. Compute them independently of the KL path;
        # when the KL is top-k conditional this branch is diagnostic-only and
        # must not add another gradient path.
        needs_global_student_topk = (
            self.zero_outside_topk or self.tvd_gate_config_mode != "none"
        )
        student_topk_global_logprobs: Optional[torch.Tensor] = None
        if needs_global_student_topk and parallel_group is not None:
            indices_local = teacher_topk_indices
            pad_len = 0
            if cp_size > 1:
                pad_len = logits_tensor.shape[1] * cp_size - indices_local.shape[1]
                if pad_len > 0:
                    indices_local = torch.nn.functional.pad(
                        indices_local, (0, 0, 0, pad_len), value=0
                    )
                cp_rank = torch.distributed.get_rank(cp_group)
                indices_local = _get_tokens_on_this_cp_rank(
                    indices_local, cp_rank, cp_size, seq_dim=1
                )

            S_local = int(logits_tensor.shape[1])
            chunk_size = max(1, min(S_local, 1024))
            with torch.set_grad_enabled(self.zero_outside_topk):
                student_topk_global_logprobs = ChunkedDistributedGatherLogprob.apply(  # type: ignore
                    logits_tensor,
                    indices_local,
                    vocab_start_index,
                    vocab_end_index,
                    chunk_size,
                    parallel_group,
                    False,
                )

                if self.zero_outside_topk and self.kl_type != "forward":
                    H_all = ChunkedDistributedEntropy.apply(  # type: ignore
                        logits_tensor,
                        chunk_size,
                        parallel_group,
                        False,
                    )

            if cp_size > 1:
                student_topk_global_logprobs = allgather_cp_sharded_tensor(
                    student_topk_global_logprobs, cp_group, seq_dim=1
                )
                if self.zero_outside_topk and self.kl_type != "forward":
                    H_all = allgather_cp_sharded_tensor(H_all, cp_group, seq_dim=1)
                if pad_len > 0:
                    student_topk_global_logprobs = student_topk_global_logprobs[
                        :, :-pad_len, :
                    ]
                    if self.zero_outside_topk and self.kl_type != "forward":
                        H_all = H_all[:, :-pad_len]
        elif needs_global_student_topk:
            with torch.set_grad_enabled(self.zero_outside_topk):
                student_full_logprobs = torch.nn.functional.log_softmax(
                    logits_tensor,
                    dim=-1,
                )
                student_topk_global_logprobs = student_full_logprobs.gather(
                    dim=-1,
                    index=teacher_topk_indices.to(student_full_logprobs.device),
                )
                if self.zero_outside_topk and self.kl_type != "forward":
                    H_all = (student_full_logprobs.exp() * student_full_logprobs).sum(
                        -1
                    )

        if self.zero_outside_topk:
            assert student_topk_global_logprobs is not None
            student_topk_logprobs = student_topk_global_logprobs
        else:
            # self.zero_outside_topk = False
            # 把 teacher 和 student 的分布都截断到 top-k，然后在这 k 个 token 上重新归一化，再计算 KL 散度
            # Gather logits at global indices
            if (parallel_group is not None) or (cp_size > 1):
                student_topk_logits = gather_logits_at_global_indices(
                    logits_tensor,
                    teacher_topk_indices,
                    tp_group=parallel_group,
                    cp_group=cp_group,
                    vocab_start_index=(
                        vocab_start_index if parallel_group is not None else 0
                    ),
                    vocab_end_index=(
                        vocab_end_index
                        if parallel_group is not None
                        else int(logits_tensor.shape[-1])
                    ),
                )
            else:
                # 从学生完整 logits 中，只取 teacher top-k 位置的原始 logits
                # [B, S, V] → [B, S, k]
                student_topk_logits = logits_tensor.gather(
                    dim=-1, index=teacher_topk_indices.to(logits_tensor.device)
                )
            # 在 k 维度上做 log_softmax（重新归一化到 k 个 token 上） [B, S, k]
            student_topk_logprobs = torch.nn.functional.log_softmax(
                student_topk_logits, dim=-1
            )

        # Move teacher tensors to the same device/dtype as student_topk_logits
        teacher_topk_logits = teacher_topk_logits.to(
            student_topk_logprobs.device, dtype=student_topk_logprobs.dtype
        )
        # 在 k 维度上做 log_softmax [B, S, k]
        teacher_topk_logprobs = torch.nn.functional.log_softmax(
            teacher_topk_logits, dim=-1
        )

        # Single point of next-token alignment after TP/CP processing
        teacher_topk_logprobs = teacher_topk_logprobs[:, :-1, :]
        student_topk_logprobs = student_topk_logprobs[:, :-1, :]
        if student_topk_global_logprobs is not None:
            student_topk_global_logprobs = student_topk_global_logprobs[:, :-1, :]
        if self.zero_outside_topk and self.kl_type != "forward":
            # Align H_all with next-token prediction
            H_all = H_all[:, :-1]

        # 预测下一个 token
        student_probs = student_topk_logprobs.exp()  # [B, S-1, k]
        teacher_probs = teacher_topk_logprobs.exp()  # [B, S-1, k]

        content_kl_component: Optional[torch.Tensor] = None
        stop_kl_component: Optional[torch.Tensor] = None
        student_stop_probability: Optional[torch.Tensor] = None
        teacher_stop_probability: Optional[torch.Tensor] = None

        loss_correction_term = torch.zeros_like(student_probs[..., 0])  # [B, S-1]
        if self.zero_outside_topk and self.kl_type != "forward":
            H_rest = H_all - (student_probs * student_topk_logprobs).sum(-1)
            P_rest = 1 - (student_probs.sum(-1))
            # The entropy and prob of the rest of the tokens [B, S-1]
            loss_correction_term = H_rest - self.log_infinitesimal * P_rest  # [B, S-1]
            if self.kl_type == "mixed":
                loss_correction_term = loss_correction_term * (
                    1.0 - self.mixed_kl_weight
                )

        if self.stop_content_enabled:
            if "teacher_logsumexp" not in data or "teacher_eos_logits" not in data:
                raise KeyError(
                    "stop-content factorization requires `teacher_logsumexp` and "
                    "`teacher_eos_logits` from the teacher DTensor worker."
                )
            assert student_eos_logprobs is not None

            # Conditional content distributions on the same teacher top-k
            # support as the existing baseline, but with EOS removed before
            # normalization. This isolates "what to write next" from whether
            # the model should stop.
            content_support = (
                teacher_topk_indices[:, :-1, :].to(student_topk_logprobs.device)
                != self.stop_content_eos_token_id
            )
            teacher_content_scores = teacher_topk_logits[:, :-1, :].masked_fill(
                ~content_support, float("-inf")
            )
            student_content_scores = student_topk_logprobs.masked_fill(
                ~content_support, float("-inf")
            )
            teacher_content_logprobs = torch.nn.functional.log_softmax(
                teacher_content_scores, dim=-1
            )
            student_content_logprobs = torch.nn.functional.log_softmax(
                student_content_scores, dim=-1
            )
            student_content_probs = student_content_logprobs.exp()
            content_log_ratio = torch.where(
                content_support,
                student_content_logprobs - teacher_content_logprobs,
                torch.zeros_like(student_content_logprobs),
            )
            content_kl_component = (student_content_probs * content_log_ratio).sum(
                dim=-1
            )

            eps = self.stop_content_probability_eps
            student_stop_probability = (
                student_eos_logprobs[:, :-1, 0].exp().clamp(min=eps, max=1.0 - eps)
            )
            teacher_stop_logprob = (
                data["teacher_eos_logits"].to(
                    device=student_topk_logprobs.device, dtype=torch.float32
                )
                - data["teacher_logsumexp"].to(
                    device=student_topk_logprobs.device, dtype=torch.float32
                )
            )[:, :-1]
            teacher_stop_probability = teacher_stop_logprob.exp().clamp(
                min=eps, max=1.0 - eps
            )

            log_student_stop = student_stop_probability.log()
            log_teacher_stop = teacher_stop_probability.log()
            log_student_continue = torch.log1p(-student_stop_probability)
            log_teacher_continue = torch.log1p(-teacher_stop_probability)
            if self.stop_content_stop_kl_type == "forward":
                stop_kl_component = teacher_stop_probability * (
                    log_teacher_stop - log_student_stop
                ) + (1.0 - teacher_stop_probability) * (
                    log_teacher_continue - log_student_continue
                )
            else:
                stop_kl_component = student_stop_probability * (
                    log_student_stop - log_teacher_stop
                ) + (1.0 - student_stop_probability) * (
                    log_student_continue - log_teacher_continue
                )

            # Reverse-KL chain rule weights the conditional content term by
            # the student's probability of continuing. A forward stop term
            # changes only the binary termination geometry.
            per_token_kl = (
                (1.0 - student_stop_probability) * content_kl_component
                + self.stop_content_stop_kl_weight * stop_kl_component
            )
        elif self.kl_type == "forward":
            # p_i * (log p_i - log q_i)  = p_i * log(p_i / q_i)
            # [B, S-1, k] = [B, S-1, k] * [B, S-1, k]
            per_token_kl = teacher_probs * (
                teacher_topk_logprobs - student_topk_logprobs
            )
        elif self.kl_type == "reverse":
            # q_i * (log q_i - log p_i)  = q_i * log(q_i / p_i)
            # [B, S-1, k] = [B, S-1, k] * [B, S-1, k]
            per_token_kl = student_probs * (
                student_topk_logprobs - teacher_topk_logprobs
            )
        else:
            #  examples/configs/distillation_math.yaml mixed KL
            # [B, S-1, k] = [B, S-1, k] * [B, S-1, k]
            kl_forward = teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)
            kl_reverse = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
            per_token_kl = (
                self.mixed_kl_weight * kl_forward
                + (1.0 - self.mixed_kl_weight) * kl_reverse
            )

        # [B, S-1, k] → [B, S-1]. The factorized path already reduced the
        # support dimension above.
        if not self.stop_content_enabled:
            per_token_kl = per_token_kl.sum(dim=-1) + loss_correction_term

        # Masking and reduction.
        # The optional TVD gate lives entirely inside the branch that has
        # both token_mask and sample_mask — attempting to gate when there is
        # no per-token mask is meaningless (per_token_kl.mean() has no notion
        # of "keep this position").
        tvd_gate_mode = str(self._tvd_gate_state.get("mode", "none"))
        tvd_gate_tau = float(self._tvd_gate_state.get("tau", float("-inf")))
        tvd_gate_active = tvd_gate_mode != "none"
        confidence_base_sum: Optional[torch.Tensor] = None
        confidence_margin_sum: Optional[torch.Tensor] = None
        confidence_weight_sum: Optional[torch.Tensor] = None
        confidence_weight_sq_sum: Optional[torch.Tensor] = None
        reference_policy_kl_loss: Optional[torch.Tensor] = None
        distillation_kl_loss: Optional[torch.Tensor] = None
        ema_kl_loss: Optional[torch.Tensor] = None

        if "token_mask" in data and "sample_mask" in data:
            token_mask = data["token_mask"][:, 1:]
            sample_mask = data["sample_mask"]
            # Align mask length to current per_token_kl
            max_len = per_token_kl.shape[1]
            token_mask = token_mask[:, :max_len]
            base_mask = token_mask * sample_mask.unsqueeze(-1)  # [B, S-1]

            # Drop target positions whose target token is an EOS-like id.
            # Rationale: on wrong-answer rollouts the student places EOS at
            # a spot where the teacher (a strong reasoner) prefers to keep
            # writing. Training that position suppresses p_S(EOS) — student
            # then never stops, sequences grow to max_length. Applied before
            # any TVD gate so the gate never sees these positions.
            if self.mask_eos_positions:
                # `input_ids` is a CP buffer, so under context parallelism the
                # worker hands us a seq-sharded DTensor. Two traps here:
                #   1. base_mask is a plain tensor (token_mask/sample_mask are
                #      not CP buffers), and plain * DTensor raises.
                #   2. CP sharding is load-balanced, i.e. rank r holds chunks
                #      r and 2*cp-1-r, so the local shards are *permuted* with
                #      respect to sequence order. `.full_tensor()` would
                #      silently give us out-of-order ids and mask the wrong
                #      positions; allgather_cp_sharded_tensor undoes the
                #      permutation the same way it does for student logprobs.
                ids = data["input_ids"]
                if isinstance(ids, torch.distributed.tensor.DTensor):
                    ids = ids.to_local()
                    if cp_size > 1:
                        ids = allgather_cp_sharded_tensor(ids, cp_group, seq_dim=1)
                # Target for position t is ids[:, t+1]; per_token_kl is already
                # aligned to targets, so slice [:, 1:1+max_len] — the same
                # convention token_mask[:, 1:][:, :max_len] uses above. `ids`
                # may be padded to a multiple of 2*cp, which the slice drops.
                target_ids = ids[:, 1 : 1 + max_len].to(device=base_mask.device)
                not_eos = torch.ones_like(base_mask)
                for eos_id in self.mask_eos_positions:
                    not_eos = not_eos * (target_ids != eos_id).to(dtype=base_mask.dtype)
                base_mask = base_mask * not_eos

            mask = base_mask

            tvd_topk: Optional[torch.Tensor] = None
            base_valid_sum: Optional[torch.Tensor] = None
            kept_valid_sum: Optional[torch.Tensor] = None
            tvd_sum: Optional[torch.Tensor] = None
            teacher_margin_sum: Optional[torch.Tensor] = None
            confident_disagreement_sum: Optional[torch.Tensor] = None
            selected_confident_disagreement_sum: Optional[torch.Tensor] = None

            teacher_p_topk_true: Optional[torch.Tensor] = None
            teacher_margin: Optional[torch.Tensor] = None

            if tvd_gate_active or self.teacher_margin_weight_enabled:
                if "teacher_logsumexp" not in data:
                    feature_name = (
                        "tvd_gate" if tvd_gate_active else "teacher_margin_weight"
                    )
                    raise KeyError(
                        f"{feature_name} requires `teacher_logsumexp` in train_data "
                        "(Path B). The teacher worker must return exact "
                        "full-vocab logsumexp (DTensor backend only)."
                    )
                needs_teacher_margin = (
                    self.teacher_margin_weight_enabled
                    or tvd_gate_mode == "top_fraction"
                )
                if needs_teacher_margin and teacher_topk_logits.shape[-1] < 2:
                    raise ValueError(
                        "teacher confidence weighting requires teacher topk >= 2."
                    )
                teacher_lse = data["teacher_logsumexp"].to(
                    device=student_topk_logprobs.device, dtype=torch.float32
                )
                teacher_true_log_p_topk = (
                    teacher_topk_logits.to(torch.float32) - teacher_lse.unsqueeze(-1)
                )[:, :-1, :]
                teacher_p_topk_true = teacher_true_log_p_topk.exp()

            if self.teacher_margin_weight_enabled:
                assert teacher_p_topk_true is not None
                top_two_teacher_probs = torch.topk(
                    teacher_p_topk_true, k=2, dim=-1
                ).values
                teacher_margin = (
                    top_two_teacher_probs[..., 0] - top_two_teacher_probs[..., 1]
                ).clamp_min(0.0)
                confidence_scores = teacher_margin.pow(self.teacher_margin_weight_power)
                confidence_weights = _mean_normalized_token_weights(
                    confidence_scores,
                    base_mask,
                    eps=self.teacher_margin_weight_eps,
                ).to(base_mask.dtype)
                mask = base_mask * confidence_weights
                with torch.no_grad():
                    confidence_base_sum = base_mask.sum()
                    confidence_margin_sum = (teacher_margin * base_mask).sum()
                    confidence_weight_sum = mask.sum()
                    confidence_weight_sq_sum = mask.square().sum()

            if tvd_gate_active:
                # Gate scores use true global probabilities regardless of
                # whether the KL uses a global or top-k conditional student
                # distribution.
                if student_topk_global_logprobs is None:
                    raise RuntimeError(
                        "tvd_gate requires global student top-k probabilities."
                    )
                assert teacher_p_topk_true is not None
                student_p_topk_true = student_topk_global_logprobs.to(
                    torch.float32
                ).exp()
                acceptance_topk = torch.minimum(
                    student_p_topk_true, teacher_p_topk_true
                ).sum(-1)  # [B, S-1]
                tvd_topk = (1.0 - acceptance_topk).clamp(0.0, 1.0)  # [B, S-1]

                if tvd_gate_mode == "top_fraction":
                    if teacher_margin is None:
                        top_two_teacher_probs = torch.topk(
                            teacher_p_topk_true, k=2, dim=-1
                        ).values
                        teacher_margin = (
                            top_two_teacher_probs[..., 0]
                            - top_two_teacher_probs[..., 1]
                        )
                    confident_disagreement = tvd_topk * teacher_margin
                    gate_w = _top_fraction_mask(
                        confident_disagreement,
                        base_mask,
                        keep_fraction=tvd_gate_tau,
                    ).to(base_mask.dtype)
                else:
                    # Strict comparison leaves the boundary outside the loss.
                    if self.tvd_gate_direction == "low":
                        gate_w = (tvd_topk < tvd_gate_tau).to(base_mask.dtype)
                    else:
                        gate_w = (tvd_topk > tvd_gate_tau).to(base_mask.dtype)
                # These sums are computed once and reused: mask -> loss, and
                # base/kept/tvd sums -> diagnostics. The gated branch returns an
                # unnormalized numerator below. The DTensor worker accumulates
                # all microbatches, all-reduces kept_valid_sum over DP, then
                # divides the accumulated gradients by that exact global count.
                mask = mask * gate_w
                with torch.no_grad():
                    base_valid_sum = base_mask.sum()
                    kept_valid_sum = mask.sum()
                    tvd_sum = (tvd_topk * base_mask).sum()
                    if tvd_gate_mode == "top_fraction":
                        teacher_margin_sum = (teacher_margin * base_mask).sum()
                        confident_disagreement_sum = (
                            confident_disagreement * base_mask
                        ).sum()
                        selected_confident_disagreement_sum = (
                            confident_disagreement * mask
                        ).sum()

            # For a gated loss, defer normalization until the worker has seen
            # every microbatch and DP shard. For the baseline, preserve the
            # existing global-valid-token normalization exactly.
            if self.reduction == "sequence_mean":
                kl_loss = _sequence_balanced_mean(
                    per_token_kl,
                    mask,
                    sample_mask,
                    global_valid_seqs,
                )
            elif tvd_gate_active:
                kl_loss = torch.sum(per_token_kl * mask)
            else:
                kl_loss = masked_mean(
                    per_token_kl,
                    mask,
                    global_normalization_factor=global_valid_toks,
                )
            distillation_kl_loss = kl_loss

            if self.reference_policy_kl_penalty > 0.0:
                assert current_token_logprobs is not None
                reference_logprobs = data["reference_policy_logprobs"].to(
                    device=current_token_logprobs.device,
                    dtype=current_token_logprobs.dtype,
                )[:, 1:]
                reference_max_len = min(
                    current_token_logprobs.shape[1],
                    reference_logprobs.shape[1],
                    base_mask.shape[1],
                )
                reference_per_token_kl = calculate_kl(
                    logprobs=current_token_logprobs[:, :reference_max_len],
                    logprobs_reference=reference_logprobs[:, :reference_max_len],
                    kl_type=self.reference_policy_kl_type,
                    input_clamp_value=self.reference_policy_kl_input_clamp,
                    output_clamp_value=self.reference_policy_kl_output_clamp,
                )
                reference_mask = base_mask[:, :reference_max_len]
                if self.reduction == "sequence_mean":
                    reference_policy_kl_loss = _sequence_balanced_mean(
                        reference_per_token_kl,
                        reference_mask,
                        sample_mask,
                        global_valid_seqs,
                    )
                else:
                    reference_policy_kl_loss = masked_mean(
                        reference_per_token_kl,
                        reference_mask,
                        global_normalization_factor=global_valid_toks,
                    )
                kl_loss = (
                    kl_loss
                    + self.reference_policy_kl_penalty * reference_policy_kl_loss
                )

            # EMA-of-student anchor: KL(p_S || p_EMA) on the EMA's own top-k
            # support. Frame-of-reference matches the main reverse-KL loss —
            # student is penalized for placing mass where the trailing EMA
            # does not. Mask is the same base_mask; no gating.
            if (
                self.ema_anchor_enabled
                and self.ema_anchor_kl_weight > 0.0
                and "ema_topk_logits" in data
                and "ema_topk_indices" in data
            ):
                ema_topk_logits_full = data["ema_topk_logits"].to(
                    device=student_topk_logprobs.device, dtype=torch.float32
                )
                ema_topk_indices_full = data["ema_topk_indices"].to(
                    device=student_topk_logprobs.device
                )
                if (parallel_group is not None) or (cp_size > 1):
                    student_ema_topk_logits = gather_logits_at_global_indices(
                        logits_tensor,
                        ema_topk_indices_full,
                        tp_group=parallel_group,
                        cp_group=cp_group,
                        vocab_start_index=(
                            vocab_start_index if parallel_group is not None else 0
                        ),
                        vocab_end_index=(
                            vocab_end_index
                            if parallel_group is not None
                            else int(logits_tensor.shape[-1])
                        ),
                    )
                else:
                    student_ema_topk_logits = logits_tensor.gather(
                        dim=-1,
                        index=ema_topk_indices_full.to(logits_tensor.device),
                    )
                student_ema_topk_logprobs = torch.nn.functional.log_softmax(
                    student_ema_topk_logits, dim=-1
                )[:, :-1, :]
                ema_topk_logprobs = torch.nn.functional.log_softmax(
                    ema_topk_logits_full, dim=-1
                )[:, :-1, :]

                # reverse-KL against EMA: q_S * (log q_S - log q_EMA)
                student_probs_ema = student_ema_topk_logprobs.exp()
                per_token_ema_kl = (
                    student_probs_ema
                    * (student_ema_topk_logprobs - ema_topk_logprobs)
                ).sum(dim=-1)

                ema_max_len = min(per_token_ema_kl.shape[1], base_mask.shape[1])
                ema_mask = base_mask[:, :ema_max_len]
                ema_per_token = per_token_ema_kl[:, :ema_max_len]
                if self.reduction == "sequence_mean":
                    ema_kl_loss = _sequence_balanced_mean(
                        ema_per_token,
                        ema_mask,
                        sample_mask,
                        global_valid_seqs,
                    )
                else:
                    ema_kl_loss = masked_mean(
                        ema_per_token,
                        ema_mask,
                        global_normalization_factor=global_valid_toks,
                    )
                kl_loss = kl_loss + self.ema_anchor_kl_weight * ema_kl_loss
        else:
            kl_loss = per_token_kl.mean()
            distillation_kl_loss = kl_loss

        assert distillation_kl_loss is not None
        metrics = {
            "loss": float(kl_loss.item()) if kl_loss.ndim == 0 else kl_loss,
            "distillation_loss": float(distillation_kl_loss.detach().item()),
            "num_valid_samples": int(batch_size),
        }
        if reference_policy_kl_loss is not None:
            reference_kl_value = float(reference_policy_kl_loss.detach().item())
            metrics["reference_policy_kl"] = reference_kl_value
            metrics["reference_policy_kl_penalty"] = (
                self.reference_policy_kl_penalty * reference_kl_value
            )
        if self.ema_anchor_enabled and ema_kl_loss is not None:
            ema_kl_value = float(ema_kl_loss.detach().item())
            metrics["ema_anchor_kl"] = ema_kl_value
            metrics["ema_anchor_kl_penalty"] = (
                self.ema_anchor_kl_weight * ema_kl_value
            )

        if self.stop_content_enabled and content_kl_component is not None:
            assert stop_kl_component is not None
            assert student_stop_probability is not None
            assert teacher_stop_probability is not None
            component_mask = base_mask if "token_mask" in data else None
            component_tensors = {
                "stop_content_content_kl": content_kl_component,
                "stop_content_stop_kl": stop_kl_component,
                "stop_content_student_eos_probability": student_stop_probability,
                "stop_content_teacher_eos_probability": teacher_stop_probability,
            }
            for metric_name, metric_tensor in component_tensors.items():
                if component_mask is None:
                    metric_value = metric_tensor.mean()
                else:
                    metric_value = masked_mean(
                        metric_tensor,
                        component_mask,
                        global_normalization_factor=global_valid_toks,
                    )
                metrics[metric_name] = float(metric_value.detach().item())

        # Gate diagnostics: only emitted when the gate ran and per-token mask
        # was available. Values are RAW SUMS. The outer worker will pre-divide
        # every loss-fn metric by num_global_batches, and the outer training
        # loop then applies np.sum across microbatches — that pipeline recovers
        # true totals for sum-shaped metrics, so downstream can compute
        #   kept_frac = kept_sum / base_sum
        #   tvd_mean  = tvd_sum  / base_sum
        # exactly. NOTE: raw absolute counts are NOT reported (they'd be
        # true_count / num_global_batches after the pipeline; only the ratio
        # is invariant). The threshold τ is logged separately by the training
        # loop, NOT here, to bypass the same pre-divide.
        if tvd_gate_active and base_valid_sum is not None:
            metrics["tvd_gate_base_tokens_sum"] = float(base_valid_sum.item())
            metrics["tvd_gate_kept_tokens_sum"] = float(kept_valid_sum.item())
            metrics["tvd_topk_sum"] = float(tvd_sum.item())
            if teacher_margin_sum is not None:
                assert confident_disagreement_sum is not None
                assert selected_confident_disagreement_sum is not None
                metrics["tvd_teacher_margin_sum"] = float(teacher_margin_sum.item())
                metrics["tvd_confident_disagreement_sum"] = float(
                    confident_disagreement_sum.item()
                )
                metrics["tvd_selected_confident_disagreement_sum"] = float(
                    selected_confident_disagreement_sum.item()
                )

        if self.teacher_margin_weight_enabled and confidence_base_sum is not None:
            assert confidence_margin_sum is not None
            assert confidence_weight_sum is not None
            assert confidence_weight_sq_sum is not None
            metrics["teacher_margin_base_tokens_sum"] = float(
                confidence_base_sum.item()
            )
            metrics["teacher_margin_sum"] = float(confidence_margin_sum.item())
            metrics["teacher_margin_weight_sum"] = float(confidence_weight_sum.item())
            metrics["teacher_margin_weight_sq_sum"] = float(
                confidence_weight_sq_sum.item()
            )

        return kl_loss, metrics


# ===============================================================================
# Overlap-Aligned Distillation (OAD)
# ===============================================================================
class OADLossConfig(TypedDict):
    """Configuration for Overlap-Aligned Distillation loss.

    Loss = -E_t[ log( sum_y min(p_S(y|y_<t), p_T(y|y_<t)) ) ]

    See BASIC_OAD_PROPOSAL.md for full design notes.
    """

    eps: NotRequired[float]


class OADLossDataDict(TypedDict):
    """Required keys for the OAD loss function (Path B).

    `teacher_logsumexp` is the per-position full-vocab logsumexp of the teacher,
    populated by the teacher worker's `get_topk_logits`. Without it, OAD cannot
    construct exact teacher probabilities and will raise a clear error.
    """

    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    teacher_topk_logits: torch.Tensor  # [B, S, k]
    teacher_topk_indices: torch.Tensor  # [B, S, k]
    teacher_logsumexp: torch.Tensor  # [B, S], exact full-vocab logsumexp


class OADLossFn(LossFunction):
    """Basic Overlap-Aligned Distillation loss (Path B: exact teacher logsumexp).

    Per-token loss = -log(sum_y min(p_S(y), p_T(y))) on the teacher's top-k support.
    Each token is weighted equally (no length weighting, no critical-token weighting).

    Implementation notes:
        - Student logsumexp is computed exactly from full-vocab logits via
          vocab_cp_logsumexp (TP+CP aware).
        - Teacher logsumexp is the exact full-vocab value, supplied by the
          teacher worker via `data["teacher_logsumexp"]` (Path B). With this in
          place, identity (student==teacher) yields loss = 0 by construction.
        - The truncated acceptance is a lower bound on the true acceptance,
          with bias <= 1 - M_T (Theorem in §3.2). Monitored via
          `teacher_topk_mass` (now meaningful under Path B).
    """

    def __init__(self, cfg: OADLossConfig):
        self.eps = cfg.get("eps", 1e-8)
        self.loss_type = LossType.TOKEN_LEVEL

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: OADLossDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute OAD loss between teacher (top-k) and student logits."""
        # 1) Teacher data — Path B requires exact teacher_logsumexp from the worker
        teacher_topk_logits = data["teacher_topk_logits"].to(torch.float32)  # [B, S, k]
        teacher_topk_indices = data["teacher_topk_indices"]  # [B, S, k]
        if "teacher_logsumexp" not in data:
            raise KeyError(
                "OADLossFn (Path B) requires `teacher_logsumexp` in train_data. "
                "Ensure the teacher worker's get_topk_logits returns the "
                "full-vocab logsumexp (currently supported on dtensor backends; "
                "Megatron backend returns top-k only and is not yet supported)."
            )
        teacher_lse = data["teacher_logsumexp"].to(torch.float32)  # [B, S]
        batch_size = teacher_topk_indices.shape[0]
        full_seq_len = teacher_topk_indices.shape[1]

        if teacher_topk_indices.shape[-1] <= 0:
            raise ValueError(
                f"topk must be positive, got {teacher_topk_indices.shape[-1]}."
            )

        # 2) Resolve TP / CP (mirror DistillationLossFn:1029-1062).
        # TODO(future): if a third distill loss is added, refactor TP/CP/DTensor
        #   resolution into a shared helper to avoid 3-way duplication.
        cp_group = context_parallel_group
        cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
        next_token_logits = next_token_logits.to(torch.float32)

        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            V_local = int(next_token_logits.shape[-1])
            vocab_start_index = vocab_parallel_rank * V_local
            vocab_end_index = (vocab_parallel_rank + 1) * V_local
            parallel_group = vocab_parallel_group
            logits_local = next_token_logits
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            device_mesh = next_token_logits.device_mesh
            tp_group = device_mesh.get_group("tp")
            tp_rank = tp_group.rank()
            local_logits = next_token_logits.to_local()
            V_local = int(local_logits.shape[-1])
            vocab_start_index = tp_rank * V_local
            vocab_end_index = (tp_rank + 1) * V_local
            parallel_group = tp_group
            logits_local = local_logits
            teacher_topk_indices = teacher_topk_indices.to(local_logits.device)
            if (
                device_mesh.mesh_dim_names is not None
                and "cp" in device_mesh.mesh_dim_names
            ):
                cp_group = device_mesh.get_group("cp")
                cp_size = cp_group.size()
            else:
                cp_group = None
                cp_size = 1
        else:
            parallel_group = None
            logits_local = next_token_logits
            vocab_start_index = 0
            vocab_end_index = int(logits_local.shape[-1])

        # 3) Student logsumexp over full vocab (TP+CP aware) -> [B, S_full]
        student_lse = vocab_cp_logsumexp(
            logits_local,
            tp_group=parallel_group,
            cp_group=cp_group,
            full_seq_len=full_seq_len,
        )

        # 4) (Path B) Teacher logsumexp is provided exactly by the worker;
        #    we already loaded it as `teacher_lse` in step 1.

        # 5) Student logits at teacher's top-k indices (TP+CP aware) -> [B, S, k]
        student_topk_logits = gather_logits_at_global_indices(
            logits_local,
            teacher_topk_indices,
            tp_group=parallel_group,
            cp_group=cp_group,
            vocab_start_index=vocab_start_index,
            vocab_end_index=vocab_end_index,
        )

        # 6) logits -> global probabilities (true softmax, NOT top-k renormalized)
        student_log_p_topk = student_topk_logits - student_lse.unsqueeze(-1)
        teacher_log_p_topk = teacher_topk_logits - teacher_lse.unsqueeze(-1)

        student_p_topk = student_log_p_topk.exp()
        teacher_p_topk = teacher_log_p_topk.exp()

        # 7) Acceptance = sum_y min(p_S, p_T) on the top-k support (lower bound; bias <= 1 - M_T)
        acceptance = torch.minimum(student_p_topk, teacher_p_topk).sum(dim=-1)  # [B, S]

        # 8) Next-token alignment: predict t+1 from position t
        # Use token_mask aligned with predicted tokens (mirror DistillationLossFn KL: token_mask[:, 1:])
        per_token_loss = -acceptance[:, :-1].clamp_min(self.eps).log()  # [B, S-1]

        token_mask = data["token_mask"][:, 1:]  # [B, S-1]
        sample_mask = data["sample_mask"]
        max_len = per_token_loss.shape[1]
        token_mask = token_mask[:, :max_len]
        mask = token_mask * sample_mask.unsqueeze(-1)

        loss = masked_mean(
            per_token_loss,
            mask,
            global_normalization_factor=global_valid_toks,
        )

        # 9) Monitoring (per v2 §7.2; Path B semantics).
        # IMPORTANT: we return *un-averaged* sums plus the token count, not
        # already-divided means. The full-step aggregation pipeline does
        #   worker: loss_metrics[k] /= num_global_batches
        #   distillation.py: metrics[k] = np.sum(across all microbatches)
        # which is the textbook "pre-divided sum" trick — correct iff each
        # microbatch's metric is already a sum-style quantity, not a mean.
        # By returning sums + count, the final printed value
        #   true_mean = sum_value / sum_count
        # is invariant to mb / dp / global-batch sharding (no need to guess
        # num_microbatches or num_global_batches).
        with torch.no_grad():
            valid = mask.bool()
            n_valid_tensor = mask.sum().clamp_min(1.0)

            # Sums over valid tokens (the count is the same denominator for
            # every per-token metric below).
            acceptance_sum = (acceptance[:, :-1] * mask).sum()
            teacher_topk_mass_sum = (teacher_p_topk[:, :-1].sum(dim=-1) * mask).sum()
            student_mass_on_teacher_topk_sum = (
                student_p_topk[:, :-1].sum(dim=-1) * mask
            ).sum()

            grad_active_per_token = (
                student_p_topk[:, :-1] < teacher_p_topk[:, :-1]
            )  # [B, S-1, k] bool
            active_grad_position_sum = (
                grad_active_per_token.any(dim=-1).to(mask.dtype) * mask
            ).sum()
            active_grad_token_sum = (
                grad_active_per_token.to(mask.dtype).mean(dim=-1) * mask
            ).sum()

            min_accept = (
                acceptance[:, :-1][valid].min()
                if valid.any()
                else torch.tensor(0.0, device=acceptance.device)
            )

        # Return per-mb sums + a token count. distillation.py prints
        #   value = oad_*_sum / oad_token_count
        # which is invariant to any mb-aggregation tricks.
        metrics = {
            "loss": float(loss.item()) if loss.ndim == 0 else loss,
            "num_valid_samples": int(batch_size),
            "oad_acceptance_sum": float(acceptance_sum.item()),
            "oad_teacher_topk_mass_sum": float(teacher_topk_mass_sum.item()),
            "oad_student_mass_on_teacher_topk_sum": float(
                student_mass_on_teacher_topk_sum.item()
            ),
            "oad_active_grad_position_sum": float(active_grad_position_sum.item()),
            "oad_active_grad_token_sum": float(active_grad_token_sum.item()),
            "oad_token_count": float(n_valid_tensor.item()),
            "oad_min_accept_pathB": float(min_accept.item()),
        }

        return loss, metrics


# ===============================================================================
# Top-K Agreement Policy Gradient (teacher-guided sharpening)
# ===============================================================================
class TopKAgreementPGLossConfig(TypedDict):
    """Configuration for the top-k agreement policy-gradient loss.

    Loss per sampled token v_t:
        A_t = +1 if v_t ∈ S_top ∩ T_top
        A_t =  0 if v_t ∈ S_top \\ T_top
        A_t = −1 if v_t ∉ S_top
        w_t = π_θ(v_t) / π_ref(v_t)      # ref = frozen initial student ckpt
        L_t = − min(w_t · A_t, clip(w_t, 1−ε_lo, 1+ε_hi) · A_t)
    """

    student_k: int
    ratio_clip_min: float
    ratio_clip_max: float
    reduction: NotRequired[str]  # "token_mean" (default) | "sequence_mean"


class TopKAgreementPGLossDataDict(TypedDict):
    """Required keys for the top-k agreement PG loss."""

    input_ids: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    teacher_topk_indices: torch.Tensor  # [B, S, k_T]
    # Per-position log π_ref(x_{t+1}). Position 0 follows the policy API
    # convention and is always 0. Reference weights = frozen initial student.
    reference_policy_logprobs: torch.Tensor  # [B, S]
    seq_index: NotRequired[torch.Tensor]


class TopKAgreementPGLossFn(LossFunction):
    """Three-class advantage × PPO-clipped IS surrogate on student rollouts.

    Semantics (OPD-style, single rollout per step):
        - π_θ    = student being trained (this forward)
        - π_ref  = frozen initial student ckpt (reference channel)
        - π_T    = teacher (top-k indices supplied by teacher worker)

    For each sampled token v_t = input_ids[t+1]:
        S_top(x_<t) = top-k global vocabulary of π_θ at position t
        T_top(x_<t) = teacher's top-k (already in teacher_topk_indices)

        A_t = +1  if v_t ∈ S_top ∩ T_top
        A_t =  0  if v_t ∈ S_top \\ T_top   (student thinks fine, teacher does not)
        A_t = −1  if v_t ∉ S_top             (student didn't rank v_t highly at all)

        w_t = exp(log π_θ(v_t) − log π_ref(v_t))
        L_t = − min(w_t · A_t, clip(w_t, 1−ε_lo, 1+ε_hi) · A_t)

    IS + clip role: bound single-step drift from the initial ckpt. Because
    π_ref is frozen (not per-step), ratios drift away from 1 as training
    proceeds; the diagnostic ``is_clipped_frac_*`` metrics make this visible.

    Fraction of A=−1 tokens is expected to be small (v_t is drawn from π_θ_old
    ≈ π_θ, and π_θ's own top-k covers most of its mass). This is a known
    property of the formulation, not a bug — the sign of the update on the
    rare −1 positions still points the student away from tokens it now
    dislikes but nevertheless emitted during rollout.
    """

    def __init__(self, cfg: TopKAgreementPGLossConfig):
        self.student_k = int(cfg["student_k"])
        self.ratio_clip_min = float(cfg["ratio_clip_min"])
        self.ratio_clip_max = float(cfg["ratio_clip_max"])
        self.reduction = str(cfg.get("reduction", "token_mean"))
        self.loss_type = LossType.TOKEN_LEVEL

        if self.student_k <= 0:
            raise ValueError(
                f"loss_fn.topk_agreement_pg.student_k must be positive, got {self.student_k}."
            )
        if not (0.0 < self.ratio_clip_min < 1.0):
            raise ValueError(
                "loss_fn.topk_agreement_pg.ratio_clip_min must be in (0, 1)."
            )
        if not (self.ratio_clip_max > 1.0 and math.isfinite(self.ratio_clip_max)):
            raise ValueError(
                "loss_fn.topk_agreement_pg.ratio_clip_max must be a finite value > 1.0."
            )
        if self.reduction not in ("token_mean", "sequence_mean"):
            raise ValueError(
                f"Unknown loss_fn.reduction={self.reduction!r}. "
                "Expected one of: 'token_mean', 'sequence_mean'."
            )

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: TopKAgreementPGLossDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        input_ids = data["input_ids"]
        batch_size = input_ids.shape[0]
        teacher_topk_indices = data["teacher_topk_indices"]  # [B, S, k_T]

        if "reference_policy_logprobs" not in data:
            raise KeyError(
                "TopKAgreementPGLossFn requires `reference_policy_logprobs` "
                "in train_data. Enable the reference logprob path in "
                "distillation.py so π_ref = frozen initial student is snapshotted."
            )

        cp_group = context_parallel_group
        cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
        next_token_logits = next_token_logits.to(torch.float32)

        # Resolve TP / CP layout (mirrors OADLossFn:2280-2320).
        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None
            V_local = int(next_token_logits.shape[-1])
            vocab_start_index = vocab_parallel_rank * V_local
            vocab_end_index = (vocab_parallel_rank + 1) * V_local
            parallel_group = vocab_parallel_group
            logits_local = next_token_logits
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            device_mesh = next_token_logits.device_mesh
            tp_group = device_mesh.get_group("tp")
            tp_rank = tp_group.rank()
            local_logits = next_token_logits.to_local()
            V_local = int(local_logits.shape[-1])
            vocab_start_index = tp_rank * V_local
            vocab_end_index = (tp_rank + 1) * V_local
            parallel_group = tp_group
            logits_local = local_logits
            teacher_topk_indices = teacher_topk_indices.to(local_logits.device)
            if (
                device_mesh.mesh_dim_names is not None
                and "cp" in device_mesh.mesh_dim_names
            ):
                cp_group = device_mesh.get_group("cp")
                cp_size = cp_group.size()
            else:
                cp_group = None
                cp_size = 1
        else:
            parallel_group = None
            logits_local = next_token_logits
            vocab_start_index = 0
            vocab_end_index = int(logits_local.shape[-1])

        # ------------------------------------------------------------------
        # 1) Current-student logprob at the sampled token v_t = input_ids[t+1].
        #    We reuse the existing TP+CP-aware kernel used by ClippedPGLossFn.
        # ------------------------------------------------------------------
        if parallel_group is not None:
            current_logp = from_parallel_logits_to_logprobs(
                next_token_logits,
                input_ids,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_end_index,
                tp_group=parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )  # [B, S-1]
            current_logp = current_logp[:, : input_ids.shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            current_logp = get_logprobs_from_vocab_parallel_logits(
                next_token_logits,
                input_ids,
                seq_index=data.get("seq_index"),
            )
        else:
            shifted_logits = logits_local[:, :-1]
            next_ids = input_ids[:, 1:].to(shifted_logits.device)
            current_logp = (
                torch.nn.functional.log_softmax(shifted_logits, dim=-1)
                .gather(dim=-1, index=next_ids.unsqueeze(-1))
                .squeeze(-1)
            )  # [B, S-1]

        # ------------------------------------------------------------------
        # 2) Student global top-k indices at every position (TP+CP aware).
        #    We only need these indices for membership testing (advantage
        #    classification). No gradient flows through this branch — the
        #    advantage A ∈ {+1, 0, −1} is a piecewise-constant function of
        #    the top-k membership set, hence its gradient is zero a.e.
        # ------------------------------------------------------------------
        with torch.no_grad():
            if parallel_group is not None:
                # distributed_vocab_topk operates on the CP-sharded logits directly
                # and returns per-shard results with seq_len == logits_local.shape[1].
                _, student_topk_idx_sharded = distributed_vocab_topk(
                    logits_local,
                    self.student_k,
                    tp_group=parallel_group,
                    vocab_start_index=vocab_start_index,
                    vocab_end_index=vocab_end_index,
                    chunk_size=max(1, min(int(logits_local.shape[1]), 1024)),
                )  # [B, S_local, k_S], global vocab ids
                if cp_size > 1:
                    student_topk_full = allgather_cp_sharded_tensor(
                        student_topk_idx_sharded, cp_group, seq_dim=1
                    )
                    # After CP allgather, sequence length is padded to
                    # logits_local.shape[1] * cp_size; teacher_topk_indices
                    # is the canonical (unpadded) length. The trim must be a
                    # true prefix — if allgather emitted a shorter tensor for
                    # any reason (mis-sharded input, unexpected CP layout),
                    # a silent trim would produce misaligned membership tests.
                    canonical_S = teacher_topk_indices.shape[1]
                    assert student_topk_full.shape[1] >= canonical_S, (
                        f"CP-gathered student top-k has seq_len "
                        f"{student_topk_full.shape[1]} < canonical S "
                        f"{canonical_S}; refusing to trim (would misalign "
                        f"advantage vs targets)."
                    )
                    student_topk_full = student_topk_full[:, :canonical_S, :]
                else:
                    student_topk_full = student_topk_idx_sharded
            else:
                # Single-process fallback.
                _, student_topk_full = torch.topk(
                    logits_local, k=self.student_k, dim=-1
                )  # [B, S, k_S], local indices == global indices in single-proc

        # Align to next-token prediction: at position t we care about v_{t+1}.
        # Teacher/Student top-k tensors are indexed by position t (context),
        # sampled token is input_ids[t+1]. So slice [:, :-1, :] and
        # input_ids[:, 1:].
        student_topk_at_ctx = student_topk_full[:, :-1, :].to(current_logp.device)
        teacher_topk_at_ctx = teacher_topk_indices[:, :-1, :].to(current_logp.device)
        v_next = input_ids[:, 1:].to(current_logp.device)  # [B, S-1]

        # ------------------------------------------------------------------
        # 3) Advantage classification A ∈ {+1, 0, −1}.
        # ------------------------------------------------------------------
        # in_student[b, t] = True iff v_next[b, t] appears anywhere in the
        # student's top-k at context position t.
        in_student = (
            student_topk_at_ctx == v_next.unsqueeze(-1)
        ).any(dim=-1)  # [B, S-1] bool
        in_teacher = (
            teacher_topk_at_ctx == v_next.unsqueeze(-1)
        ).any(dim=-1)  # [B, S-1] bool

        pos_mask = in_student & in_teacher  # A = +1
        neg_mask = ~in_student  # A = -1
        # zero elsewhere (in_student & ~in_teacher)

        advantage = torch.zeros_like(current_logp)
        advantage = torch.where(
            pos_mask,
            torch.ones_like(advantage),
            torch.where(neg_mask, -torch.ones_like(advantage), advantage),
        )

        # ------------------------------------------------------------------
        # 4) IS ratio + PPO-style clip.
        # ------------------------------------------------------------------
        ref_logp = data["reference_policy_logprobs"].to(
            device=current_logp.device, dtype=current_logp.dtype
        )
        # `reference_policy_logprobs` has shape [B, S]; position 0 is a
        # convention-zero. Align to targets t+1 by slicing [:, 1:].
        max_len = current_logp.shape[1]
        ref_logp = ref_logp[:, 1 : 1 + max_len]

        log_ratio = current_logp - ref_logp
        ratio = log_ratio.exp()
        ratio_clipped = ratio.clamp(self.ratio_clip_min, self.ratio_clip_max)

        # PPO clipped surrogate: minimum over unclipped/clipped ratio times A.
        surrogate_unclipped = ratio * advantage
        surrogate_clipped = ratio_clipped * advantage
        per_token_surrogate = torch.minimum(surrogate_unclipped, surrogate_clipped)
        per_token_loss = -per_token_surrogate  # minimize -surrogate = maximize surrogate

        # ------------------------------------------------------------------
        # 5) Masking and reduction (matches DistillationLossFn conventions).
        # ------------------------------------------------------------------
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        token_mask = token_mask[:, :max_len]
        mask = token_mask * sample_mask.unsqueeze(-1)  # [B, S-1]

        if self.reduction == "sequence_mean":
            loss = _sequence_balanced_mean(
                per_token_loss,
                mask,
                sample_mask,
                global_valid_seqs,
            )
        else:
            loss = masked_mean(
                per_token_loss,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        # ------------------------------------------------------------------
        # 6) Diagnostics. Return raw sums + a token count so the outer
        #    "pre-divide by num_global_batches → np.sum over microbatches"
        #    pipeline reconstructs the correct ratios (mirrors OADLossFn).
        # ------------------------------------------------------------------
        with torch.no_grad():
            valid = mask
            n_valid = valid.sum().clamp_min(1.0)

            pos_sum = (pos_mask.to(mask.dtype) * mask).sum()
            zero_sum = (
                (in_student & ~in_teacher).to(mask.dtype) * mask
            ).sum()
            neg_sum = (neg_mask.to(mask.dtype) * mask).sum()

            # IS ratio statistics on valid tokens (unclipped).
            ratio_valid = ratio * mask
            ratio_sum = ratio_valid.sum()
            ratio_sq_sum = (ratio * ratio * mask).sum()

            clipped_hi = (ratio > self.ratio_clip_max).to(mask.dtype) * mask
            clipped_lo = (ratio < self.ratio_clip_min).to(mask.dtype) * mask
            clipped_hi_sum = clipped_hi.sum()
            clipped_lo_sum = clipped_lo.sum()

            # Per-class loss contributions.
            loss_from_pos_sum = (per_token_loss * pos_mask.to(mask.dtype) * mask).sum()
            loss_from_neg_sum = (per_token_loss * neg_mask.to(mask.dtype) * mask).sum()
            # (A=0 contributes zero surrogate, so loss_from_zero_sum = 0.)

            # Effective learning signal: PPO clipped surrogate zeros out gradient
            # when the ratio saturates on the "wrong" side (see Schulman 2017 §6.1).
            #   A = +1: gradient flows only when ratio ≤ clip_max
            #           (above that, min() picks the constant clipped branch)
            #   A = −1: gradient flows only when ratio ≥ clip_min
            # As training proceeds and π_θ drifts from the frozen π_ref, these
            # counts fall toward 0 → learning signal dies. Track them explicitly.
            effective_pos = pos_mask & (ratio <= self.ratio_clip_max)
            effective_neg = neg_mask & (ratio >= self.ratio_clip_min)
            effective_pos_sum = (effective_pos.to(mask.dtype) * mask).sum()
            effective_neg_sum = (effective_neg.to(mask.dtype) * mask).sum()

        metrics = {
            "loss": float(loss.item()) if loss.ndim == 0 else loss,
            "num_valid_samples": int(batch_size),
            "topk_pg_token_count": float(n_valid.item()),
            "topk_pg_pos_sum": float(pos_sum.item()),
            "topk_pg_zero_sum": float(zero_sum.item()),
            "topk_pg_neg_sum": float(neg_sum.item()),
            "topk_pg_is_ratio_sum": float(ratio_sum.item()),
            "topk_pg_is_ratio_sq_sum": float(ratio_sq_sum.item()),
            "topk_pg_is_clipped_hi_sum": float(clipped_hi_sum.item()),
            "topk_pg_is_clipped_lo_sum": float(clipped_lo_sum.item()),
            "topk_pg_effective_pos_sum": float(effective_pos_sum.item()),
            "topk_pg_effective_neg_sum": float(effective_neg_sum.item()),
            "topk_pg_loss_from_pos_sum": float(loss_from_pos_sum.item()),
            "topk_pg_loss_from_neg_sum": float(loss_from_neg_sum.item()),
        }

        return loss, metrics
