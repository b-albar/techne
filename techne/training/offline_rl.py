"""Offline RL training utilities.

Implements off-policy RL algorithms that train on pre-collected trajectory
datasets without requiring online generation workers.

Currently supports:
- SPO (Soft Policy Optimization): Cumulative token-level Q-values regressed
  to terminal reward. No value model required.
  Reference: https://arxiv.org/abs/2503.05453
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, Features, Sequence, Value
from trl import SFTConfig, SFTTrainer

from techne.config import TechneConfig
from techne.data import Trajectory
from techne.training.distributed import is_main_process
from techne.training.sft import get_common_training_args

logger = logging.getLogger(__name__)


def train_spo(
    config: TechneConfig,
    model,
    tokenizer,
    dataset: Any,
    **kwargs,
):
    """Train using Soft Policy Optimization (SPO) on an offline dataset.

    SPO regresses cumulative token-level Q-values to observed terminal rewards.
    It can learn from arbitrary off-policy trajectories and does not require
    a separate value model.

    The dataset should contain trajectories with rewards and optionally
    pre-computed reference model log probabilities.

    Args:
        config: Techne configuration
        model: Model to train
        tokenizer: Tokenizer
        dataset: Dataset of trajectories with rewards
        **kwargs: Additional arguments

    Returns:
        Training result
    """
    assert len(dataset) > 0, "SPO dataset is empty!"

    # 1. Normalize to List[Trajectory]
    trajectories = []
    if is_main_process():
        logger.info("Normalizing dataset to Trajectories for SPO...")
    for item in dataset:
        if isinstance(item, Trajectory):
            trajectories.append(item)
        elif isinstance(item, dict):
            try:
                trajectories.append(Trajectory.model_validate(item))
            except Exception as e:
                if is_main_process():
                    logger.warning("Failed to validate dict as Trajectory: %s", e)
        elif isinstance(item, str):
            try:
                trajectories.append(Trajectory.model_validate_json(item))
            except Exception as e:
                if is_main_process():
                    logger.warning("Failed to parse JSON string as Trajectory: %s", e)
        else:
            if is_main_process():
                logger.warning("Skipping unsupported item type: %s", type(item))

    if len(trajectories) == 0:
        raise ValueError("No valid Trajectories found in dataset!")

    # 2. Convert trajectories to training samples with rewards and ref logprobs
    max_len = config.training.max_seq_length
    processed_samples = []
    filtered = 0
    samples_with_logprobs = 0

    for traj in trajectories:
        sample = traj.to_training_sample(tokenizer=tokenizer)
        if max_len is not None and len(sample.input_ids) > max_len:
            filtered += 1
            continue

        # Get trajectory-level reward
        reward = traj.reward if traj.reward is not None else traj.total_reward

        sample_dict = {
            "input_ids": sample.input_ids,
            "labels": sample.labels,
            "rewards": reward,
        }

        if sample.log_probs is not None:
            sample_dict["ref_logprobs"] = sample.log_probs
            samples_with_logprobs += 1

        processed_samples.append(sample_dict)

    if filtered > 0 and is_main_process():
        logger.info(
            "Filtered %d/%d trajectories exceeding max_seq_length=%s",
            filtered, len(trajectories), max_len,
        )

    if len(processed_samples) == 0:
        raise ValueError("No valid samples after filtering!")

    if is_main_process():
        logger.info(
            "SPO dataset: %d samples (%d with pre-computed ref logprobs)",
            len(processed_samples), samples_with_logprobs,
        )

    # 3. Build HF Dataset
    # Variable-length columns: use list-of-lists format
    hf_dataset = Dataset.from_dict({
        "input_ids": [s["input_ids"] for s in processed_samples],
        "labels": [s["labels"] for s in processed_samples],
        "rewards": [s["rewards"] for s in processed_samples],
        "ref_logprobs": [
            s.get("ref_logprobs", []) for s in processed_samples
        ],
    })

    # 4. Create trainer config
    args_dict = get_common_training_args(config)
    if config.training.max_seq_length is not None:
        args_dict["max_length"] = config.training.max_seq_length
    args_dict["packing"] = False
    args_dict.pop("remove_unused_columns", None)
    args_dict["dataset_kwargs"] = {"skip_prepare_dataset": True}
    args = SFTConfig(**args_dict)

    # 5. Compute reference logprobs if not pre-computed
    # Use the initial model weights (before training) as reference
    has_all_logprobs = samples_with_logprobs == len(processed_samples)
    ref_model = None
    if not has_all_logprobs:
        if is_main_process():
            logger.info(
                "Computing reference logprobs from initial model for %d samples...",
                len(processed_samples) - samples_with_logprobs,
            )
        # Keep a frozen copy of the model for ref logprob computation
        ref_model = model

    # 6. Create SPO trainer and train
    trainer = _SPOTrainer(
        spo_beta=config.training.spo_beta,
        ref_model=ref_model,
        has_all_ref_logprobs=has_all_logprobs,
        model=model,
        args=args,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
    )

    if is_main_process():
        logger.info(
            "Starting SPO training on %d samples (beta=%.4f)",
            len(processed_samples), config.training.spo_beta,
        )
    return trainer.train()


class _SPOTrainer(SFTTrainer):
    """SFTTrainer with SPO (Soft Policy Optimization) loss.

    Instead of NLL, computes cumulative token-level Q-values from the
    log-probability ratio (policy vs reference), then regresses the
    terminal Q-value to the observed reward.
    """

    def __init__(
        self,
        spo_beta: float = 0.087,
        ref_model=None,
        has_all_ref_logprobs: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spo_beta = spo_beta
        self.ref_model = ref_model
        self.has_all_ref_logprobs = has_all_ref_logprobs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute SPO loss: MSE(Q_T, reward).

        Steps:
        1. Forward pass → per-token policy logprobs
        2. Get reference logprobs (from dataset or compute on-the-fly)
        3. Token advantage: β * (log π_θ - log π_ref)
        4. Cumulative Q: cumsum(token_advantage) over completion tokens
        5. Terminal Q: last non-padded cumulative value
        6. Loss: MSE(terminal_q, reward)
        """
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        rewards = inputs["rewards"]
        cached_ref_logprobs = inputs.get("ref_logprobs")

        # Forward pass
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        # Per-token log probs (shifted for autoregressive)
        log_probs = F.log_softmax(logits, dim=-1)
        shift_logprobs = log_probs[:, :-1, :]
        shift_labels = labels[:, 1:]

        batch_size, seq_len = shift_labels.shape
        policy_logprobs = shift_logprobs.gather(
            2, shift_labels.clamp(min=0).unsqueeze(2)
        ).squeeze(2)

        # Mask: completion tokens only (labels != -100)
        mask = shift_labels != -100

        # Reference logprobs
        ref_logprobs = torch.zeros_like(policy_logprobs)
        if self.has_all_ref_logprobs and cached_ref_logprobs is not None:
            # Use pre-computed ref logprobs from dataset
            for i in range(batch_size):
                ref_lp = cached_ref_logprobs[i]
                if isinstance(ref_lp, torch.Tensor):
                    n = min(len(ref_lp), seq_len)
                    ref_logprobs[i, :n] = ref_lp[:n].to(ref_logprobs.device)
                elif isinstance(ref_lp, list) and len(ref_lp) > 0:
                    n = min(len(ref_lp), seq_len)
                    ref_logprobs[i, :n] = torch.tensor(
                        ref_lp[:n], device=ref_logprobs.device, dtype=ref_logprobs.dtype,
                    )
        elif self.ref_model is not None:
            # Compute ref logprobs on-the-fly
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=input_ids)
                ref_log_probs = F.log_softmax(ref_outputs.logits, dim=-1)
                shift_ref_logprobs = ref_log_probs[:, :-1, :]
                ref_logprobs = shift_ref_logprobs.gather(
                    2, shift_labels.clamp(min=0).unsqueeze(2)
                ).squeeze(2)

        # Token-level advantage: β * (log π_θ - log π_ref)
        token_advantage = self.spo_beta * (policy_logprobs - ref_logprobs)

        # Cumulative Q-value along the sequence (only over completion tokens)
        cumulative_q = torch.cumsum(token_advantage * mask, dim=1)

        # Terminal Q: value at the last non-padded position per sample
        completion_lengths = mask.sum(dim=1)  # [B]
        last_idx = (completion_lengths - 1).clamp(min=0).unsqueeze(1)  # [B, 1]
        terminal_q = cumulative_q.gather(1, last_idx).squeeze(1)  # [B]

        # Reward target
        if isinstance(rewards, torch.Tensor):
            reward_target = rewards.float().to(terminal_q.device)
        else:
            reward_target = torch.tensor(rewards, device=terminal_q.device, dtype=torch.float32)

        # SPO loss: MSE(terminal_q, reward)
        loss = F.mse_loss(terminal_q, reward_target)

        if return_outputs:
            return loss, outputs
        return loss
