"""Distillation training utilities.

This module provides:
- Offline distillation: train on teacher completions with optional KL loss
- Tokenizer alignment for cross-tokenizer distillation (GOLD approach)
- KL reward computation for on-policy distillation
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer

from techne.config import DistillationMode, TechneConfig
from techne.data import Trajectory
from techne.training.alignment import DirectAligner, GoldAligner
from techne.training.estimators import (
    ForwardKLEstimator,
    SlimEstimator,
    SparseLogits,
)
from techne.training.model import create_teacher_model
from techne.training.sft import get_common_training_args

logger = logging.getLogger(__name__)

# =============================================================================
# Tokenizer Alignment Utilities
# =============================================================================


def are_tokenizers_identical(
    student_tokenizer: PreTrainedTokenizer,
    teacher_tokenizer: PreTrainedTokenizer,
) -> bool:
    """Check if two tokenizers are effectively identical."""
    if student_tokenizer.vocab_size != teacher_tokenizer.vocab_size:
        return False

    # Heuristic: Encode a test string and check if IDs are identical
    test_str = "Test system prompt: 1+1=2"
    s_enc = student_tokenizer(test_str, add_special_tokens=False).input_ids
    t_enc = teacher_tokenizer(test_str, add_special_tokens=False).input_ids
    return s_enc == t_enc


# =============================================================================
# Offline Distillation Training
# =============================================================================


def train_distill_offline(
    config: TechneConfig,
    model,
    tokenizer,
    dataset: Any,
    **kwargs,
):
    """Standard offline distillation: train student on teacher completions.

    Supports:
    - SFT-style training on teacher outputs (when no teacher_model specified)
    - KL divergence loss from teacher logits (when teacher_model specified)
    - Different tokenizers (GOLD approach: decode and re-tokenize)

    Args:
        config: Techne configuration
        model: Student model to train
        tokenizer: Student tokenizer
        dataset: Dataset with teacher-generated completions
        **kwargs: Additional arguments

    Returns:
        Training result
    """
    teacher_model_path = config.training.teacher_model

    # Data Validation
    assert len(dataset) > 0, "Distillation dataset is empty!"

    # 1. Normalize to List[Trajectory]
    # We strictly expect serialized trajectories (dicts, json strings) or Trajectory objects
    trajectories = []
    logger.info("Normalizing dataset to Trajectories...")
    for item in dataset:
        if isinstance(item, Trajectory):
            trajectories.append(item)
        elif isinstance(item, dict):
            # Expect serialized Trajectory
            try:
                trajectories.append(Trajectory.model_validate(item))
            except Exception as e:
                logger.warning("Failed to validate dict as Trajectory: %s", e)
        elif isinstance(item, str):
            # Expect JSON string
            try:
                trajectories.append(Trajectory.model_validate_json(item))
            except Exception as e:
                logger.warning("Failed to parse JSON string as Trajectory: %s", e)
        else:
            logger.warning("Skipping unsupported item type: %s", type(item))

    if len(trajectories) == 0:
        raise ValueError("No valid Trajectories found in dataset!")


    # 2. Prepare for Distillation (KL Loss)
    logger.info("Preparing distillation dataset from Trajectories...")
    processed_samples = []
    samples_with_logprobs = 0

    for traj in trajectories:
        sample = traj.to_training_sample(tokenizer=tokenizer)
        sample_dict = {"input_ids": sample.input_ids, "labels": sample.labels}

        if sample.log_probs is not None:
            sample_dict["teacher_logprobs"] = sample.log_probs
            samples_with_logprobs += 1

        processed_samples.append(sample_dict)

    dataset = processed_samples
    all_have_logprobs = samples_with_logprobs == len(processed_samples)

    # Load teacher model only if needed (not all samples have cached logprobs)
    teacher_model = None
    teacher_tokenizer = None
    same_tokenizer = True

    if all_have_logprobs:
        logger.info("All %d samples have cached teacher logprobs - skipping teacher model load", len(dataset))
    else:
        if not teacher_model_path:
            raise ValueError(
                "Distillation requires either cached logprobs in trajectories "
                "or a teacher model path in config."
            )
        logger.info("Loading teacher model for logit distillation: %s", teacher_model_path)
        teacher_model = create_teacher_model(
            teacher_model_path,
            device="auto",
            dtype=config.model.dtype,
        )
        teacher_tokenizer = teacher_model.tokenizer
        same_tokenizer = are_tokenizers_identical(tokenizer, teacher_tokenizer)

        if same_tokenizer:
            logger.info("Tokenizers are identical - using direct logit alignment")
        else:
            logger.info("Different tokenizers detected - using GOLD approach (decode & re-tokenize)")

    # Create trainer config
    args_dict = get_common_training_args(config)
    args_dict["max_length"] = config.training.max_seq_length
    args_dict["packing"] = False
    args_dict.pop("remove_unused_columns", None)
    args = SFTConfig(**args_dict)

    # Create distillation trainer with custom loss
    trainer = _DistillationTrainer(
        teacher=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=tokenizer,
        same_tokenizer=same_tokenizer,
        kl_weight=config.training.kl_weight,
        distillation_mode=config.training.distillation_mode,
        slim_top_k=config.training.slim_top_k,
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting offline distillation with KL loss on %d samples", len(dataset))
    return trainer.train()


def compute_distillation_reward(
    completion_text: str,
    prompt_text: str,
    teacher_model: PreTrainedModel,
    teacher_tokenizer: PreTrainedTokenizer,
    student_tokenizer: PreTrainedTokenizer | None = None,
    max_length: int = 2048,
    use_kl: bool = False,
    student_logits: torch.Tensor | None = None,
) -> float:
    """Compute distillation reward.

    If use_kl is False (default): Uses Teacher Log-Likelihood of the completion.
    If use_kl is True: Uses -KL(Student || Teacher) (requires student_logits and alignment).
    """
    full_text = prompt_text + completion_text

    # 1. Teacher Forward
    teacher_inputs = teacher_tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(teacher_model.device)

    with torch.no_grad():
        teacher_outputs = teacher_model(**teacher_inputs)

    # 2. Log-Likelihood Reward (Simple)
    if not use_kl or student_logits is None or student_tokenizer is None:
        teacher_logprobs = F.log_softmax(teacher_outputs.logits, dim=-1)
        input_ids = teacher_inputs.input_ids[0]
        # Shift rights
        token_logprobs = (
            teacher_logprobs[0, :-1, :].gather(1, input_ids[1:].unsqueeze(-1)).squeeze(-1)
        )
        return token_logprobs.mean().item()

    # 3. KL Reward (Aligned)
    # We need student logits corresponding to the full text
    # Assuming student_logits matches full_text length

    # Check alignment
    s_ids = student_tokenizer(full_text, return_tensors="pt").input_ids[0]
    t_ids = teacher_inputs.input_ids[0]

    # Select aligner based on tokenizer identity
    if are_tokenizers_identical(student_tokenizer, teacher_tokenizer):
        aligner = DirectAligner(student_tokenizer, teacher_tokenizer)
    else:
        aligner = GoldAligner(student_tokenizer, teacher_tokenizer)

    # Compute KL loss (scalar)
    # Note: student_logits might need to be computed for full_text if only passed for generation
    # Here we assume caller passed valid logits

    kl = aligner.compute_kl_loss(
        student_logits.unsqueeze(0), teacher_outputs.logits, s_ids.unsqueeze(0), t_ids.unsqueeze(0)
    )
    return -kl.item()  # Negative KL as reward


class _DistillationTrainer(SFTTrainer):
    """SFTTrainer with additional KL distillation loss."""

    def __init__(
        self,
        teacher,
        teacher_tokenizer,
        student_tokenizer,
        same_tokenizer: bool,
        kl_weight: float = 0.5,
        distillation_mode: DistillationMode = DistillationMode.FORWARD_KL,
        slim_top_k: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher = teacher
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.kl_weight = kl_weight
        self.distillation_mode = distillation_mode
        self.slim_top_k = slim_top_k

        # Aligner only needed if we have a teacher model
        self.aligner = None
        if teacher is not None:
            if same_tokenizer:
                self.aligner = DirectAligner(student_tokenizer, teacher_tokenizer)
            else:
                self.aligner = GoldAligner(student_tokenizer, teacher_tokenizer)

        # Initialize Estimator
        if distillation_mode == DistillationMode.SLIM:
            self.estimator = SlimEstimator(temperature=2.0)
        else:
            self.estimator = ForwardKLEstimator(temperature=2.0)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        student_loss = outputs.loss
        student_logits = outputs.logits

        kl_loss = torch.tensor(0.0, device=student_logits.device)

        if self.kl_weight > 0:
            # Check if we have cached teacher logprobs
            if "teacher_logprobs" in inputs:
                kl_loss = self._compute_kl_from_cached_logprobs(
                    student_logits, inputs["teacher_logprobs"], inputs
                )
            elif self.teacher is not None:
                kl_loss = self._compute_kl_from_teacher(student_logits, inputs)

        loss = (1 - self.kl_weight) * student_loss + self.kl_weight * kl_loss
        return (loss, outputs) if return_outputs else loss

    def _compute_kl_from_cached_logprobs(
        self,
        student_logits: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        inputs: dict,
    ) -> torch.Tensor:
        """Compute KL loss using pre-computed teacher logprobs."""
        # teacher_logprobs shape: (batch, seq_len) - just the logprob of chosen tokens
        # We compute KL as: -sum(teacher_logprob) + sum(student_logprob_of_same_token)
        # This is equivalent to minimizing cross-entropy with teacher's distribution

        student_logprobs = F.log_softmax(student_logits, dim=-1)
        seq_len = min(student_logprobs.shape[1], teacher_logprobs.shape[1])

        # Get student logprobs for the actual tokens
        input_ids = inputs["input_ids"][:, :seq_len]
        student_token_logprobs = student_logprobs[:, :seq_len, :].gather(
            2, input_ids.unsqueeze(-1)
        ).squeeze(-1)

        # KL divergence: teacher_logprob - student_logprob (we want student to match teacher)
        # Since we want to minimize, loss = teacher_logprob - student_logprob
        teacher_lp = teacher_logprobs[:, :seq_len]
        kl_per_token = teacher_lp - student_token_logprobs

        # Build mask
        mask = torch.ones_like(kl_per_token)
        pad_id = self.student_tokenizer.pad_token_id
        if pad_id is not None:
            mask = mask * (input_ids != pad_id).float()
        if "labels" in inputs:
            labels = inputs["labels"][:, :seq_len]
            mask = mask * (labels != -100).float()

        # Masked mean
        kl_loss = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)
        return kl_loss

    def _compute_kl_from_teacher(
        self,
        student_logits: torch.Tensor,
        inputs: dict,
    ) -> torch.Tensor:
        """Compute KL loss by running teacher forward pass."""
        with torch.no_grad():
            if isinstance(self.aligner, DirectAligner):
                # Move inputs to teacher device
                teacher_inputs = {
                    k: v.to(self.teacher.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
                teacher_outputs = self.teacher(**teacher_inputs)

                # Direct Alignment Logic: Length Truncation
                min_len = min(student_logits.shape[1], teacher_outputs.logits.shape[1])
                s_logits = student_logits[:, :min_len, :]
                t_logits = teacher_outputs.logits[:, :min_len, :]

                # Prepare Teacher Data (Sparse or Dense)
                if self.distillation_mode == DistillationMode.SLIM:
                    k = self.slim_top_k or 100
                    vals, idxs = t_logits.topk(k, dim=-1)
                    teacher_data = SparseLogits(indices=idxs, values=vals)
                else:
                    teacher_data = t_logits

                # Build mask
                pad_id = self.student_tokenizer.pad_token_id
                if pad_id is not None:
                    mask = (inputs["input_ids"][:, :min_len] != pad_id).float()
                else:
                    mask = torch.ones(
                        inputs["input_ids"][:, :min_len].shape, device=s_logits.device
                    )

                if "labels" in inputs:
                    labels = inputs["labels"][:, :min_len]
                    valid_label_mask = (labels != -100).float()
                    mask = mask * valid_label_mask

                return self.estimator.compute_loss(s_logits, teacher_data, mask=mask)

            else:
                # GoldAligner case (Different Tokenizers)
                texts = self.student_tokenizer.batch_decode(
                    inputs["input_ids"], skip_special_tokens=False
                )
                teacher_inputs = self.teacher_tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=inputs["input_ids"].shape[1] * 2,
                ).to(self.teacher.device)

                teacher_outputs = self.teacher(**teacher_inputs)

                return self.aligner.compute_kl_loss(
                    student_logits,
                    teacher_outputs.logits,
                    inputs["input_ids"],
                    teacher_inputs["input_ids"],
                    temperature=2.0,
                )
