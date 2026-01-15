"""Distillation training utilities.

This module provides:
- Offline distillation: train on teacher completions with optional KL loss
- Tokenizer alignment for cross-tokenizer distillation (GOLD approach)
- KL reward computation for on-policy distillation
"""

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer

from techne.config import DistillationMode, TechneConfig
from techne.training.alignment import DirectAligner, GoldAligner
from techne.training.estimators import (
    ForwardKLEstimator,
    SlimEstimator,
    SparseLogits,
)
from techne.training.sft import get_common_training_args, get_sft_trainer

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


def compute_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    valid_length: int | None = None,
) -> torch.Tensor:
    """Compute KL divergence loss between student and teacher logits.

    Args:
        student_logits: Student model logits [batch, seq, vocab]
        teacher_logits: Teacher model logits [batch, seq, vocab]
        temperature: Softmax temperature for softer distributions
        valid_length: Optional length to truncate both to (for alignment)

    Returns:
        KL divergence loss (scalar)
    """
    # Align sequence lengths if needed
    if valid_length is not None:
        student_logits = student_logits[:, :valid_length, :]
        teacher_logits = teacher_logits[:, :valid_length, :]

    # Handle vocabulary size mismatch
    student_vocab = student_logits.shape[-1]
    teacher_vocab = teacher_logits.shape[-1]

    if student_vocab != teacher_vocab:
        min_vocab = min(student_vocab, teacher_vocab)
        student_logits = student_logits[:, :, :min_vocab]
        teacher_logits = teacher_logits[:, :, :min_vocab]

    # Compute KL divergence with temperature scaling
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    kl_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
    return kl_loss * (temperature**2)


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

    # If no teacher model specified, fall back to simple SFT on teacher data
    if not teacher_model_path:
        print("No teacher model specified. Using SFT-style distillation on teacher outputs...")
        trainer = get_sft_trainer(config, model, tokenizer, dataset, **kwargs)
        return trainer.train()

    # Data Validation
    assert len(dataset) > 0, "Distillation dataset is empty!"

    # Check structure of first sample
    sample = dataset[0]
    has_input = "input_ids" in sample or "prompt" in sample
    assert has_input, (
        f"Dataset must contain 'input_ids' or 'prompt' column. Found: {list(sample.keys())}"
    )

    if "input_ids" in sample:
        ids = sample["input_ids"]
        assert len(ids) > 0, "Found sample with empty 'input_ids'!"
        if isinstance(ids, torch.Tensor):
            assert ids.numel() > 0, "Found sample with empty 'input_ids' tensor!"

    # Load teacher model and tokenizer
    print(f"Loading teacher model for logit distillation: {teacher_model_path}...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float16,
    )
    teacher_model.eval()
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)

    # Check if tokenizers are identical
    same_tokenizer = are_tokenizers_identical(tokenizer, teacher_tokenizer)

    if same_tokenizer:
        print("Tokenizers are identical - using direct logit alignment")
    else:
        print("Different tokenizers detected - using GOLD approach (decode & re-tokenize)")

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

    print(f"Starting offline distillation with KL loss on {len(dataset)} samples...")
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
            with torch.no_grad():
                # Prepare teacher inputs
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

                    # Compute Loss using Estimator
                    # Handle potential None pad_token_id
                    pad_id = self.student_tokenizer.pad_token_id
                    if pad_id is not None:
                        mask = (inputs["input_ids"][:, :min_len] != pad_id).float()
                    else:
                        mask = torch.ones(
                            inputs["input_ids"][:, :min_len].shape, device=s_logits.device
                        )
                    kl_loss = self.estimator.compute_loss(s_logits, teacher_data, mask=mask)

                else:
                    # GoldAligner case (Different Tokenizers)
                    # Re-tokenize
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

                    # Compute Aligned KL (GoldAligner encapsulates ULD logic)
                    kl_loss = self.aligner.compute_kl_loss(
                        student_logits,
                        teacher_outputs.logits,
                        inputs["input_ids"],
                        teacher_inputs["input_ids"],
                        temperature=2.0,
                    )

        loss = (1 - self.kl_weight) * student_loss + self.kl_weight * kl_loss
        return (loss, outputs) if return_outputs else loss
