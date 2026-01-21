"""SFT/DFT training utilities."""

from typing import Any

import torch
from trl import SFTConfig, SFTTrainer

from techne.config import TechneConfig, TrainingAlgorithm
from techne.data import TrainingSample


def get_sft_trainer(
    config: TechneConfig,
    model,
    tokenizer,
    samples: list[TrainingSample] | Any,
    **kwargs,
) -> SFTTrainer:
    """Create an SFTTrainer for supervised fine-tuning.

    Args:
        config: Techne configuration
        model: The model to train
        tokenizer: The tokenizer
        samples: Training samples (list of TrainingSample or HF Dataset)
        **kwargs: Additional arguments for SFTConfig

    Returns:
        Configured SFTTrainer
    """
    algo = config.training.algorithm
    loss_type = "dft" if algo == TrainingAlgorithm.DFT else "nll"

    args_dict = get_common_training_args(config)
    args_dict["max_length"] = config.training.max_seq_length
    args_dict["packing"] = False
    args_dict.update(kwargs)
    args_dict.pop("remove_unused_columns", None)

    args = SFTConfig(
        **args_dict,
        loss_type=loss_type,
    )

    train_dataset = samples
    # Convert TrainingSample list to dicts if needed
    if isinstance(samples, list) and len(samples) > 0 and hasattr(samples[0], "input_ids"):
        train_dataset = [{"input_ids": s.input_ids, "labels": s.labels} for s in samples]

    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )


def get_common_training_args(config: TechneConfig) -> dict:
    """Get common training arguments from config.

    Args:
        config: Techne configuration

    Returns:
        Dictionary of training arguments
    """
    return {
        "output_dir": config.output_dir,
        "learning_rate": config.training.learning_rate,
        "per_device_train_batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "max_steps": config.training.max_steps,
        "num_train_epochs": config.training.num_train_epochs,
        "warmup_ratio": config.training.warmup_ratio,
        "weight_decay": config.training.weight_decay,
        "max_grad_norm": config.training.max_grad_norm,
        "bf16": config.model.dtype == torch.bfloat16,
        "report_to": config.training.report_to,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_strategy": "steps",
        "remove_unused_columns": False,
        "torch_compile": config.model.compile,
    }
