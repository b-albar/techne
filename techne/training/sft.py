"""SFT/DFT training utilities."""

import logging
from typing import Any

import torch
from datasets import Dataset, Features, Sequence, Value
from trl import SFTConfig, SFTTrainer

from techne.config import DistributedBackend, TechneConfig, TrainingAlgorithm
from techne.data import TrainingSample, Trajectory
from techne.training.distributed import build_fsdp_config, is_main_process

logger = logging.getLogger(__name__)


def get_sft_trainer(
    config: TechneConfig,
    model,
    tokenizer,
    samples: list[TrainingSample] | list[Trajectory] | Any,
    **kwargs,
) -> SFTTrainer:
    """Create an SFTTrainer for supervised fine-tuning.

    Args:
        config: Techne configuration
        model: The model to train
        tokenizer: The tokenizer
        samples: Training samples (list of TrainingSample, list of Trajectory, or HF Dataset)
        **kwargs: Additional arguments for SFTConfig

    Returns:
        Configured SFTTrainer
    """
    algo = config.training.algorithm
    loss_type = "dft" if algo == TrainingAlgorithm.DFT else "nll"

    args_dict = get_common_training_args(config)
    if config.training.max_seq_length is not None:
        args_dict["max_length"] = config.training.max_seq_length
    args_dict["packing"] = False
    args_dict.update(kwargs)
    args_dict.pop("remove_unused_columns", None)

    args = SFTConfig(
        **args_dict,
        loss_type=loss_type,
    )

    train_dataset = samples

    # Convert typed objects to dicts for SFTTrainer
    # Explicit features to ensure input_ids/labels are integers (not floats)
    int_features = Features({
        "input_ids": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
    })

    max_len = config.training.max_seq_length

    def _fits(n_tokens: int) -> bool:
        return max_len is None or n_tokens <= max_len

    if isinstance(samples, list) and len(samples) > 0:
        first = samples[0]
        if isinstance(first, Trajectory):
            data_list = []
            for traj in samples:
                sample = traj.to_training_sample(tokenizer=tokenizer)
                if _fits(len(sample.input_ids)):
                    data_list.append({"input_ids": sample.input_ids, "labels": sample.labels})
            filtered = len(samples) - len(data_list)
            if filtered > 0 and is_main_process():
                logger.info(f"Filtered {filtered}/{len(samples)} samples exceeding max_seq_length={max_len}")
            train_dataset = Dataset.from_list(data_list, features=int_features)
        elif isinstance(first, TrainingSample):
            data_list = [
                {"input_ids": s.input_ids, "labels": s.labels}
                for s in samples if _fits(len(s.input_ids))
            ]
            filtered = len(samples) - len(data_list)
            if filtered > 0 and is_main_process():
                logger.info(f"Filtered {filtered}/{len(samples)} samples exceeding max_seq_length={max_len}")
            train_dataset = Dataset.from_list(data_list, features=int_features)
        elif isinstance(first, dict):
            data_list = [s for s in samples if _fits(len(s.get("input_ids", [])))]
            filtered = len(samples) - len(data_list)
            if filtered > 0 and is_main_process():
                logger.info(f"Filtered {filtered}/{len(samples)} samples exceeding max_seq_length={max_len}")
            train_dataset = Dataset.from_list(data_list, features=int_features)
        else:
            raise TypeError(f"Unsupported sample type: {type(first)}")

    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )


def get_common_training_args(config: TechneConfig) -> dict:
    """Get common training arguments from config.

    When distributed_backend is FSDP, passes the appropriate
    configuration so HF Trainer handles multi-GPU data parallelism.

    Args:
        config: Techne configuration

    Returns:
        Dictionary of training arguments
    """
    args = {
        "output_dir": config.output_dir,
        "learning_rate": config.training.learning_rate,
        "per_device_train_batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "max_steps": config.training.max_steps,
        "num_train_epochs": config.training.num_train_epochs,
        "warmup_steps": config.training.warmup_steps,
        "weight_decay": config.training.weight_decay,
        "max_grad_norm": config.training.max_grad_norm,
        "bf16": config.model.dtype == torch.bfloat16,
        "report_to": config.training.report_to,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_strategy": "steps",
        "remove_unused_columns": False,
        "disable_tqdm": not is_main_process(),
        "log_level": "info" if is_main_process() else "warning",
    }

    if config.training.distributed_backend == DistributedBackend.FSDP:
        args.update(build_fsdp_config(config.training.tensor_parallel_size))

    return args
