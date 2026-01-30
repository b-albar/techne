"""Shared distributed training utilities.

Centralizes distributed environment checks, FSDP configuration,
process group management, device placement, and launcher validation.
"""

from __future__ import annotations

import os

from techne.config import DistributedBackend


# =============================================================================
# Environment checks
# =============================================================================


def is_main_process() -> bool:
    """Check if this is the main process (rank 0) in distributed training."""
    return int(os.environ.get("RANK", "0")) == 0


def is_distributed_env() -> bool:
    """Check if we're already running inside a distributed launcher (torchrun/accelerate)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


# =============================================================================
# FSDP configuration
# =============================================================================


def build_fsdp_config(tp_size: int = 1) -> dict[str, str | dict]:
    """Build FSDP strategy and config dict for HF Trainer.

    Args:
        tp_size: Tensor parallel size. When >1, uses hybrid_shard
                 for combined TP+DP parallelism.

    Returns:
        Dict with "fsdp" strategy string and "fsdp_config" sub-dict,
        ready to merge into SFTConfig / TrainingArguments kwargs.
    """
    base_config = {
        "backward_prefetch": "backward_pre",
        "forward_prefetch": True,
        "use_orig_params": True,
    }

    if tp_size > 1:
        base_config["sharding_factor"] = tp_size
        return {
            "fsdp": "hybrid_shard auto_wrap",
            "fsdp_config": base_config,
        }

    return {
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": base_config,
    }


# =============================================================================
# Process group management
# =============================================================================


def init_process_group(rank: int, world_size: int) -> None:
    """Initialize NCCL distributed process group.

    Sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT env vars
    and calls torch.distributed.init_process_group.
    """
    import torch.distributed

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def destroy_process_group() -> None:
    """Destroy the distributed process group if initialized."""
    import torch.distributed

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# =============================================================================
# Device placement
# =============================================================================


def get_device_map(
    backend: DistributedBackend,
    num_workers: int = 1,
) -> str | None:
    """Determine device_map for model loading.

    FSDP with multiple workers manages its own sharding and needs
    device_map=None. All other cases use device_map="auto".
    """
    wants_fsdp = backend == DistributedBackend.FSDP and num_workers > 1
    return None if wants_fsdp else "auto"


def detect_device(model) -> str:
    """Detect the appropriate device string from a model.

    Handles TP models (device_map="auto") where .device may point
    to a specific GPU, meta-device models, and standard CUDA models.
    """
    import torch

    if hasattr(model, "device") and model.device.type != "meta":
        return str(model.device)
    return "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Launcher validation
# =============================================================================


def check_distributed_launcher(
    backend: DistributedBackend,
    num_workers: int,
    tp_size: int = 1,
) -> None:
    """Raise RuntimeError if FSDP is configured without a multi-process launcher.

    No-op if backend is not FSDP, if only 1 worker, or if already
    inside a distributed context (RANK/WORLD_SIZE set).
    """
    wants_fsdp = backend == DistributedBackend.FSDP and num_workers > 1
    if not wants_fsdp or is_distributed_env():
        return

    total_procs = num_workers * tp_size
    hint = f"accelerate launch --use_fsdp --num_processes {total_procs} your_script.py"
    tp_note = f" x {tp_size} TP size" if tp_size > 1 else ""
    raise RuntimeError(
        f"Distributed training requires a multi-process launcher. "
        f"Config requests FSDP with {num_workers} DP workers"
        f"{tp_note} = {total_procs} processes, but the "
        f"current process was not launched in a distributed context "
        f"(RANK/WORLD_SIZE not set). Launch with:\n  {hint}"
    )
