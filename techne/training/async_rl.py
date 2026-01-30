"""Async RL training using Ray.

Architecture:
- InferenceWorker: Generates completions + computes ref logprobs (HuggingFace)
- DynamicBatcher: Continuously dispatches to workers, collects results dynamically
- TrainingWorker: Distributed training with FSDP/hybrid TP+DP support

Supports: GRPO, PPO, GSPO, DISTILL
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Suppress Ray warnings
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "0")
os.environ.setdefault("RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING", "0")
os.environ.setdefault("RAY_metrics_report_interval_ms", "0")

import ray  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402, N812
from tqdm.auto import tqdm  # noqa: E402

from techne.config import (  # noqa: E402
    DistributedBackend,
    InferenceConfig,
    OptimizerType,
    SchedulerType,
    TechneConfig,
    TrainingConfig,
)
from techne.training.model import InferenceModel  # noqa: E402

# Type alias for model factories
ModelFactory = Callable[[], InferenceModel]


def create_optimizer(
    params,
    training_config: TrainingConfig,
) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    lr = training_config.learning_rate
    wd = training_config.weight_decay
    extra = training_config.optimizer_kwargs

    if training_config.optimizer == OptimizerType.ADAMW:
        betas = (extra.get("beta1", 0.9), extra.get("beta2", 0.999))
        eps = extra.get("eps", 1e-8)
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    elif training_config.optimizer == OptimizerType.ADAM:
        betas = (extra.get("beta1", 0.9), extra.get("beta2", 0.999))
        eps = extra.get("eps", 1e-8)
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    elif training_config.optimizer == OptimizerType.SGD:
        momentum = extra.get("momentum", 0.0)
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
    elif training_config.optimizer == OptimizerType.ADAFACTOR:
        try:
            from transformers.optimization import Adafactor

            return Adafactor(
                params,
                lr=lr,
                weight_decay=wd,
                scale_parameter=extra.get("scale_parameter", False),
                relative_step=extra.get("relative_step", False),
            )
        except ImportError:
            raise ImportError("Adafactor requires the `transformers` package.")
    else:
        raise ValueError(f"Unknown optimizer: {training_config.optimizer}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    max_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create LR scheduler from config."""
    warmup_steps = training_config.warmup_steps

    if training_config.scheduler == SchedulerType.COSINE:

        def cosine_with_warmup(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)

    elif training_config.scheduler == SchedulerType.LINEAR:

        def linear_with_warmup(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            return max(0.0, 1.0 - (current_step - warmup_steps) / max(1, max_steps - warmup_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, linear_with_warmup)

    elif training_config.scheduler == SchedulerType.CONSTANT_WITH_WARMUP:

        def constant_with_warmup(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, constant_with_warmup)

    elif training_config.scheduler == SchedulerType.CONSTANT:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    else:
        raise ValueError(f"Unknown scheduler: {training_config.scheduler}")


def get_merged_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Get state dict from model, merging LoRA weights if present.

    When the trainer uses LoRA (via peft), the state dict has keys prefixed with
    'base_model.model.' and includes LoRA adapter weights. Inference workers use
    plain HuggingFace models without LoRA. This function merges LoRA weights into
    the base model and returns a state dict compatible with plain HF models.
    """
    # Check if this is a peft model with LoRA adapters
    if hasattr(model, "merge_and_unload"):
        # This is a peft model - we need to merge LoRA weights
        # Use merge_adapter() to merge without unloading, then get state dict
        # from the underlying base model
        try:
            # Merge LoRA weights into base model weights
            model.merge_adapter()
            # Get state dict from the base model (without 'base_model.model.' prefix)
            base_model = model.get_base_model()
            state_dict = {k: v.cpu() for k, v in base_model.state_dict().items()}
            # Unmerge so training can continue
            model.unmerge_adapter()
            return state_dict
        except Exception as e:
            logger.warning(f"Failed to merge LoRA adapter: {e}. Falling back to direct state dict.")
            # Fall through to default behavior

    # Standard model - just return state dict
    return {k: v.cpu() for k, v in model.state_dict().items()}


@dataclass
class Sample:
    """A single generated sample."""

    prompt: str
    completion: str
    prompt_ids: list[int]
    completion_ids: list[int]
    ref_logprobs: list[float]
    old_logprobs: list[float] | None = None  # Policy logprobs at generation time (for ratio)
    reward: float = 0.0
    advantage: float = 0.0
    policy_version: int = 0  # For staleness tracking


class AsyncResultCache:
    """Async cache for collecting worker results with dynamic batching.

    Results are pushed as they complete and batches are formed as soon as
    enough samples are available, regardless of which workers produced them.
    """

    def __init__(self, minibatch_size: int = 8):
        self.minibatch_size = minibatch_size
        self._queue: asyncio.Queue[Sample] = asyncio.Queue()
        self._pending: list[Sample] = []
        self._lock = asyncio.Lock()

    async def push(self, samples: list[Sample]) -> None:
        """Push completed samples to the cache."""
        for sample in samples:
            await self._queue.put(sample)

    async def push_one(self, sample: Sample) -> None:
        """Push a single sample to the cache."""
        await self._queue.put(sample)

    async def get_batch(self, timeout: float = 0.5) -> list[Sample] | None:
        """Get a batch of samples as soon as minibatch_size are available.

        Returns None if timeout expires before enough samples arrive.
        """
        async with self._lock:
            # First drain any pending samples from queue
            while not self._queue.empty():
                try:
                    sample = self._queue.get_nowait()
                    self._pending.append(sample)
                except asyncio.QueueEmpty:
                    break

            # If we have enough, return a batch immediately
            if len(self._pending) >= self.minibatch_size:
                batch = self._pending[: self.minibatch_size]
                self._pending = self._pending[self.minibatch_size :]
                return batch

        # Wait for more samples with timeout
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None

            try:
                sample = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                async with self._lock:
                    self._pending.append(sample)
                    if len(self._pending) >= self.minibatch_size:
                        batch = self._pending[: self.minibatch_size]
                        self._pending = self._pending[self.minibatch_size :]
                        return batch
            except asyncio.TimeoutError:
                return None

    def pending_count(self) -> int:
        """Return number of samples waiting in the cache."""
        return len(self._pending) + self._queue.qsize()


@dataclass
class Minibatch:
    """Batch of samples for training."""

    samples: list[Sample]


@ray.remote
class InferenceWorker:
    """Generates completions and computes reference logprobs."""

    def __init__(
        self,
        model_factory: ModelFactory,
        num_generations: int = 4,
        max_new_tokens: int = 512,
        max_seq_length: int | None = None,
        temperature: float = 0.7,
        agent_class: Any | None = None,
        agent_config: TechneConfig | None = None,
        reward_fn_class: type | None = None,
    ):
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        self.temperature = temperature
        self.agent_class = agent_class
        self.agent_config = agent_config

        self.model = model_factory()
        self.tokenizer = self.model.get_tokenizer()
        self.device = self.model.device

        # Initialize agent if class provided
        if self.agent_class:
            self.agent = self.agent_class(
                self.agent_config, model=self.model, tokenizer=self.tokenizer
            )
        else:
            self.agent = None

        # Instantiate reward function locally to avoid serialization issues
        self.reward_fn = reward_fn_class() if reward_fn_class else None

        # Metrics tracking
        self._metrics = {
            "prompts_processed": 0,
            "trajectories_collected": 0,
            "total_reward": 0.0,
            "positive_rewards": 0,
            "generation_time": 0.0,
            "filtered_long": 0,
        }
        self._start_time = time.time()

    def get_metrics(self) -> dict:
        """Return current worker metrics."""
        elapsed = time.time() - self._start_time
        return {
            **self._metrics,
            "elapsed_time": elapsed,
            "prompts_per_sec": self._metrics["prompts_processed"] / elapsed if elapsed > 0 else 0,
        }

    async def generate_and_score(
        self,
        samples_meta: list[dict],
    ) -> list[Sample]:
        """Generate completions and compute reference logprobs."""
        gen_start = time.time()
        ret_samples = []

        for sample in samples_meta:
            # Extract prompt content
            if "prompt" in sample:
                prompt_content = sample["prompt"]
            elif "text" in sample:
                prompt_content = sample["text"]
            else:
                prompt_content = str(sample)

            # Tokenize prompt correctly
            if isinstance(prompt_content, list):
                # Chat template
                prompt_ids = self.tokenizer.apply_chat_template(
                    prompt_content, tokenize=True, add_generation_prompt=True
                )
            else:
                prompt_ids = self.tokenizer(prompt_content).input_ids

            prompt_ids_tensor = torch.tensor([prompt_ids], device=self.device)

            if self.agent:
                # Use Agent rollouts (which can have tool calls, multi-turn, etc.)
                # Repeat prompt for num_generations
                agent_prompts = [prompt_content] * self.num_generations
                trajs = await self.agent.collect_trajectories(agent_prompts)

                for traj in trajs:
                    completion_ids = []
                    completion_text = ""

                    for step in traj.steps:
                        if step.role in ["assistant", "tool"]:
                            completion_ids.extend(step.token_ids)
                            completion_text += step.content

                    # Drop sequences exceeding max_seq_length
                    if self.max_seq_length is not None and len(prompt_ids) + len(completion_ids) > self.max_seq_length:
                        self._metrics["filtered_long"] += 1
                        continue

                    # Compute both reference and current policy logprobs
                    # ref_logprobs: from reference/initial model (for KL penalty)
                    # old_logprobs: from current policy at generation time (for ratio)
                    logprobs = self.model.compute_logprobs(prompt_ids, completion_ids)
                    ref_logprobs = logprobs
                    old_logprobs = logprobs  # Same model at generation time

                    reward = 0.0
                    if self.reward_fn:
                        try:
                            reward = self.reward_fn(sample, completion_text)
                        except Exception:
                            reward = 0.0

                    ret_samples.append(
                        Sample(
                            prompt=prompt_content,
                            completion=completion_text,
                            prompt_ids=prompt_ids,
                            completion_ids=completion_ids,
                            ref_logprobs=ref_logprobs,
                            old_logprobs=old_logprobs,
                            reward=reward,
                        )
                    )
            else:
                # Raw model generation
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=prompt_ids_tensor,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        num_return_sequences=self.num_generations,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                    )

                for seq in outputs.sequences:
                    completion_ids = seq[len(prompt_ids) :].tolist()

                    # Drop sequences exceeding max_seq_length
                    if self.max_seq_length is not None and len(prompt_ids) + len(completion_ids) > self.max_seq_length:
                        self._metrics["filtered_long"] += 1
                        continue

                    completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                    # Compute both reference and current policy logprobs
                    logprobs = self.model.compute_logprobs(prompt_ids, completion_ids)
                    ref_logprobs = logprobs
                    old_logprobs = logprobs  # Same model at generation time

                    reward = 0.0
                    if self.reward_fn:
                        try:
                            reward = self.reward_fn(sample, completion)
                        except Exception:
                            reward = 0.0

                    ret_samples.append(
                        Sample(
                            prompt=prompt_content,
                            completion=completion,
                            prompt_ids=prompt_ids,
                            completion_ids=completion_ids,
                            ref_logprobs=ref_logprobs,
                            old_logprobs=old_logprobs,
                            reward=reward,
                        )
                    )

        # Update metrics
        self._metrics["prompts_processed"] += len(samples_meta)
        self._metrics["trajectories_collected"] += len(ret_samples)
        self._metrics["total_reward"] += sum(s.reward for s in ret_samples)
        self._metrics["positive_rewards"] += sum(1 for s in ret_samples if s.reward > 0)
        self._metrics["generation_time"] += time.time() - gen_start

        return ret_samples

    def update_weights(self, state_dict: dict):
        """Update model weights from trainer."""
        self.model.load_state_dict(state_dict)


@ray.remote
class TrainingWorker:
    """Distributed training worker with FSDP/hybrid TP+DP support."""

    def __init__(
        self,
        inference_config: InferenceConfig,
        rank: int,
        world_size: int,
        distributed_backend: DistributedBackend = DistributedBackend.NONE,
        training_config: TrainingConfig | None = None,
        tensor_parallel_size: int = 1,
    ):
        from techne.training.distributed import init_process_group

        self.rank = rank
        self.world_size = world_size
        self.distributed_backend = distributed_backend
        self.training_config = training_config
        self._hybrid_mode = False

        uses_device_map = (
            distributed_backend == DistributedBackend.TP
            or (tensor_parallel_size > 1 and distributed_backend == DistributedBackend.FSDP)
        )

        if uses_device_map:
            # TP or Hybrid TP+DP: load with device_map="auto".
            # For hybrid, DP gradient sync is manual (allreduce via Ray).
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                inference_config.name_or_path,
                trust_remote_code=getattr(inference_config, "trust_remote_code", False),
                torch_dtype=inference_config.dtype,
                attn_implementation=getattr(inference_config, "attn_implementation", None),
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                inference_config.name_or_path,
                trust_remote_code=getattr(inference_config, "trust_remote_code", False),
            )
            self._hybrid_mode = tensor_parallel_size > 1
        elif distributed_backend == DistributedBackend.FSDP:
            init_process_group(rank, world_size)

            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817

            # Create base model via config, then wrap with FSDP
            base_model = inference_config.create_training_model()
            self.model = FSDP(base_model._model.cuda())
            self.tokenizer = base_model.get_tokenizer()
        else:
            training_model = inference_config.create_training_model()
            self.model = training_model._model
            self.tokenizer = training_model.get_tokenizer()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Device placement: device_map models expose .device, FSDP pins to rank.
        if uses_device_map:
            self.device = str(self.model.device)
        elif distributed_backend == DistributedBackend.FSDP:
            self.device = f"cuda:{rank}"
        else:
            self.device = "cuda"

    def train_step(
        self,
        batch: Minibatch,
        optimizer_state: dict | None,
        algorithm: str,
        clip_eps: float,
        kl_coef: float,
        lr: float,
        weight_decay: float,
        clip_range_ratio: list[float] | None = None,
    ) -> tuple[dict, dict]:
        """Execute a training step and return metrics + new optimizer state."""
        if not hasattr(self, "optimizer"):
            if self.training_config is not None:
                self.optimizer = create_optimizer(self.model.parameters(), self.training_config)
            else:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                )
        elif optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        self.model.train()
        loss, metrics = compute_rl_loss(
            model=self.model,
            batch=batch,
            tokenizer=self.tokenizer,
            device=self.device,
            algorithm=algorithm,
            clip_eps=clip_eps,
            kl_coef=kl_coef,
            clip_range_ratio=clip_range_ratio,
        )

        loss.backward()
        return metrics, self.optimizer.state_dict()

    def optimizer_step(self, max_grad_norm: float):
        """Execute optimizer step after gradient accumulation."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_state_dict(self) -> dict:
        """Get model state dict for syncing to inference workers."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def get_gradients(self) -> dict[str, torch.Tensor]:
        """Collect gradients from all parameters (for manual allreduce in hybrid mode)."""
        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.cpu()
        return grads

    def set_averaged_gradients(self, avg_grads: dict[str, torch.Tensor]):
        """Set averaged gradients back to model parameters (for hybrid mode)."""
        for name, param in self.model.named_parameters():
            if name in avg_grads:
                param.grad = avg_grads[name].to(param.device)

    def cleanup(self):
        """Cleanup distributed process group."""
        if self.distributed_backend == DistributedBackend.FSDP and not self._hybrid_mode:
            from techne.training.distributed import destroy_process_group

            destroy_process_group()


def compute_advantages(samples: list[Sample]) -> list[Sample]:
    """Compute advantages for samples grouped by prompt (GRPO style)."""
    prompt_groups: dict[Any, list[Sample]] = {}
    for s in samples:
        # Handle unhashable prompts (e.g. list of messages)
        if isinstance(s.prompt, list):
            key = tuple(frozenset(d.items()) for d in s.prompt)
        else:
            key = s.prompt

        if key not in prompt_groups:
            prompt_groups[key] = []
        prompt_groups[key].append(s)

    for group in prompt_groups.values():
        rewards = [s.reward for s in group]
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        std_r = (
            (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            if len(rewards) > 1
            else 1.0
        )
        std_r = max(std_r, 1e-8)

        for s in group:
            s.advantage = (s.reward - mean_r) / std_r

    return samples


def should_skip_batch(samples: list[Sample], min_advantage_std: float = 1e-6) -> bool:
    """Check if batch should be skipped (zero gradient in GRPO).

    In GRPO, if all rewards in a group are identical, advantages are zero
    and the gradient is zero. This wastes compute.

    From Open-Instruct: "groups of instances whose rewards are identical
    (which leads to zero gradients in GRPO) are removed."
    """
    # Check if all advantages are effectively zero
    advantages = [s.advantage for s in samples]
    if not advantages:
        return True
    std = (
        sum((a - sum(advantages) / len(advantages)) ** 2 for a in advantages) / len(advantages)
    ) ** 0.5
    return std < min_advantage_std


class DynamicBatcher:
    """Manages continuous dispatch to workers and collects results dynamically.

    Key features:
    - Workers are kept busy with continuous dispatch
    - Results are collected as they arrive (not waiting for all workers)
    - Batches are formed as soon as minibatch_size samples are ready
    - Decouples generation speed from training speed
    - Active sampling filter to skip zero-gradient batches (Open-Instruct)
    - Staleness tracking to bound off-policy drift (AReaL/VeRL)
    """

    def __init__(
        self,
        workers: list,
        dataset: Any,
        minibatch_size: int = 8,
        max_pending_per_worker: int = 2,
        num_prompts_per_dispatch: int = 1,
        staleness_threshold: int = 3,
    ):
        self.workers = workers
        self.dataset = dataset
        self.minibatch_size = minibatch_size
        self.max_pending_per_worker = max_pending_per_worker
        self.num_prompts_per_dispatch = num_prompts_per_dispatch
        self.staleness_threshold = staleness_threshold

        self.cache = AsyncResultCache(minibatch_size)
        self.prompt_idx = 0
        self._worker_pending: dict[int, int] = {i: 0 for i in range(len(workers))}
        self._active_tasks: set[asyncio.Task] = set()
        self._stop_event = asyncio.Event()
        self._dispatch_task: asyncio.Task | None = None

        # Policy version tracking
        self._policy_version = 0
        self._policy_version_lock = asyncio.Lock()

        # Stats
        self.stats = {
            "samples_generated": 0,
            "samples_dropped_stale": 0,
            "batches_skipped_zero_grad": 0,
        }

        # Inference metrics tracking
        self._start_time = time.time()
        self._response_lengths: list[int] = []  # Track recent response lengths
        self._max_response_history = 100  # Keep last N for moving average

    def start(self) -> None:
        """Start the continuous dispatch loop."""
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        """Stop dispatching and wait for pending tasks."""
        self._stop_event.set()
        if self._dispatch_task:
            await self._dispatch_task
        # Wait for remaining active tasks
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

    async def _dispatch_loop(self) -> None:
        """Continuously dispatch work to workers that have capacity."""
        while not self._stop_event.is_set():
            dispatched = False

            for worker_idx, worker in enumerate(self.workers):
                # Check if worker has capacity
                if self._worker_pending[worker_idx] >= self.max_pending_per_worker:
                    continue

                # Check if we have more prompts
                if self.prompt_idx >= len(self.dataset):
                    continue

                # Collect multiple prompts for this dispatch
                prompts_batch = []
                for _ in range(self.num_prompts_per_dispatch):
                    if self.prompt_idx >= len(self.dataset):
                        break
                    prompts_batch.append(self.dataset[self.prompt_idx])
                    self.prompt_idx += 1

                if not prompts_batch:
                    continue

                self._worker_pending[worker_idx] += 1
                dispatched = True

                # Capture current policy version for this generation
                generation_version = self._policy_version

                # Create task to handle this worker's result
                task = asyncio.create_task(
                    self._handle_worker_result(
                        worker, worker_idx, prompts_batch, generation_version
                    )
                )
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)

            # If no work dispatched and all prompts consumed, we're done
            if not dispatched:
                if self.prompt_idx >= len(self.dataset) and all(
                    p == 0 for p in self._worker_pending.values()
                ):
                    break
                await asyncio.sleep(0.01)  # Small delay before retry

    async def _handle_worker_result(
        self, worker, worker_idx: int, prompts: list[dict], generation_version: int
    ) -> None:
        """Handle result from a single worker dispatch."""
        try:
            future = worker.generate_and_score.remote(prompts)
            samples = await asyncio.wrap_future(future.future())

            # Stamp samples with the policy version they were generated under
            for s in samples:
                s.policy_version = generation_version

            # Compute advantages and push to cache immediately
            samples = compute_advantages(samples)
            self.stats["samples_generated"] += len(samples)

            # Track response lengths for metrics
            for s in samples:
                self._response_lengths.append(len(s.completion_ids))
            # Keep only recent history
            if len(self._response_lengths) > self._max_response_history:
                self._response_lengths = self._response_lengths[-self._max_response_history :]

            await self.cache.push(samples)
        except Exception as e:
            logger.error(f"Worker {worker_idx} error: {e}")
        finally:
            self._worker_pending[worker_idx] -= 1

    async def get_batch(
        self,
        timeout: float = 0.5,
        current_policy_version: int | None = None,
        skip_zero_gradient: bool = True,
    ) -> list[Sample] | None:
        """Get the next available batch from the cache.

        Args:
            timeout: Max time to wait for a batch
            current_policy_version: If provided, filter out samples that are too stale
            skip_zero_gradient: If True, skip batches where all advantages are ~0
        """
        while True:
            batch = await self.cache.get_batch(timeout)
            if batch is None:
                return None

            # Filter stale samples if staleness tracking is enabled
            if current_policy_version is not None and self.staleness_threshold > 0:
                fresh_samples = [
                    s
                    for s in batch
                    if current_policy_version - s.policy_version <= self.staleness_threshold
                ]
                stale_count = len(batch) - len(fresh_samples)
                if stale_count > 0:
                    self.stats["samples_dropped_stale"] += stale_count
                    # Re-push fresh samples as a group if not enough for a batch
                    if len(fresh_samples) < self.minibatch_size:
                        if fresh_samples:
                            await self.cache.push(fresh_samples)
                        continue
                    batch = fresh_samples[: self.minibatch_size]
                    # Re-push any excess fresh samples as a group
                    excess = fresh_samples[self.minibatch_size :]
                    if excess:
                        await self.cache.push(excess)

            # Skip zero-gradient batches (Open-Instruct active sampling filter)
            if skip_zero_gradient and should_skip_batch(batch):
                self.stats["batches_skipped_zero_grad"] += 1
                rewards = [s.reward for s in batch]
                logger.debug(
                    f"Skipping zero-gradient batch: rewards={rewards}, "
                    f"total_skipped={self.stats['batches_skipped_zero_grad']}"
                )
                continue

            return batch

    async def increment_policy_version(self) -> int:
        """Increment policy version after a weight sync. Returns new version."""
        async with self._policy_version_lock:
            self._policy_version += 1
            return self._policy_version

    @property
    def policy_version(self) -> int:
        """Current policy version."""
        return self._policy_version

    @property
    def prompts_consumed(self) -> int:
        """Number of prompts that have been dispatched."""
        return self.prompt_idx

    @property
    def is_exhausted(self) -> bool:
        """True if all prompts consumed and no pending work."""
        return (
            self.prompt_idx >= len(self.dataset)
            and all(p == 0 for p in self._worker_pending.values())
            and self.cache.pending_count() < self.minibatch_size
        )

    async def fetch_worker_metrics(self) -> dict[str, Any]:
        """Fetch and aggregate metrics from all workers."""
        try:
            futures = [w.get_metrics.remote() for w in self.workers]
            metrics_list = await asyncio.gather(
                *[asyncio.wrap_future(f.future()) for f in futures],
                return_exceptions=True,
            )

            # Aggregate metrics
            total_prompts = 0
            total_trajectories = 0
            total_reward = 0.0
            positive_rewards = 0
            total_gen_time = 0.0
            total_filtered_long = 0

            for m in metrics_list:
                if isinstance(m, dict):
                    total_prompts += m.get("prompts_processed", 0)
                    total_trajectories += m.get("trajectories_collected", 0)
                    total_reward += m.get("total_reward", 0.0)
                    positive_rewards += m.get("positive_rewards", 0)
                    total_gen_time += m.get("generation_time", 0.0)
                    total_filtered_long += m.get("filtered_long", 0)

            return {
                "prompts": total_prompts,
                "trajs": total_trajectories,
                "reward": total_reward,
                "pos_rewards": positive_rewards,
                "gen_time": total_gen_time,
                "filtered_long": total_filtered_long,
            }
        except Exception:
            return {}

    def get_inference_metrics(self, worker_metrics: dict | None = None) -> dict[str, float]:
        """Get current inference metrics for progress display."""
        elapsed = time.time() - self._start_time
        samples_generated = self.stats["samples_generated"]

        # Samples per second
        samples_per_sec = samples_generated / elapsed if elapsed > 0 else 0.0

        # Average response length (from recent samples)
        avg_resp_len = (
            sum(self._response_lengths) / len(self._response_lengths)
            if self._response_lengths
            else 0.0
        )

        # Pending samples in cache
        pending = self.cache.pending_count()

        # Active workers (those with pending tasks)
        active_workers = sum(1 for p in self._worker_pending.values() if p > 0)

        metrics = {
            "gen/s": round(samples_per_sec, 1),
            "resp_len": round(avg_resp_len, 0),
            "pending": pending,
            "active": active_workers,
            "stale": self.stats["samples_dropped_stale"],
            "skipped": self.stats["batches_skipped_zero_grad"],
        }

        # Add worker metrics if provided
        if worker_metrics:
            trajs = worker_metrics.get("trajs", 0)
            pos = worker_metrics.get("pos_rewards", 0)
            filtered_long = worker_metrics.get("filtered_long", 0)
            metrics["trajs"] = trajs
            metrics["reward%"] = round(100 * pos / trajs, 1) if trajs > 0 else 0.0
            if filtered_long > 0:
                metrics["dropped"] = filtered_long

        return metrics


def compute_rl_loss(
    model: torch.nn.Module,
    batch: Minibatch,
    tokenizer: Any,
    device: str = "cuda",
    algorithm: str = "grpo",
    clip_eps: float = 0.2,
    kl_coef: float = 0.1,
    clip_range_ratio: list[float] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute RL loss for a minibatch.

    Supports: grpo, ppo, gspo

    Args:
        model: Policy model
        batch: Minibatch
        tokenizer: Tokenizer
        device: Device
        algorithm: "grpo", "ppo", or "gspo"
        clip_eps: Clipping epsilon
        kl_coef: KL penalty coefficient
        clip_range_ratio: Optional list [min, max] to override clip_eps

    Returns:
        loss, metrics dict
    """
    # Prepare tensors
    max_len = max(len(s.prompt_ids) + len(s.completion_ids) for s in batch.samples)
    pad_id = tokenizer.pad_token_id

    all_input_ids = []
    all_labels = []
    all_advantages = []

    for s in batch.samples:
        input_ids = s.prompt_ids + s.completion_ids
        labels = [-100] * len(s.prompt_ids) + s.completion_ids

        pad_len = max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_advantages.append(s.advantage)

    input_ids = torch.tensor(all_input_ids, device=device)
    labels = torch.tensor(all_labels, device=device)
    advantages = torch.tensor(all_advantages, device=device, dtype=torch.float32)

    # Forward pass
    outputs = model(input_ids)
    logits = outputs.logits

    # Per-token log probs
    log_probs = F.log_softmax(logits, dim=-1)
    shift_logprobs = log_probs[:, :-1, :]
    shift_labels = labels[:, 1:]

    batch_size, seq_len = shift_labels.shape
    policy_logprobs = shift_logprobs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)

    mask = shift_labels != -100

    # Build old policy logprobs tensor (for ratio / trust region)
    # and reference logprobs tensor (for KL penalty)
    old_logprobs_tensor = torch.zeros_like(policy_logprobs)
    ref_logprobs_tensor = torch.zeros_like(policy_logprobs)
    for i, s in enumerate(batch.samples):
        start = len(s.prompt_ids) - 1
        n_completion = len(s.ref_logprobs)
        end = start + n_completion
        if end <= seq_len and n_completion > 0:
            ref_logprobs_tensor[i, start:end] = torch.tensor(
                s.ref_logprobs, device=device, dtype=torch.float32
            )
            # Use old_logprobs for ratio if available, else fall back to ref_logprobs
            old_lp = s.old_logprobs if s.old_logprobs is not None else s.ref_logprobs
            old_logprobs_tensor[i, start:end] = torch.tensor(
                old_lp, device=device, dtype=torch.float32
            )

    # Compute ratio: π_θ / π_old (current policy vs. policy at generation time)
    ratio = torch.exp(policy_logprobs - old_logprobs_tensor)

    # Algorithm-specific loss
    adv_expanded = advantages.unsqueeze(1).expand(-1, seq_len)

    if algorithm == "gspo":
        # GSPO: soft clipping with tanh
        log_ratio = policy_logprobs - old_logprobs_tensor
        soft_clip = torch.tanh(log_ratio / clip_eps) * clip_eps
        policy_loss = -adv_expanded * soft_clip
    else:
        # PPO / GRPO: standard PPO-style clipping
        if clip_range_ratio is not None:
            clipped_ratio = torch.clamp(ratio, clip_range_ratio[0], clip_range_ratio[1])
        else:
            clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        loss1 = -adv_expanded * ratio
        loss2 = -adv_expanded * clipped_ratio
        policy_loss = torch.max(loss1, loss2)

    # KL penalty against reference model (NOT old policy)
    kl = policy_logprobs - ref_logprobs_tensor
    total_loss = policy_loss + kl_coef * kl

    # Per-sample masked mean, then average across samples
    # Each sample gets equal weight regardless of completion length
    per_sample_loss = (total_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    masked_loss = per_sample_loss.mean()

    # Compute entropy of policy distribution (measures exploration)
    probs = F.softmax(logits[:, :-1, :], dim=-1)
    entropy = -(probs * log_probs[:, :-1, :]).sum(dim=-1)  # [B, S]
    per_sample_entropy = (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    masked_entropy = per_sample_entropy.mean()

    # Compute clipping statistics
    if algorithm in ("ppo", "grpo"):
        clipped = (ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)
        clip_fraction = (clipped * mask).sum().item() / mask.sum().clamp(min=1).item()
    else:
        clip_fraction = 0.0

    # Response length statistics
    response_lengths = [len(s.completion_ids) for s in batch.samples]
    mean_response_len = sum(response_lengths) / len(response_lengths)

    # Per-sample metrics for consistency
    per_sample_policy_loss = (policy_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    per_sample_kl = (kl * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    per_sample_ratio = (ratio * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    metrics = {
        "loss": masked_loss.item(),
        "policy_loss": per_sample_policy_loss.mean().item(),
        "kl": per_sample_kl.mean().item(),
        "entropy": masked_entropy.item(),
        "mean_advantage": advantages.mean().item(),
        "mean_reward": sum(s.reward for s in batch.samples) / len(batch.samples),
        "response_len": mean_response_len,
        "ratio_mean": per_sample_ratio.mean().item(),
        "clip_frac": clip_fraction,
    }

    return masked_loss, metrics


async def train_async_rl(
    config: TechneConfig,
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: Any,
    reward_fn_class: type | None = None,
    algorithm: str = "grpo",
    agent_class: Any | None = None,
    **kwargs,
) -> dict:
    """Train with async RL using Ray.

    All parameters are read from config:
    - config.training.num_inference_workers
    - config.training.num_training_workers
    - config.training.distributed_backend
    - config.training.num_generations
    - config.training.batch_size
    - config.training.gradient_accumulation_steps
    - config.training.clip_eps
    - config.training.kl_coef
    - config.training.sync_weights_interval

    Args:
        config: Techne config
        model: Model to train (used for single-GPU mode)
        tokenizer: Tokenizer
        dataset: HF dataset with "prompt" column
        reward_fn_class: Reward function class (instantiated in workers)
        algorithm: "grpo", "ppo", "gspo", or "distill"

    Returns:
        Training metrics dict
    """
    # Read all params from config
    num_inference_workers = config.training.num_inference_workers
    num_training_workers = config.training.num_training_workers
    distributed_backend = config.training.distributed_backend
    tp_size = config.training.tensor_parallel_size
    num_generations = config.training.num_generations
    per_gpu_batch_size = config.training.batch_size
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    clip_eps = config.training.clip_eps
    clip_range_ratio = config.training.clip_range_ratio
    kl_coef = config.training.kl_coef
    ppo_epochs = config.training.ppo_epochs
    sync_weights_interval = config.training.sync_weights_interval

    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            _metrics_export_port=None,
            configure_logging=False,
            log_to_driver=False,
            _system_config={"metrics_report_interval_ms": 0},
        )

    from techne.training.distributed import detect_device

    device = detect_device(model)

    # Data-parallel distributed training (FSDP) uses multiple Ray training
    # workers. TP (tensor parallelism) runs in a single process with device_map.
    use_distributed = (
        num_training_workers > 1
        and distributed_backend == DistributedBackend.FSDP
    )

    # For distributed: each GPU gets per_gpu_batch_size, total minibatch = per_gpu * num_workers
    effective_minibatch_size = (
        per_gpu_batch_size * num_training_workers if use_distributed else per_gpu_batch_size
    )
    ppo_batch_size = config.training.ppo_batch_size or effective_minibatch_size

    # Validate ppo_batch_size >= effective_minibatch_size
    if ppo_batch_size < effective_minibatch_size:
        raise ValueError(
            f"ppo_batch_size ({ppo_batch_size}) must be >= effective minibatch size "
            f"({effective_minibatch_size} = batch_size {per_gpu_batch_size} × "
            f"{num_training_workers} workers). "
            f"Increase ppo_batch_size or decrease batch_size/num_training_workers."
        )

    # Calculate GPU resources
    num_gpus = torch.cuda.device_count()
    hybrid_mode = tp_size > 1 and use_distributed
    if num_gpus > 0:
        if hybrid_mode:
            # Hybrid TP+DP: each training worker gets tp_size full GPUs,
            # inference workers share the remaining GPUs.
            total_training_gpus = num_training_workers * tp_size
            remaining_gpus = max(1, num_gpus - total_training_gpus)
            gpu_per_inference_worker = remaining_gpus / max(1, num_inference_workers)
            gpu_per_training_worker = tp_size
        else:
            # Standard: fractional GPU allocation across all workers.
            total_workers = num_inference_workers + (num_training_workers if use_distributed else 0)
            gpu_per_worker = 1.0 / (total_workers + 1)  # +1 buffer for main process
            if gpu_per_worker < 0.1:
                gpu_per_worker = 0.1
            gpu_per_inference_worker = gpu_per_worker
            gpu_per_training_worker = gpu_per_worker
    else:
        gpu_per_inference_worker = 0
        gpu_per_training_worker = 0

    # Create inference workers
    inference_config = config.get_inference_config(device=device)
    inference_workers = [
        InferenceWorker.options(num_gpus=gpu_per_inference_worker).remote(
            model_factory=inference_config.create_model_factory(for_training=False),
            num_generations=num_generations,
            max_new_tokens=inference_config.max_new_tokens,
            max_seq_length=config.training.max_seq_length,
            temperature=inference_config.temperature,
            agent_class=agent_class,
            agent_config=config,
            reward_fn_class=reward_fn_class,
        )
        for _ in range(num_inference_workers)
    ]

    # Create training workers for distributed training
    training_workers = None
    if use_distributed:
        training_workers = [
            TrainingWorker.options(num_gpus=gpu_per_training_worker).remote(
                inference_config=inference_config,
                rank=i,
                world_size=num_training_workers,
                distributed_backend=distributed_backend,
                training_config=config.training,
                tensor_parallel_size=tp_size,
            )
            for i in range(num_training_workers)
        ]
        if hybrid_mode:
            logger.info(
                f"Hybrid TP+DP training: {num_training_workers} workers x {tp_size} GPUs/worker "
                f"with {distributed_backend.value.upper()}"
            )
        else:
            logger.info(
                f"Distributed training: {num_training_workers} workers with {distributed_backend.value.upper()}"
            )
    else:
        optimizer = create_optimizer(model.parameters(), config.training)

    global_step = 0
    grad_accum_count = 0
    total_metrics: dict[str, float] = {}
    max_steps = (
        config.training.max_steps
        if config.training.max_steps > 0
        else len(dataset) // effective_minibatch_size
    )

    # Learning rate scheduler
    if not use_distributed:
        scheduler = create_scheduler(optimizer, config.training, max_steps)

    # Get num_prompts_per_dispatch from config or use default
    num_prompts_per_dispatch = getattr(config.training, "num_prompts_per_dispatch", 1)

    logger.info(
        f"Async {algorithm.upper()} (dynamic batching): {num_inference_workers} inference workers, "
        f"{num_generations} generations/prompt, {num_prompts_per_dispatch} prompts/dispatch"
    )

    # Create dynamic batcher for continuous dispatch and result caching
    dynamic_batcher = DynamicBatcher(
        workers=inference_workers,
        dataset=dataset,
        minibatch_size=effective_minibatch_size,
        max_pending_per_worker=2,
        num_prompts_per_dispatch=num_prompts_per_dispatch,
    )

    # Start continuous dispatch to workers
    dynamic_batcher.start()

    pbar = tqdm(total=max_steps, initial=global_step, desc=f"Training {algorithm.upper()}")

    if not use_distributed:
        model.train()

    # Buffer for collecting samples when ppo_batch_size > effective_minibatch_size
    ppo_buffer: list[Sample] = []

    try:
        while global_step < max_steps:
            # Get batch with staleness filtering and active sampling filter
            batch_samples = await dynamic_batcher.get_batch(
                timeout=1.0,
                current_policy_version=dynamic_batcher.policy_version,
                skip_zero_gradient=True,
            )

            if batch_samples is None:
                if dynamic_batcher.is_exhausted:
                    logger.info("Dataset exhausted, stopping training")
                    break
                # Update progress bar while waiting (shows skipped batches, generation progress)
                pbar.set_postfix(dynamic_batcher.get_inference_metrics())
                continue

            # Collect samples into PPO buffer
            ppo_buffer.extend(batch_samples)

            # Wait until we have enough samples for a full PPO batch
            if len(ppo_buffer) < ppo_batch_size:
                # Update progress bar while collecting samples
                pbar.set_postfix(dynamic_batcher.get_inference_metrics())
                continue

            # Extract ppo_batch_size samples for training
            training_samples = ppo_buffer[:ppo_batch_size]
            ppo_buffer = ppo_buffer[ppo_batch_size:]

            # PPO epochs: iterate over the collected batch multiple times
            for _epoch in range(ppo_epochs):
                # Shuffle samples at the start of each epoch for better training
                shuffled_samples = training_samples.copy()
                random.shuffle(shuffled_samples)

                # Process in minibatches (effective_minibatch_size samples per step)
                for mb_start in range(0, len(shuffled_samples), effective_minibatch_size):
                    mb_samples = shuffled_samples[mb_start : mb_start + effective_minibatch_size]
                    if len(mb_samples) < effective_minibatch_size:
                        # Skip incomplete minibatch at the end
                        continue

                    if use_distributed:
                        # Split samples across workers: each gets per_gpu_batch_size
                        futures = []
                        for worker_idx, worker in enumerate(training_workers):
                            worker_start = worker_idx * per_gpu_batch_size
                            worker_end = worker_start + per_gpu_batch_size
                            worker_samples = mb_samples[worker_start:worker_end]
                            worker_batch = Minibatch(samples=worker_samples)
                            futures.append(
                                worker.train_step.remote(
                                    worker_batch,
                                    None,
                                    algorithm,
                                    clip_eps,
                                    kl_coef,
                                    config.training.learning_rate,
                                    config.training.weight_decay,
                                    clip_range_ratio,
                                )
                            )
                        results = await asyncio.gather(
                            *[asyncio.wrap_future(f.future()) for f in futures]
                        )
                        # Average metrics across workers
                        metrics = {}
                        for worker_metrics, _ in results:
                            for k, v in worker_metrics.items():
                                metrics[k] = metrics.get(k, 0) + v / len(results)

                        if hybrid_mode:
                            # Hybrid TP+DP: manual gradient allreduce across workers
                            # (FSDP wrapping is incompatible with device_map="auto" TP models)
                            grad_futures = [w.get_gradients.remote() for w in training_workers]
                            all_grads = await asyncio.gather(
                                *[asyncio.wrap_future(f.future()) for f in grad_futures]
                            )
                            avg_grads = {}
                            for name in all_grads[0]:
                                avg_grads[name] = sum(g[name] for g in all_grads) / len(all_grads)
                            set_futures = [
                                w.set_averaged_gradients.remote(avg_grads)
                                for w in training_workers
                            ]
                            await asyncio.gather(
                                *[asyncio.wrap_future(f.future()) for f in set_futures]
                            )
                    else:
                        batch = Minibatch(samples=mb_samples)
                        loss, metrics = compute_rl_loss(
                            model=model,
                            batch=batch,
                            tokenizer=tokenizer,
                            device=device,
                            algorithm=algorithm,
                            clip_eps=clip_eps,
                            kl_coef=kl_coef,
                            clip_range_ratio=clip_range_ratio,
                        )
                        scaled_loss = loss / gradient_accumulation_steps
                        scaled_loss.backward()

                    for k, v in metrics.items():
                        total_metrics[k] = total_metrics.get(k, 0) + v

                    grad_accum_count += 1

                    if grad_accum_count >= gradient_accumulation_steps:
                        if use_distributed:
                            futures = [
                                worker.optimizer_step.remote(config.training.max_grad_norm)
                                for worker in training_workers
                            ]
                            await asyncio.gather(
                                *[asyncio.wrap_future(f.future()) for f in futures]
                            )
                        else:
                            # Compute gradient norm before clipping
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), config.training.max_grad_norm
                            )
                            total_metrics["grad_norm"] = total_metrics.get("grad_norm", 0) + grad_norm.item()
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()

                        global_step += 1
                        pbar.update(1)
                        grad_accum_count = 0

                        # Update progress bar with combined training + inference metrics
                        # Fetch worker metrics periodically (every logging_steps)
                        worker_metrics = None
                        if global_step % config.logging_steps == 0:
                            worker_metrics = await dynamic_batcher.fetch_worker_metrics()

                        inference_metrics = dynamic_batcher.get_inference_metrics(worker_metrics)
                        if global_step % config.logging_steps == 0:
                            avg_metrics = {
                                k: v / config.logging_steps for k, v in total_metrics.items()
                            }
                            # Add current learning rate
                            if not use_distributed:
                                avg_metrics["lr"] = scheduler.get_last_lr()[0]
                            # Combine training and inference metrics
                            combined_metrics = {**avg_metrics, **inference_metrics}
                            pbar.set_postfix(combined_metrics)
                            total_metrics = {}
                        else:
                            # Still show inference metrics between logging steps
                            pbar.set_postfix(inference_metrics)

                        if (
                            config.training.sync_weights
                            and global_step % sync_weights_interval == 0
                        ):
                            if use_distributed:
                                state_dict = await asyncio.wrap_future(
                                    training_workers[0].get_state_dict.remote().future()
                                )
                            else:
                                # Use helper to merge LoRA weights if present
                                state_dict = get_merged_state_dict(model)
                            for worker in inference_workers:
                                worker.update_weights.remote(state_dict)
                            # Increment policy version after weight sync
                            await dynamic_batcher.increment_policy_version()

                        if global_step % config.save_steps == 0:
                            checkpoint_path = f"{config.output_dir}/checkpoint-{global_step}"
                            if not use_distributed:
                                model.save_pretrained(checkpoint_path)
                            tokenizer.save_pretrained(checkpoint_path)
                            logger.info(f"Saved checkpoint to {checkpoint_path}")

                        # Check if we've reached max_steps
                        if global_step >= max_steps:
                            break

                # Break out of epoch loop if max_steps reached
                if global_step >= max_steps:
                    break

    finally:
        await dynamic_batcher.stop()
        pbar.close()

    # Cleanup distributed workers
    if use_distributed:
        for worker in training_workers:
            worker.cleanup.remote()

    ray.shutdown()

    return {
        "global_step": global_step,
        "final_metrics": total_metrics,
        "batcher_stats": dynamic_batcher.stats,
    }
