"""Async RL training using Ray.

Architecture:
- InferenceWorker: Generates completions + computes ref logprobs (HuggingFace)
- ExperienceBatcher: Collects samples from workers, dispatches to trainer
- Trainer: Pulls minibatches, computes loss, updates model

Supports: GRPO, PPO, GSPO
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Queue
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
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from techne.config import DistributedBackend, TechneConfig  # noqa: E402
from techne.training.model import LocalModel  # noqa: E402


@dataclass
class Sample:
    """A single generated sample."""

    prompt: str
    completion: str
    prompt_ids: list[int]
    completion_ids: list[int]
    ref_logprobs: list[float]
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
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_generations: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        agent_class: Any | None = None,
        agent_config: TechneConfig | None = None,
    ):
        self.device = device
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.agent_class = agent_class
        self.agent_config = agent_config

        self.model = LocalModel.from_pretrained(
            model_name,
            device=device,
            dtype=dtype,
        )
        self.tokenizer = self.model.tokenizer

        # Initialize agent if class provided
        if self.agent_class:
            self.agent = self.agent_class(
                self.agent_config, model=self.model, tokenizer=self.tokenizer
            )
        else:
            self.agent = None

    async def generate_and_score(
        self,
        samples_meta: list[dict],
        reward_fn: Callable[[dict, str], float] | None = None,
    ) -> list[Sample]:
        """Generate completions and compute reference logprobs."""
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

                    ref_logprobs = self.model.compute_logprobs(prompt_ids, completion_ids)

                    reward = 0.0
                    if reward_fn:
                        try:
                            reward = reward_fn(sample, completion_text)
                        except Exception:
                            reward = 0.0

                    ret_samples.append(
                        Sample(
                            prompt=prompt_content,
                            completion=completion_text,
                            prompt_ids=prompt_ids,
                            completion_ids=completion_ids,
                            ref_logprobs=ref_logprobs,
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
                    completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                    ref_logprobs = self.model.compute_logprobs(prompt_ids, completion_ids)

                    reward = 0.0
                    if reward_fn:
                        try:
                            reward = reward_fn(sample, completion)
                        except Exception:
                            reward = 0.0

                    ret_samples.append(
                        Sample(
                            prompt=prompt_content,
                            completion=completion,
                            prompt_ids=prompt_ids,
                            completion_ids=completion_ids,
                            ref_logprobs=ref_logprobs,
                            reward=reward,
                        )
                    )

        return ret_samples

    def update_weights(self, state_dict: dict):
        """Update model weights from trainer."""
        self.model.load_state_dict(state_dict)


@ray.remote
class TrainingWorker:
    """Distributed training worker with FSDP/DDP support."""

    def __init__(
        self,
        model_name: str,
        rank: int,
        world_size: int,
        dtype: torch.dtype = torch.bfloat16,
        distributed_backend: DistributedBackend = DistributedBackend.NONE,
    ):
        import os

        self.rank = rank
        self.world_size = world_size
        self.distributed_backend = distributed_backend

        if distributed_backend == DistributedBackend.FSDP:
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817

            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True,
            )
            self.model = FSDP(base_model.cuda())
        elif distributed_backend == DistributedBackend.DDP:
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True,
            ).cuda(rank)
            self.model = torch.nn.parallel.DistributedDataParallel(base_model, device_ids=[rank])
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map="cuda",
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = f"cuda:{rank}" if distributed_backend != DistributedBackend.NONE else "cuda"

    def train_step(
        self,
        batch: Minibatch,
        optimizer_state: dict | None,
        algorithm: str,
        clip_eps: float,
        kl_coef: float,
        lr: float,
        weight_decay: float,
    ) -> tuple[dict, dict]:
        """Execute a training step and return metrics + new optimizer state."""
        if optimizer_state is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
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

    def cleanup(self):
        """Cleanup distributed process group."""
        if self.distributed_backend != DistributedBackend.NONE:
            torch.distributed.destroy_process_group()


def compute_advantages(samples: list[Sample]) -> list[Sample]:
    """Compute advantages for samples grouped by prompt (GRPO style)."""
    prompt_groups: dict[any, list[Sample]] = {}
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
    std = (sum((a - sum(advantages) / len(advantages)) ** 2 for a in advantages) / len(advantages)) ** 0.5
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
        reward_fn: Callable[[dict, str], float] | None,
        minibatch_size: int = 8,
        max_pending_per_worker: int = 2,
        num_prompts_per_dispatch: int = 1,
        staleness_threshold: int = 3,
    ):
        self.workers = workers
        self.dataset = dataset
        self.reward_fn = reward_fn
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
            future = worker.generate_and_score.remote(prompts, self.reward_fn)
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
                self._response_lengths = self._response_lengths[-self._max_response_history:]

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
                    s for s in batch
                    if current_policy_version - s.policy_version <= self.staleness_threshold
                ]
                stale_count = len(batch) - len(fresh_samples)
                if stale_count > 0:
                    self.stats["samples_dropped_stale"] += stale_count
                    # Re-push fresh samples if not enough for a batch
                    if len(fresh_samples) < self.minibatch_size:
                        for s in fresh_samples:
                            await self.cache.push_one(s)
                        continue
                    batch = fresh_samples[:self.minibatch_size]

            # Skip zero-gradient batches (Open-Instruct active sampling filter)
            if skip_zero_gradient and should_skip_batch(batch):
                self.stats["batches_skipped_zero_grad"] += 1
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

    def get_inference_metrics(self) -> dict[str, float]:
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

        return {
            "gen/s": round(samples_per_sec, 1),
            "resp_len": round(avg_resp_len, 0),
            "pending": pending,
            "active": active_workers,
            "stale": self.stats["samples_dropped_stale"],
            "skipped": self.stats["batches_skipped_zero_grad"],
        }


@ray.remote
class ExperienceBatcher:
    """Collects samples and prepares minibatches."""

    def __init__(self, minibatch_size: int = 8):
        self.minibatch_size = minibatch_size
        self.sample_buffer: list[Sample] = []
        self.ready_batches: Queue[Minibatch] = Queue()

    def add_samples(self, samples: list[Sample]):
        """Add samples and compute advantages within groups."""
        # Group by prompt for advantage computation (GRPO style)
        prompt_groups: dict[any, list[Sample]] = {}
        for s in samples:
            # Handle unhashable prompts (e.g. list of messages)
            if isinstance(s.prompt, list):
                # Convert list of dicts to tuple of frozensets (hashable)
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

        self.sample_buffer.extend(samples)
        self._maybe_flush()

    def _maybe_flush(self):
        """Create minibatch if enough samples."""
        while len(self.sample_buffer) >= self.minibatch_size:
            batch_samples = self.sample_buffer[: self.minibatch_size]
            self.sample_buffer = self.sample_buffer[self.minibatch_size :]
            self.ready_batches.put(Minibatch(samples=batch_samples))

    def get_batch(self, timeout: float = 1.0) -> Minibatch | None:
        """Get a ready minibatch."""
        try:
            return self.ready_batches.get(timeout=timeout)
        except Empty:
            return None


def compute_rl_loss(
    model: torch.nn.Module,
    batch: Minibatch,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    algorithm: str = "grpo",
    clip_eps: float = 0.2,
    kl_coef: float = 0.1,
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

    Returns:
        loss, metrics dict
    """
    # Prepare tensors
    max_len = max(len(s.prompt_ids) + len(s.completion_ids) for s in batch.samples)
    pad_id = tokenizer.pad_token_id

    all_input_ids = []
    all_labels = []
    all_ref_logprobs = []
    all_advantages = []

    for s in batch.samples:
        input_ids = s.prompt_ids + s.completion_ids
        labels = [-100] * len(s.prompt_ids) + s.completion_ids

        pad_len = max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_ref_logprobs.append(s.ref_logprobs)
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

    # Reference logprobs tensor
    ref_logprobs_tensor = torch.zeros_like(policy_logprobs)
    for i, s in enumerate(batch.samples):
        start = len(s.prompt_ids) - 1
        end = start + len(s.ref_logprobs)
        if end <= seq_len and len(s.ref_logprobs) > 0:
            ref_logprobs_tensor[i, start:end] = torch.tensor(
                s.ref_logprobs, device=device, dtype=torch.float32
            )

    # Compute ratio
    ratio = torch.exp(policy_logprobs - ref_logprobs_tensor)

    # Algorithm-specific loss
    if algorithm == "ppo":
        # PPO: clip ratio, multiply by advantage
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        adv_expanded = advantages.unsqueeze(1).expand(-1, seq_len)
        loss1 = -adv_expanded * ratio
        loss2 = -adv_expanded * clipped_ratio
        policy_loss = torch.max(loss1, loss2)

    elif algorithm == "gspo":
        # GSPO: soft clipping with tanh
        adv_expanded = advantages.unsqueeze(1).expand(-1, seq_len)
        log_ratio = policy_logprobs - ref_logprobs_tensor
        soft_clip = torch.tanh(log_ratio / clip_eps) * clip_eps
        policy_loss = -adv_expanded * soft_clip

    else:  # grpo (default)
        # GRPO: standard PPO-style clipping
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        adv_expanded = advantages.unsqueeze(1).expand(-1, seq_len)
        loss1 = -adv_expanded * ratio
        loss2 = -adv_expanded * clipped_ratio
        policy_loss = torch.max(loss1, loss2)

    # KL penalty
    kl = policy_logprobs - ref_logprobs_tensor
    total_loss = policy_loss + kl_coef * kl

    # Masked mean
    masked_loss = (total_loss * mask).sum() / mask.sum().clamp(min=1)

    metrics = {
        "loss": masked_loss.item(),
        "policy_loss": (policy_loss * mask).sum().item() / mask.sum().clamp(min=1).item(),
        "kl": (kl * mask).sum().item() / mask.sum().clamp(min=1).item(),
        "mean_advantage": advantages.mean().item(),
        "mean_reward": sum(s.reward for s in batch.samples) / len(batch.samples),
    }

    return masked_loss, metrics


async def train_async_rl(
    config: TechneConfig,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Any,
    reward_fn: Callable[[dict, str], float] | None = None,
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
        reward_fn: Reward function (prompt, completion) -> float
        algorithm: "grpo", "ppo", "gspo", or "distill"

    Returns:
        Training metrics dict
    """
    # Read all params from config
    num_inference_workers = config.training.num_inference_workers
    num_training_workers = config.training.num_training_workers
    distributed_backend = config.training.distributed_backend
    num_generations = config.training.num_generations
    minibatch_size = config.training.batch_size
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    clip_eps = config.training.clip_eps
    kl_coef = config.training.kl_coef
    sync_weights_interval = config.training.sync_weights_interval

    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            _metrics_export_port=None,
            configure_logging=False,
            log_to_driver=False,
            _system_config={"metrics_report_interval_ms": 0},
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_distributed = num_training_workers > 1 and distributed_backend != DistributedBackend.NONE

    # Calculate GPU resources
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        # Simple heuristic: Split GPUs between inference and training
        # If not distributed, training happens in main process (needs GPU), so reserved some implicitly?
        # Actually, Ray doesn't track main process usage.
        # We just need to fit workers.
        # Example: 1 GPU. 1 Inf Worker. Main process training.
        # Main process takes e.g. 50% memory. Inf worker takes rest.
        # We tell Ray to give Inf worker 0.5 GPU.

        total_workers = num_inference_workers + (num_training_workers if use_distributed else 0)
        gpu_per_worker = 1.0 / (total_workers + 1)  # +1 buffer for main process or safety
        # Ensure at least some fractional amount
        if gpu_per_worker < 0.1:
            gpu_per_worker = 0.1
    else:
        gpu_per_worker = 0

    # Create inference workers
    inference_workers = [
        InferenceWorker.options(num_gpus=gpu_per_worker).remote(
            model_name=config.model.name_or_path,
            device=device,
            dtype=config.model.dtype,
            num_generations=num_generations,
            max_new_tokens=config.training.max_seq_length // 2,
            temperature=config.rollout.temperature,
            agent_class=agent_class,
            agent_config=config,
        )
        for _ in range(num_inference_workers)
    ]

    # Create training workers for distributed training
    training_workers = None
    if use_distributed:
        training_workers = [
            TrainingWorker.options(num_gpus=gpu_per_worker).remote(
                model_name=config.model.name_or_path,
                rank=i,
                world_size=num_training_workers,
                dtype=config.model.dtype,
                distributed_backend=distributed_backend,
            )
            for i in range(num_training_workers)
        ]
        logger.info(
            f"Distributed training: {num_training_workers} workers with {distributed_backend.value.upper()}"
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    global_step = 0
    grad_accum_count = 0
    total_metrics: dict[str, float] = {}
    max_steps = (
        config.training.max_steps
        if config.training.max_steps > 0
        else len(dataset) // minibatch_size
    )

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
        reward_fn=reward_fn,
        minibatch_size=minibatch_size,
        max_pending_per_worker=2,
        num_prompts_per_dispatch=num_prompts_per_dispatch,
    )

    # Start continuous dispatch to workers
    dynamic_batcher.start()

    pbar = tqdm(total=max_steps, initial=global_step, desc=f"Training {algorithm.upper()}")

    if not use_distributed:
        model.train()

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
                continue

            batch = Minibatch(samples=batch_samples)

            if use_distributed:
                futures = [
                    worker.train_step.remote(
                        batch,
                        None,
                        algorithm,
                        clip_eps,
                        kl_coef,
                        config.training.learning_rate,
                        config.training.weight_decay,
                    )
                    for worker in training_workers
                ]
                results = await asyncio.gather(*[asyncio.wrap_future(f.future()) for f in futures])
                metrics = results[0][0]
            else:
                loss, metrics = compute_rl_loss(
                    model=model,
                    batch=batch,
                    tokenizer=tokenizer,
                    device=device,
                    algorithm=algorithm,
                    clip_eps=clip_eps,
                    kl_coef=kl_coef,
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
                    await asyncio.gather(*[asyncio.wrap_future(f.future()) for f in futures])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.training.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                global_step += 1
                pbar.update(1)
                grad_accum_count = 0

                # Update progress bar with combined training + inference metrics
                inference_metrics = dynamic_batcher.get_inference_metrics()
                if global_step % config.logging_steps == 0:
                    avg_metrics = {k: v / config.logging_steps for k, v in total_metrics.items()}
                    # Combine training and inference metrics
                    combined_metrics = {**avg_metrics, **inference_metrics}
                    pbar.set_postfix(combined_metrics)
                    total_metrics = {}
                else:
                    # Still show inference metrics between logging steps
                    pbar.set_postfix(inference_metrics)

                if config.training.sync_weights and global_step % sync_weights_interval == 0:
                    if use_distributed:
                        state_dict = await asyncio.wrap_future(
                            training_workers[0].get_state_dict.remote().future()
                        )
                    else:
                        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
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
