"""Async RL training using Ray.

Architecture:
- InferenceWorker: Generates completions + computes ref logprobs (HuggingFace)
- ExperienceBatcher: Collects samples from workers, dispatches to trainer
- Trainer: Pulls minibatches, computes loss, updates model

Supports: GRPO, PPO, GSPO
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

import ray
import torch
import torch.nn.functional as F  # noqa: N812
from transformers import AutoModelForCausalLM, AutoTokenizer

from techne.config import DistributedBackend, TechneConfig
from techne.training.model import LocalModel
from tqdm.auto import tqdm


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
        ray.init()

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
        print(
            f"Distributed training: {num_training_workers} workers with {distributed_backend.value.upper()}"
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    batcher = ExperienceBatcher.remote(minibatch_size=minibatch_size)

    prompt_idx = 0
    batch_size_per_worker = max(1, minibatch_size // (num_inference_workers * num_generations))

    global_step = 0
    grad_accum_count = 0
    total_metrics: dict[str, float] = {}
    max_steps = (
        config.training.max_steps
        if config.training.max_steps > 0
        else len(dataset) // minibatch_size
    )

    print(
        f"Async {algorithm.upper()}: {num_inference_workers} inference workers, {num_generations} generations/prompt"
    )

    async def generate_experience():
        nonlocal prompt_idx

        while prompt_idx < len(dataset) and global_step < max_steps:
            worker_samples = []
            for _ in range(num_inference_workers):
                batch_samples = []
                for _ in range(batch_size_per_worker):
                    if prompt_idx < len(dataset):
                        batch_samples.append(dataset[prompt_idx])
                        prompt_idx += 1
                worker_samples.append(batch_samples)

            futures = [
                worker.generate_and_score.remote(ws, reward_fn)
                for worker, ws in zip(inference_workers, worker_samples)
                if ws
            ]

            for future in futures:
                samples = await asyncio.wrap_future(future.future())
                await asyncio.wrap_future(batcher.add_samples.remote(samples).future())

            await asyncio.sleep(0.01)

    async def train_loop():
        nonlocal global_step, grad_accum_count, total_metrics

        pbar = tqdm(total=max_steps, initial=global_step, desc=f"Training {algorithm.upper()}")

        if not use_distributed:
            model.train()

        while global_step < max_steps:
            batch = await asyncio.wrap_future(batcher.get_batch.remote(timeout=0.5).future())

            if batch is None:
                await asyncio.sleep(0.1)
                continue

            if use_distributed:
                # Distributed training: send batch to all workers
                futures = [
                    worker.train_step.remote(
                        batch,
                        None,  # optimizer state managed internally
                        algorithm,
                        clip_eps,
                        kl_coef,
                        config.training.learning_rate,
                        config.training.weight_decay,
                    )
                    for worker in training_workers
                ]
                results = await asyncio.gather(*[asyncio.wrap_future(f.future()) for f in futures])
                metrics = results[0][0]  # Take metrics from rank 0
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

                if global_step % config.logging_steps == 0:
                    avg_metrics = {k: v / config.logging_steps for k, v in total_metrics.items()}
                    pbar.set_postfix(avg_metrics)
                    total_metrics = {}

                if config.training.sync_weights and global_step % sync_weights_interval == 0:
                    if use_distributed:
                        state_dict = await asyncio.wrap_future(
                            training_workers[0].get_state_dict.remote().future()
                        )
                    else:
                        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                    for worker in inference_workers:
                        worker.update_weights.remote(state_dict)

                if global_step % config.save_steps == 0:
                    checkpoint_path = f"{config.output_dir}/checkpoint-{global_step}"
                    if not use_distributed:
                        model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

        pbar.close()

    await asyncio.gather(generate_experience(), train_loop())

    # Cleanup distributed workers
    if use_distributed:
        for worker in training_workers:
            worker.cleanup.remote()

    ray.shutdown()

    return {"global_step": global_step, "final_metrics": total_metrics}
