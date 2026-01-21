"""Distributed data generation using Ray.

Generate SFT and distillation training data in parallel using Ray workers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import ray
import torch

from techne.training.model import LocalModel


@dataclass
class GeneratedSample:
    """A generated sample for SFT or distillation."""

    prompt: str
    completion: str
    prompt_ids: list[int]
    completion_ids: list[int]
    # For distillation: teacher logprobs on completion
    teacher_logprobs: list[float] | None = None
    # Optional metadata
    metadata: dict | None = None


@ray.remote
class GenerationWorker:
    """Worker that generates completions for SFT/distillation data."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.model = LocalModel.from_pretrained(
            model_name,
            device=device,
            dtype=dtype,
        )
        self.tokenizer = self.model.tokenizer

    def generate(self, prompts: list[str]) -> list[GeneratedSample]:
        """Generate completions for prompts."""
        samples = []

        for prompt in prompts:
            prompt_enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_ids = prompt_enc.input_ids[0].tolist()

            with torch.no_grad():
                outputs = self.model.generate(
                    **prompt_enc,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            completion_ids = outputs[0][len(prompt_ids) :].tolist()
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            samples.append(
                GeneratedSample(
                    prompt=prompt,
                    completion=completion,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                )
            )

        return samples


@ray.remote
class TeacherWorker:
    """Worker that computes teacher logprobs for distillation."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.model = LocalModel.from_pretrained(
            model_name,
            device=device,
            dtype=dtype,
        )
        self.tokenizer = self.model.tokenizer

    def compute_logprobs(self, samples: list[GeneratedSample]) -> list[GeneratedSample]:
        """Compute teacher logprobs for completions."""
        for sample in samples:
            sample.teacher_logprobs = self.model.compute_logprobs(
                sample.prompt_ids, sample.completion_ids
            )
        return samples


async def generate_sft_data(
    prompts: list[str],
    model_name: str,
    num_workers: int = 1,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    dtype: torch.dtype = torch.bfloat16,
    filter_fn: Callable[[GeneratedSample], bool] | None = None,
) -> list[GeneratedSample]:
    """Generate SFT training data using Ray workers.

    Args:
        prompts: List of prompts to generate completions for
        model_name: HuggingFace model name/path
        num_workers: Number of generation workers
        batch_size: Prompts per worker batch
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        dtype: Model dtype
        filter_fn: Optional filter function to keep only good samples

    Returns:
        List of generated samples
    """
    if not ray.is_initialized():
        ray.init()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    workers = [
        GenerationWorker.remote(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for _ in range(num_workers)
    ]

    all_samples: list[GeneratedSample] = []
    prompt_idx = 0

    while prompt_idx < len(prompts):
        # Distribute prompts to workers
        futures = []
        for worker in workers:
            batch_prompts = prompts[prompt_idx : prompt_idx + batch_size]
            if batch_prompts:
                futures.append(worker.generate.remote(batch_prompts))
                prompt_idx += len(batch_prompts)

        # Collect results
        for future in futures:
            samples = await asyncio.wrap_future(future.future())
            if filter_fn:
                samples = [s for s in samples if filter_fn(s)]
            all_samples.extend(samples)

    ray.shutdown()
    return all_samples


async def generate_distill_data(
    prompts: list[str],
    student_model_name: str,
    teacher_model_name: str,
    num_gen_workers: int = 1,
    num_teacher_workers: int = 1,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    dtype: torch.dtype = torch.bfloat16,
    filter_fn: Callable[[GeneratedSample], bool] | None = None,
) -> list[GeneratedSample]:
    """Generate distillation training data using Ray workers.

    Student generates completions, teacher computes logprobs.

    Args:
        prompts: List of prompts
        student_model_name: Student model for generation
        teacher_model_name: Teacher model for logprobs
        num_gen_workers: Number of generation workers
        num_teacher_workers: Number of teacher workers
        batch_size: Prompts per worker batch
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        dtype: Model dtype
        filter_fn: Optional filter function

    Returns:
        List of generated samples with teacher logprobs
    """
    if not ray.is_initialized():
        ray.init()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create workers
    gen_workers = [
        GenerationWorker.remote(
            model_name=student_model_name,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for _ in range(num_gen_workers)
    ]

    teacher_workers = [
        TeacherWorker.remote(
            model_name=teacher_model_name,
            device=device,
            dtype=dtype,
        )
        for _ in range(num_teacher_workers)
    ]

    all_samples: list[GeneratedSample] = []
    pending_samples: list[GeneratedSample] = []
    prompt_idx = 0
    teacher_idx = 0

    async def process_with_teacher(samples: list[GeneratedSample]) -> list[GeneratedSample]:
        nonlocal teacher_idx
        worker = teacher_workers[teacher_idx % len(teacher_workers)]
        teacher_idx += 1
        return await asyncio.wrap_future(worker.compute_logprobs.remote(samples).future())

    # Generate and score
    while prompt_idx < len(prompts) or pending_samples:
        # Generate batch
        gen_futures = []
        for worker in gen_workers:
            if prompt_idx < len(prompts):
                batch_prompts = prompts[prompt_idx : prompt_idx + batch_size]
                if batch_prompts:
                    gen_futures.append(worker.generate.remote(batch_prompts))
                    prompt_idx += len(batch_prompts)

        # Collect generated samples
        for future in gen_futures:
            samples = await asyncio.wrap_future(future.future())
            if filter_fn:
                samples = [s for s in samples if filter_fn(s)]
            pending_samples.extend(samples)

        # Process with teacher in batches
        while len(pending_samples) >= batch_size:
            batch = pending_samples[:batch_size]
            pending_samples = pending_samples[batch_size:]
            scored = await process_with_teacher(batch)
            all_samples.extend(scored)

    # Process remaining samples
    if pending_samples:
        scored = await process_with_teacher(pending_samples)
        all_samples.extend(scored)

    ray.shutdown()
    return all_samples


def samples_to_dataset(
    samples: list[GeneratedSample], include_teacher_logprobs: bool = False
) -> dict[str, list]:
    """Convert generated samples to HuggingFace dataset format.

    Args:
        samples: List of generated samples
        include_teacher_logprobs: Include teacher logprobs in output

    Returns:
        Dict suitable for datasets.Dataset.from_dict()
    """
    data = {
        "prompt": [s.prompt for s in samples],
        "completion": [s.completion for s in samples],
        "text": [s.prompt + s.completion for s in samples],
    }

    if include_teacher_logprobs:
        data["teacher_logprobs"] = [s.teacher_logprobs for s in samples]

    return data


# Convenience sync wrappers
def generate_sft_data_sync(
    prompts: list[str],
    model_name: str,
    **kwargs,
) -> list[GeneratedSample]:
    """Sync wrapper for generate_sft_data."""
    return asyncio.run(generate_sft_data(prompts, model_name, **kwargs))


def generate_distill_data_sync(
    prompts: list[str],
    student_model_name: str,
    teacher_model_name: str,
    **kwargs,
) -> list[GeneratedSample]:
    """Sync wrapper for generate_distill_data."""
    return asyncio.run(
        generate_distill_data(prompts, student_model_name, teacher_model_name, **kwargs)
    )
