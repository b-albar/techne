"""Masking utilities for training."""

from __future__ import annotations

import torch

from techne.rollout.parser import TagParser


def create_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Create attention mask from input IDs.

    Args:
        input_ids: Token IDs tensor
        pad_token_id: Padding token ID

    Returns:
        Attention mask (1 for real tokens, 0 for padding)
    """
    return (input_ids != pad_token_id).long()


def create_label_mask(
    input_ids: torch.Tensor,
    tokenizer,
    parser: TagParser,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Create label mask that ignores interpreter responses.

    Tokens in <interpreter>...</interpreter> blocks are set to ignore_index
    so they don't contribute to the loss.

    Args:
        input_ids: Token IDs tensor [batch, seq_len] or [seq_len]
        tokenizer: Tokenizer for decoding
        parser: TagParser for finding response regions
        ignore_index: Value to use for masked tokens (default: -100)

    Returns:
        Labels tensor with masked regions set to ignore_index
    """
    # Handle both batched and single inputs
    is_batched = input_ids.dim() == 2
    if not is_batched:
        input_ids = input_ids.unsqueeze(0)

    batch_size, seq_len = input_ids.shape
    labels = input_ids.clone()

    for b in range(batch_size):
        # Decode to text
        text = tokenizer.decode(input_ids[b])

        # Get mask ranges in character space
        mask_ranges = parser.get_response_mask_ranges(text)

        if not mask_ranges:
            continue

        # Convert character ranges to token ranges
        # Use offset mapping if available
        encoding = tokenizer(text, return_offsets_mapping=True)
        offsets = encoding.get("offset_mapping", [])

        if not offsets:
            continue

        for token_idx, (char_start, char_end) in enumerate(offsets):
            if token_idx >= seq_len:
                break

            # Check if this token overlaps with any masked range
            for mask_start, mask_end in mask_ranges:
                if char_start < mask_end and char_end > mask_start:
                    labels[b, token_idx] = ignore_index
                    break

    if not is_batched:
        labels = labels.squeeze(0)

    return labels


def create_prompt_mask(
    input_ids: torch.Tensor,
    prompt_lengths: list[int],
    ignore_index: int = -100,
) -> torch.Tensor:
    """Create label mask that ignores prompt tokens.

    Only response tokens should contribute to the loss.

    Args:
        input_ids: Token IDs tensor
        prompt_lengths: Length of prompt for each example
        ignore_index: Value for masked tokens

    Returns:
        Labels tensor with prompts masked
    """
    is_batched = input_ids.dim() == 2
    if not is_batched:
        input_ids = input_ids.unsqueeze(0)
        prompt_lengths = [prompt_lengths]

    labels = input_ids.clone()

    for b, prompt_len in enumerate(prompt_lengths):
        labels[b, :prompt_len] = ignore_index

    if not is_batched:
        labels = labels.squeeze(0)

    return labels


def combine_masks(
    labels: torch.Tensor,
    *additional_masks: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Combine multiple masks by taking union of masked regions.

    Args:
        labels: Base labels tensor
        *additional_masks: Additional mask tensors (True = mask out)
        ignore_index: Value for masked positions

    Returns:
        Labels with all masks applied
    """
    result = labels.clone()

    for mask in additional_masks:
        result = torch.where(mask, torch.tensor(ignore_index), result)

    return result
