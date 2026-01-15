"""Tokenizer alignment strategies for distillation.

This module provides abstractions to handle vocabulary and tokenization differences
between student and teacher models during distillation.
"""

import abc

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


class TokenAligner(abc.ABC):
    """Abstract base class for token alignment strategies."""

    def __init__(
        self, student_tokenizer: PreTrainedTokenizer, teacher_tokenizer: PreTrainedTokenizer
    ):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

    @abc.abstractmethod
    def compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_ids: torch.Tensor,
        teacher_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute KL divergence loss with appropriate alignment.

        Args:
            student_logits: [batch, s_seq, s_vocab]
            teacher_logits: [batch, t_seq, t_vocab]
            student_ids: [batch, s_seq]
            teacher_ids: [batch, t_seq]
            temperature: Softmax temperature

        Returns:
            Scalar KL loss
        """
        pass


class DirectAligner(TokenAligner):
    """Aligner for identical tokenizers. Assumes 1-to-1 mapping via truncation."""

    def compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_ids: torch.Tensor,
        teacher_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        # Truncate to min length
        min_len = min(student_logits.shape[1], teacher_logits.shape[1])
        s_logits = student_logits[:, :min_len, :]
        t_logits = teacher_logits[:, :min_len, :]

        # Compute KL
        s_probs = F.log_softmax(s_logits / temperature, dim=-1)
        t_probs = F.softmax(t_logits / temperature, dim=-1)

        # Compute KL [batch, seq] (summed over vocab)
        kl_map = F.kl_div(s_probs, t_probs, reduction="none").sum(-1)

        # Mask padding
        if student_ids is not None and self.student_tokenizer.pad_token_id is not None:
            # Look up pad token
            pad_id = self.student_tokenizer.pad_token_id
            # Ensure mask matches truncated length
            mask = (student_ids[:, :min_len] != pad_id).float()
            kl_map = kl_map * mask
            num_tokens = mask.sum()
        else:
            num_tokens = torch.tensor(kl_map.numel(), device=kl_map.device)

        if num_tokens > 0:
            kl = kl_map.sum() / num_tokens
        else:
            kl = kl_map.sum() * 0.0

        return kl * (temperature**2)


class GoldAligner(TokenAligner):
    """Aligner using GOLD/ULD strategy.

    Uses greedy substring matching to align tokens, then computes L1 loss
    on sorted probability distributions (Universal Logit Distillation approach).
    """

    def compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_ids: torch.Tensor,
        teacher_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        batch_size = student_ids.shape[0]
        batch_losses = []

        # Process each sample in batch
        for i in range(batch_size):
            s_ids_i = student_ids[i].tolist()
            t_ids_i = teacher_ids[i].tolist()

            s_ids_i = [t for t in s_ids_i if t != self.student_tokenizer.pad_token_id]
            t_ids_i = [t for t in t_ids_i if t != self.teacher_tokenizer.pad_token_id]

            # 1. Build Alignment Groups
            s_groups, t_groups = self._build_alignment_groups(s_ids_i, t_ids_i)

            if not s_groups:
                batch_losses.append(torch.tensor(0.0, device=student_logits.device))
                continue

            # 2. Get Probs
            s_probs = F.softmax(student_logits[i, : len(s_ids_i)] / temperature, dim=-1)
            t_probs = F.softmax(teacher_logits[i, : len(t_ids_i)] / temperature, dim=-1)

            # 3. Iterative Loss Computation (Save Memory)
            sample_loss = 0.0
            valid_groups = 0

            # Common max vocab for padding
            max_v = max(s_probs.size(-1), t_probs.size(-1))

            for g_idx in range(len(s_groups)):
                s_grp = s_groups[g_idx]
                t_grp = t_groups[g_idx]

                # Compute merged prob for single group [vocab]
                s_p = self._compute_group_prob(s_probs, s_grp, s_ids_i)
                t_p = self._compute_group_prob(t_probs, t_grp, t_ids_i)

                # Sort descending (ULD)
                s_sorted, _ = torch.sort(s_p, descending=True)
                t_sorted, _ = torch.sort(t_p, descending=True)

                # Pad if needed
                if s_sorted.size(0) < max_v:
                    s_sorted = F.pad(s_sorted, (0, max_v - s_sorted.size(0)))
                if t_sorted.size(0) < max_v:
                    t_sorted = F.pad(t_sorted, (0, max_v - t_sorted.size(0)))

                # L1 Loss
                # Note: gradients flow through s_p -> s_probs -> logits
                loss_val = F.l1_loss(s_sorted, t_sorted, reduction="mean")
                sample_loss += loss_val
                valid_groups += 1

            if valid_groups > 0:
                batch_losses.append(sample_loss / valid_groups)
            else:
                batch_losses.append(torch.tensor(0.0, device=student_logits.device))

        return torch.stack(batch_losses).mean() * (temperature**2)

    def _compute_group_prob(self, probs, group, ids):
        """Compute merged probability for a single group."""
        if not group:
            return torch.zeros(probs.size(-1), device=probs.device, dtype=probs.dtype)

        # P(group) = P(tok0) * P(tok1|tok0) * ...
        first_pos = group[0]
        marginal_probs = probs[first_pos]

        if len(group) == 1:
            return marginal_probs

        eps = 1e-8
        conditional_prob_product = 1.0
        for idx in group[1:]:
            actual_token_id = ids[idx]
            token_prob = probs[idx, actual_token_id].clamp_min(eps)
            conditional_prob_product *= token_prob

        return marginal_probs * conditional_prob_product

    def _build_alignment_groups(self, s_ids, t_ids):
        """Build alignment groups using greedy substring equality."""

        def to_canonical_pieces(tok, ids):
            pieces = []
            prev = ""
            for k in range(len(ids)):
                cur = tok.decode(
                    ids[: k + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                pieces.append(cur[len(prev) :])
                prev = cur
            return pieces

        s_pieces = to_canonical_pieces(self.student_tokenizer, s_ids)
        t_pieces = to_canonical_pieces(self.teacher_tokenizer, t_ids)

        i = j = 0
        s_buf = t_buf = ""
        s_group, t_group = [], []
        s_groups, t_groups = [], []

        def flush():
            if s_group and t_group:
                s_groups.append(list(s_group))
                t_groups.append(list(t_group))

        while i < len(s_pieces) or j < len(t_pieces):
            if s_buf == t_buf and s_buf != "":
                flush()
                s_buf, t_buf = "", ""
                s_group, t_group = [], []
                continue

            # Greedy advancement
            if s_buf == "" and i < len(s_pieces):
                s_buf += s_pieces[i]
                s_group.append(i)
                i += 1
                continue
            if t_buf == "" and j < len(t_pieces):
                t_buf += t_pieces[j]
                t_group.append(j)
                j += 1
                continue

            if len(s_buf) <= len(t_buf):
                if i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1
                elif j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
            else:
                if j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
                elif i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1

        if s_buf == t_buf and s_group and t_group:
            flush()
        # Handle remainders (force flush)
        if s_group or t_group:
            s_groups.append(list(s_group))
            t_groups.append(list(t_group))

        return s_groups, t_groups
