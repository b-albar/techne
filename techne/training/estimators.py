from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SparseLogits:
    """Structure to hold sparse logits data (top-k or top-p)."""

    indices: torch.Tensor  # Indices of the top k logits [batch, seq, k]
    values: torch.Tensor  # Values of the top k logits [batch, seq, k]

    def to(self, device):
        self.indices = self.indices.to(device)
        self.values = self.values.to(device)
        return self


class DistillationEstimator(ABC):
    """Abstract base class for distillation loss estimators."""

    @abstractmethod
    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_data: torch.Tensor | SparseLogits,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_logits: Logits from student model [batch, seq, vocab]
            teacher_data: Full logits [batch, seq, vocab] or SparseLogits

        Returns:
            Scalar loss tensor
        """
        pass


class ForwardKLEstimator(DistillationEstimator):
    """Standard Forward KL Divergence: KL(Teacher || Student).
    Minimizes cross-entropy H(T, S) - H(T). H(T) is constant, so ~ CE.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def compute_loss(self, student_logits, teacher_data, **kwargs):
        if isinstance(teacher_data, SparseLogits):
            # Sparse Forward KL not typically approximated this way for full distribution matching
            # but we can try estimating it. SLIM usually does this.
            return self._compute_sparse(student_logits, teacher_data)

        # Dense
        return self._compute_dense(student_logits, teacher_data)

    def _compute_dense(self, s_logits, t_logits):
        s_probs = F.log_softmax(s_logits / self.temperature, dim=-1)
        t_probs = F.softmax(t_logits / self.temperature, dim=-1)
        # batchmean: sum over seq, mean over batch. We want mean over tokens usually.
        # But F.kl_div with 'batchmean' is mathematically correct for KL sum.
        # To match custom normalization we might want 'reduction=none'
        loss = F.kl_div(s_probs, t_probs, reduction="batchmean")
        return loss * (self.temperature**2)

    def _compute_sparse(self, s_logits, t_sparse: SparseLogits):
        # NOT implemented as standard forward KL requires full teacher distribution for proper matching
        # unless we assume tail is 0.
        raise NotImplementedError(
            "Standard Forward KL with sparse logits not fully defined. Use SlimEstimator."
        )


class ReverseKLEstimator(DistillationEstimator):
    """Reverse KL Divergence: KL(Student || Teacher).

    Mode-seeking behavior: student focuses on high-probability regions of teacher.
    Useful when you want the student to be confident on a subset of modes.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def compute_loss(self, student_logits, teacher_data, **kwargs):
        if isinstance(teacher_data, SparseLogits):
            raise NotImplementedError("Reverse KL with sparse logits not supported.")

        # Dense reverse KL: KL(S || T) = sum(S * log(S/T))
        s_logprobs = F.log_softmax(student_logits / self.temperature, dim=-1)
        s_probs = F.softmax(student_logits / self.temperature, dim=-1)
        t_logprobs = F.log_softmax(teacher_data / self.temperature, dim=-1)

        # KL(S || T) = sum(s * (log_s - log_t))
        kl_per_token = (s_probs * (s_logprobs - t_logprobs)).sum(dim=-1)  # [B, S]

        mask = kwargs.get("mask", None)
        if mask is not None:
            kl_per_token = kl_per_token * mask
            return (kl_per_token.sum() / mask.sum().clamp(min=1)) * (self.temperature**2)

        return kl_per_token.mean() * (self.temperature**2)


class AKDEstimator(DistillationEstimator):
    """Adaptive KL Divergence (AKD) from "Rethinking Kullback-Leibler Divergence in KD for LLMs".

    Combines forward and reverse KL with adaptive weighting based on probability mass
    partitioning. High-confidence and low-confidence regions are weighted differently.

    Reference: https://github.com/wutaiqiang/LLM_KD_AKL
    """

    def __init__(self, temperature: float = 1.0, mu: float = 0.5):
        """
        Args:
            temperature: Softmax temperature for smoothing distributions.
            mu: Cumulative probability threshold for partitioning (default 0.5).
        """
        self.temperature = temperature
        self.mu = mu

    def compute_loss(self, student_logits, teacher_data, **kwargs):
        if isinstance(teacher_data, SparseLogits):
            raise NotImplementedError("AKD with sparse logits not supported.")

        t_logits = teacher_data
        s_logits = student_logits

        # Get probabilities
        t_probs = F.softmax(t_logits / self.temperature, dim=-1)
        s_probs = F.softmax(s_logits / self.temperature, dim=-1)
        t_logprobs = F.log_softmax(t_logits / self.temperature, dim=-1)
        s_logprobs = F.log_softmax(s_logits / self.temperature, dim=-1)

        # Compute adaptive weights
        high_ratio, low_ratio = self._get_adaptive_weights(t_probs, s_probs)

        # Forward KL: KL(T || S) on high-confidence region
        # Reverse KL: KL(S || T) on low-confidence region
        fkl = (t_probs * (t_logprobs - s_logprobs)).sum(dim=-1)  # [B, S]
        rkl = (s_probs * (s_logprobs - t_logprobs)).sum(dim=-1)  # [B, S]

        # Weighted combination
        loss = high_ratio * fkl + low_ratio * rkl

        mask = kwargs.get("mask", None)
        if mask is not None:
            loss = loss * mask
            return (loss.sum() / mask.sum().clamp(min=1)) * (self.temperature**2)

        return loss.mean() * (self.temperature**2)

    def _get_adaptive_weights(
        self, t_probs: torch.Tensor, s_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptive weights based on probability mass partitioning.

        Partitions probability mass at threshold mu, then computes error
        ratios for high-confidence and low-confidence regions.
        """
        # Sort by teacher probability (descending)
        t_sorted, indices = torch.sort(t_probs, dim=-1, descending=True)
        s_sorted = torch.gather(s_probs, dim=-1, index=indices)

        # Cumulative sum to find partition point
        t_cumsum = torch.cumsum(t_sorted, dim=-1)

        # Find where cumsum exceeds mu (high-confidence region)
        high_mask = t_cumsum <= self.mu

        # Compute absolute errors in each region
        errors = torch.abs(t_sorted - s_sorted)
        high_errors = (errors * high_mask.float()).sum(dim=-1)
        low_errors = (errors * (~high_mask).float()).sum(dim=-1)

        # Normalize to get ratios (avoid division by zero)
        total_errors = high_errors + low_errors + 1e-8
        high_ratio = high_errors / total_errors
        low_ratio = low_errors / total_errors

        # Reshape for broadcasting [B, S] -> [B, S, 1] if needed
        return high_ratio, low_ratio


class SlimEstimator(DistillationEstimator):
    """SLIM: Sparse Logit Infused Modeling.
    Uses top-k sparse teacher logits to efficiently distill.
    Conceptually similar to Forward KL but computed only on the top-k support.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def compute_loss(self, student_logits, teacher_data, **kwargs):
        if not isinstance(teacher_data, SparseLogits):
            raise TypeError("SlimEstimator requires SparseLogits. Use ForwardKLEstimator for dense logits.")

        # Unpack sparse
        indices = teacher_data.indices  # [B, S, K]
        values = teacher_data.values  # [B, S, K] (logits)

        # Gather student logits at the same indices
        batch, seq, k = indices.shape
        flat_indices = indices.view(batch * seq, k)
        flat_student = student_logits.view(batch * seq, -1)

        student_k_logits = torch.gather(flat_student, 1, flat_indices).view(batch, seq, k)

        # Standard approach for Top-K distillation:
        # 1. Softmax over the K values for teacher
        # 2. Softmax over the SAME K values for student
        # 3. KL between these K-dim distributions
        t_sub_probs = F.softmax(values / self.temperature, dim=-1)
        s_sub_logprobs = F.log_softmax(student_k_logits / self.temperature, dim=-1)

        # KL Divergence on the subset
        loss = F.kl_div(s_sub_logprobs, t_sub_probs, reduction="none")
        loss = loss.sum(-1)  # Sum over K dimension => [B, S]

        mask = kwargs.get("mask", None)
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1)

        return loss.mean()
