import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Optional


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
        teacher_data: Union[torch.Tensor, SparseLogits],
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


class SlimEstimator(DistillationEstimator):
    """SLIM: Sparse Logit Infused Modeling.
    Uses top-k sparse teacher logits to efficienty distill.
    Conceptually similar to Forward KL but computed only on the top-k support.
    """

    def __init__(self, temperature: float = 1.0, alpha: float = 0.0):
        # alpha can be used for dynamic weighting parameter if needed
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, student_logits, teacher_data, **kwargs):
        if not isinstance(teacher_data, SparseLogits):
            # Fallback to dense if provided?
            return ForwardKLEstimator(self.temperature).compute_loss(student_logits, teacher_data)

        # Unpack sparse
        indices = teacher_data.indices  # [B, S, K]
        values = teacher_data.values  # [B, S, K] (logits)

        # Gather student logits at the same indices
        # student_logits: [B, S, V]
        # We need to gather along dim 2
        B, S, K = indices.shape
        flat_indices = indices.view(B * S, K)
        flat_student = student_logits.view(B * S, -1)

        student_k_logits = torch.gather(flat_student, 1, flat_indices).view(B, S, K)

        # Normalize to probability distributions locally on the k support?
        # Or treat them as unnormalized logits and do Softmax over K?
        # SLIM paper usually implies approximating the full distribution q(x) approx truncated q_hat(x).

        # Standard approach for Top-K distillation:
        # 1. Softmax over the K values for teacher
        # 2. Softmax over the SAME K values for student
        # 3. KL between these K-dim distributions

        t_sub_probs = F.softmax(values / self.temperature, dim=-1)
        s_sub_logprobs = F.log_softmax(student_k_logits / self.temperature, dim=-1)

        # KL Divergence on the subset
        loss = F.kl_div(s_sub_logprobs, t_sub_probs, reduction="none")
        loss = loss.sum(-1)  # Sum over K dimension => [B, S]

        # Masking? (Assumed handled by caller or mask passed in kwargs)
        mask = kwargs.get("mask", None)
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum()

        return loss.mean()
