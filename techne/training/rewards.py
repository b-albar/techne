"""Reward functions for RL training."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any


class RewardFunction(ABC):
    """Abstract base class for reward functions.

    Reward functions evaluate trajectories/responses and return a scalar reward.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this reward function."""
        ...

    @abstractmethod
    def compute(
        self,
        response: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Compute reward for a response.

        Args:
            response: Model's response
            ground_truth: Expected answer (for accuracy-based rewards)
            **kwargs: Additional context

        Returns:
            Scalar reward value
        """
        ...


class AccuracyReward(RewardFunction):
    """Accuracy-based reward for math/code problems.

    Extracts answer from boxed format (e.g., \\boxed{answer}) and
    compares with ground truth.
    """

    def __init__(
        self,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        format_penalty: float = -0.1,
        answer_pattern: str = r"\\boxed\{([^}]+)\}",
    ):
        """Initialize accuracy reward.

        Args:
            correct_reward: Reward for correct answer
            incorrect_reward: Reward for incorrect answer
            format_penalty: Penalty for missing answer format
            answer_pattern: Regex pattern to extract answer
        """
        self._correct = correct_reward
        self._incorrect = incorrect_reward
        self._format_penalty = format_penalty
        self._pattern = re.compile(answer_pattern)

    @property
    def name(self) -> str:
        return "accuracy"

    def extract_answer(self, text: str) -> str | None:
        """Extract answer from text using pattern.

        Args:
            text: Text to extract from

        Returns:
            Extracted answer or None
        """
        match = self._pattern.search(text)
        return match.group(1).strip() if match else None

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison.

        Args:
            answer: Answer to normalize

        Returns:
            Normalized answer string
        """
        # Remove whitespace, convert to lowercase
        normalized = answer.strip().lower()
        # Remove common formatting
        normalized = normalized.replace(" ", "")
        return normalized

    def compute(
        self,
        response: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Compute accuracy reward.

        Args:
            response: Model's response
            ground_truth: Expected answer

        Returns:
            Reward value
        """
        if ground_truth is None:
            return 0.0

        predicted = self.extract_answer(response)
        if predicted is None:
            return self._format_penalty

        # Normalize and compare
        pred_normalized = self.normalize_answer(predicted)
        truth_normalized = self.normalize_answer(ground_truth)

        if pred_normalized == truth_normalized:
            return self._correct
        return self._incorrect


class FormatReward(RewardFunction):
    """Reward for correct output format."""

    def __init__(
        self,
        required_patterns: list[str] | None = None,
        bonus: float = 0.1,
    ):
        """Initialize format reward.

        Args:
            required_patterns: Regex patterns that should be present
            bonus: Bonus for each pattern found
        """
        self._patterns = [re.compile(p) for p in (required_patterns or [])]
        self._bonus = bonus

    @property
    def name(self) -> str:
        return "format"

    def compute(
        self,
        response: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Compute format reward.

        Args:
            response: Model's response

        Returns:
            Format bonus
        """
        if not self._patterns:
            return 0.0

        matches = sum(1 for p in self._patterns if p.search(response))
        return self._bonus * matches / len(self._patterns)


class CompositeReward(RewardFunction):
    """Combines multiple reward functions with weights."""

    def __init__(self, rewards: list[tuple[RewardFunction, float]]):
        """Initialize composite reward.

        Args:
            rewards: List of (reward_function, weight) tuples
        """
        self._rewards = rewards

    @property
    def name(self) -> str:
        return "composite"

    def compute(
        self,
        response: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Compute weighted sum of rewards.

        Args:
            response: Model's response
            ground_truth: Expected answer

        Returns:
            Weighted reward sum
        """
        total = 0.0
        for reward_fn, weight in self._rewards:
            total += weight * reward_fn.compute(response, ground_truth, **kwargs)
        return total
