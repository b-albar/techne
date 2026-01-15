import re
from typing import Any


class MathReward:
    """Reward function for mathematical accuracy.

    Extracts the answer from the model output and compares it with the ground truth.
    """

    @staticmethod
    def extract_answer(text: str) -> str | None:
        """Extracts the answer from <answer> tags or 'Answer:'."""
        # 1. Look for <answer>value</answer>
        tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if tag_match:
            answer_part = tag_match.group(1).strip()
            # 2. Look for \boxed{...} inside the answer tag
            boxed_match = re.search(r"\\boxed\{([^}]+)\}", answer_part)
            if boxed_match:
                return boxed_match.group(1).strip()
            return answer_part

        # 3. Look for \boxed{...} outside tags (GSM8K fallback)
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # 4. Look for "Answer: <value>" (Legacy/Fallback)
        match = re.search(r"Answer:\s*(.*)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: last number in text
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        return numbers[-1] if numbers else None

    @staticmethod
    def is_correct(predicted: str, reference: str) -> bool:
        """Robust comparison of math answers."""
        if not predicted or not reference:
            return False

        # Simple string cleaning
        clean_p = predicted.replace("$", "").replace(",", "").strip()
        clean_r = reference.replace("$", "").replace(",", "").strip()

        try:
            # Numerical comparison
            return float(clean_p) == float(clean_r)
        except ValueError:
            # String comparison fallback
            return clean_p.lower() == clean_r.lower()

    def __call__(self, trajectory: Any, reference_answer: str) -> float:
        """Assign 1.0 for correct, 0.0 for incorrect."""
        # Find the last assistant message that might contain the answer
        last_content = ""
        for step in reversed(trajectory.steps):
            if step.role == "assistant":
                last_content = step.content
                break

        predicted = self.extract_answer(last_content)
        return 1.0 if self.is_correct(predicted, reference_answer) else 0.0
