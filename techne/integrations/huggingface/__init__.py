"""HuggingFace integration for Techne.

Provides HuggingFace implementations of Techne's model interfaces.
"""

from techne.integrations.huggingface.model import (
    HuggingFaceInferenceModel,
    HuggingFaceTrainingModel,
    create_teacher_model,
)

__all__ = [
    "HuggingFaceInferenceModel",
    "HuggingFaceTrainingModel",
    "create_teacher_model",
]
