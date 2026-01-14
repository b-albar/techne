"""Tests for Techne configuration system."""

import pytest
from techne.config import (
    TagConfig,
    ToolConfig,
    LoRAConfig,
    ModelConfig,
    RolloutConfig,
    TrainingConfig,
    TechneConfig,
    TrainingAlgorithm,
    RolloutBackendType,
)


class TestTagConfig:
    """Tests for TagConfig."""

    def test_default_tags(self):
        """Test default tag values."""
        tags = TagConfig()
        assert tags.tool_start == "<code>"
        assert tags.tool_end == "</code>"
        assert tags.response_start == "<interpreter>"
        assert tags.response_end == "</interpreter>"

    def test_custom_tags(self):
        """Test custom tag configuration."""
        tags = TagConfig(
            tool_start="<tool_call>",
            tool_end="</tool_call>",
            response_start="<tool_response>",
            response_end="</tool_response>",
        )
        assert tags.tool_start == "<tool_call>"
        assert tags.tool_end == "</tool_call>"

    def test_get_tool_pattern(self):
        """Test regex pattern generation."""
        tags = TagConfig()
        pattern = tags.get_tool_pattern()
        assert "<code>" in pattern
        assert "</code>" in pattern

    def test_wrap_tool_call(self):
        """Test wrapping content in tool tags."""
        tags = TagConfig()
        wrapped = tags.wrap_tool_call("print('hello')")
        assert wrapped == "<code>print('hello')</code>"

    def test_wrap_response(self):
        """Test wrapping content in response tags."""
        tags = TagConfig()
        wrapped = tags.wrap_response("hello")
        assert wrapped == "<interpreter>hello</interpreter>"


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_defaults(self):
        """Test default LoRA values."""
        lora = LoRAConfig()
        assert lora.enabled is True
        assert lora.r == 64
        assert lora.alpha == 128
        assert "q_proj" in lora.target_modules


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_algorithm_enum(self):
        """Test algorithm enum values."""
        config = TrainingConfig(algorithm=TrainingAlgorithm.GRPO)
        assert config.algorithm == TrainingAlgorithm.GRPO

        config = TrainingConfig(algorithm=TrainingAlgorithm.GSPO)
        assert config.algorithm == TrainingAlgorithm.GSPO

    def test_all_algorithms(self):
        """Test all supported algorithms."""
        for algo in [TrainingAlgorithm.PPO, TrainingAlgorithm.GRPO, TrainingAlgorithm.GSPO]:
            config = TrainingConfig(algorithm=algo)
            assert config.algorithm == algo


class TestRolloutConfig:
    """Tests for RolloutConfig."""

    def test_backend_types(self):
        """Test backend enum values."""
        config = RolloutConfig(backend=RolloutBackendType.VLLM)
        assert config.backend == RolloutBackendType.VLLM

        config = RolloutConfig(backend=RolloutBackendType.SGLANG)
        assert config.backend == RolloutBackendType.SGLANG


class TestTechneConfig:
    """Tests for main TechneConfig."""

    def test_minimal_config(self):
        """Test creating config with minimal required fields."""
        config = TechneConfig(model=ModelConfig(name_or_path="gpt2"))
        assert config.model.name_or_path == "gpt2"
        assert config.tags.tool_start == "<code>"  # Default

    def test_full_config(self):
        """Test creating config with all fields."""
        config = TechneConfig(
            model=ModelConfig(
                name_or_path="Qwen/Qwen2.5-7B-Instruct",
                lora=LoRAConfig(r=32),
            ),
            tags=TagConfig(tool_start="<tool>"),
            tools=ToolConfig(sandbox_url="http://localhost:5555"),
            rollout=RolloutConfig(max_turns=5),
            training=TrainingConfig(algorithm=TrainingAlgorithm.GSPO),
        )
        assert config.model.lora.r == 32
        assert config.tags.tool_start == "<tool>"
        assert config.training.algorithm == TrainingAlgorithm.GSPO
