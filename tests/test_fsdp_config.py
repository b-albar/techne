from techne.config import TechneConfig, TrainingConfig


def test_fsdp_config_fields():
    """Test that FSDP fields are correctly defined in TrainingConfig."""
    config = TrainingConfig()
    assert config.fsdp is None
    assert config.fsdp_config is None


def test_fsdp_config_instantiation():
    """Test instantiation with FSDP values."""
    fsdp_strategy = "full_shard auto_wrap"
    fsdp_conf = {"backward_prefetch": "backward_pre"}

    config = TrainingConfig(fsdp=fsdp_strategy, fsdp_config=fsdp_conf)

    assert config.fsdp == fsdp_strategy
    assert config.fsdp_config == fsdp_conf


def test_techne_config_with_fsdp():
    """Test full TechneConfig with FSDP settings."""
    config_dict = {
        "model": {"name_or_path": "gpt2"},
        "training": {
            "algorithm": "grpo",
            "fsdp": ["full_shard", "auto_wrap"],
            "fsdp_config": {"limit_all_gathers": True},
        },
    }

    config = TechneConfig.model_validate(config_dict)
    assert config.training.fsdp == ["full_shard", "auto_wrap"]
    assert config.training.fsdp_config["limit_all_gathers"] is True
