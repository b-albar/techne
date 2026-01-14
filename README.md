# Techne

**Abstract tool-augmented reinforcement learning framework for LLMs**

Techne provides tools for training and distilling agentic language models with multi-turn tool use. It supports SFT, RL (GRPO/PPO), and distillation from any agent system (whitebox and blackbox).

## Features

- 🔧 **Customizable Tool Calling**: Configure any tags for tool invocation (`<code>`, `</code>`, etc.)
- 🚀 **Dual Backend Support**: vLLM and SGLang for efficient rollouts
- 📊 **Multi-Algorithm Training**: PPO and GRPO via TRL
- 🧩 **LoRA Support**: Memory-efficient training with PEFT
- ⚡ **Multi-GPU Scaling**: DeepSpeed/FSDP for distributed training
- 🔌 **Safe Tools Execution**: Code sandbox

## Installation

```bash
pip install techne

# With inference backends
pip install techne[vllm]      # vLLM backend
pip install techne[sglang]    # SGLang backend
pip install techne[all]       # Everything
```

## Quick Start

```python
from techne import TechneConfig, TagConfig
from techne.training import SFTTrainer

# Configure custom tags
tags = TagConfig(
    tool_start="<code>",
    tool_end="</code>",
    response_start="<interpreter>",
    response_end="</interpreter>",
)

config = TechneConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    tags=tags,
    use_lora=True,
)

# Cold-start SFT
trainer = SFTTrainer(config)
trainer.train("path/to/dataset")
```

## Architecture

```
techne/
├── config.py          # Configuration system
├── tools/             # Tool interfaces and executors
├── rollout/           # Multi-turn rollout engine
│   └── backends/      # vLLM, SGLang backends
├── training/          # SFT and RL trainers
└── data/              # Dataset formatting
```

## Training Pipeline

1. **Cold-Start SFT**: Fine-tune base model on tool-augmented reasoning traces
2. **RL Training**: PPO/GRPO with outcome-based rewards for optimal tool use

## References

- [ReTool Paper](https://arxiv.org/abs/2504.11536)
- [TRL Library](https://github.com/huggingface/trl)

## License

Apache License 2.0
