# Math Tool-Use Example

This example demonstrates training a language model to solve math problems using tool calls (Python code execution).

## Overview

Following the ReTool approach, we train models in two stages:
1. **SFT (Supervised Fine-Tuning)**: Teach the model to use tool calls for math
2. **RL (Reinforcement Learning)**: Improve problem-solving through GRPO/PPO

## Datasets

| Type | Dataset | Description | Size |
|------|---------|-------------|------|
| SFT | `swordfaith/ReTool-SFT-multi-turn` | Math problems with tool call examples | 2K |
| RL | `BytedTsinghua-SIA/DAPO-Math-17k` | Math problems and answers | 17K |
| Eval | `BytedTsinghua-SIA/AIME-2024` | AIME 2024 competition problems | 30 |

## Quick Start

### 1. Download Datasets

```bash
# Install dependencies
pip install -e ".[all]"

# Download and preprocess datasets
python examples/maths/scripts/prepare_data.py
```

### 2. SFT Training

```bash
# Train with tool calling
python examples/maths/scripts/train_sft.py \
    --config examples/maths/configs/sft.yaml \
    --data-path examples/maths/data/sft
```

### 3. RL Training

**Option A: White-Box**

Uses vLLM/SGLang backend for rollouts. Policy model generates trajectories directly.

```bash
python examples/maths/scripts/train_rl_whitebox.py \
    --config examples/maths/configs/rl_whitebox.yaml \
    --data-path examples/maths/data/rl
```

**Option B: Black-Box**

Uses the **same model** accessed via API endpoint. Policy model learns from agent's trajectories.

> **Purpose**: Compare white-box vs black-box consistency using identical models

```bash
# First, start a vLLM server with your trained model in another terminal:
# vllm serve ./output/sft/checkpoint-final --port 8000

# Then run black-box training:
python examples/maths/scripts/train_rl_blackbox.py \
    --config examples/maths/configs/rl_blackbox.yaml \
    --data-path examples/maths/data/rl \
    --agent-model ./output/sft/checkpoint-final \
    --inference-server http://localhost:8000/v1
```

### 4. Evaluation

```bash
# Evaluate on AIME-2024
python examples/maths/scripts/evaluate.py \
    --checkpoint output/rl_whitebox/checkpoint-final \
    --dataset examples/maths/data/eval
```

## Citation

Based on the ReTool paper and methodology:
- Dataset: [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- Evaluation: [AIME-2024](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024)
