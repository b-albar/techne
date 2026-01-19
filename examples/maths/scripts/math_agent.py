"""Example black-box agent implementation for ReTool recipe.
Kept simple but robust.
"""

from __future__ import annotations
import os
import re
import io
import contextlib
import traceback
import signal
import torch
import asyncio
from typing import Any
from techne.agent import Agent
from techne.config import TechneConfig
from techne.data import Trajectory, Step


class CodeSandbox:
    """A safer, stateful python execution environment."""

    def __init__(self, timeout_seconds: int = 5):
        self.scope = {"__builtins__": __builtins__}
        self.timeout = timeout_seconds

    def run(self, code: str) -> str:
        output = io.StringIO()

        # Simple timeout handler using signal (Unix only)
        def handler(signum, frame):
            raise TimeoutError("Execution timed out")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.timeout)

        try:
            with contextlib.redirect_stdout(output):
                exec(code, self.scope)
            result = output.getvalue().strip()
            if not result:
                result = "[No Output]"
            return result
        except TimeoutError:
            return "Error: Execution timed out."
        except Exception:
            return f"Error:\n{traceback.format_exc()}"
        finally:
            signal.alarm(0)


class MathToolAgent(Agent):
    """Agent with multi-round reasoning loop and integrated sandbox (ReTool style)."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant capable of solving math problems.\n"
        "You can use a Python interpreter to calculate results.\n"
        "To execute Python code, wrap it in a markdown block:\n"
        "```python\n"
        "print(12 * 12)\n"
        "```\n"
        "The output will be provided to you in a ```output block.\n"
        "The last line of your response should be of the form:\n"
        "<answer>\n"
        "\\boxed{Answer}\n"
        "</answer>\n"
        "where Answer is the answer to the problem.\n"
        "Solve the following problem:\n"
    )

    def __init__(self, config: TechneConfig, model: Any = None, tokenizer: Any = None):
        self.config = config
        self.model_name = config.model.name_or_path
        self.max_turns = config.rollout.max_turns

        # In-memory model/tokenizer sharing (On-Policy training)
        self.model = model
        self.tokenizer = tokenizer

        # Tags: use from config if available, otherwise defaults
        self.tool_start = self.config.tags.get("tool_start", "```python")
        self.tool_end = self.config.tags.get("tool_end", "```")
        self.resp_start = self.config.tags.get("resp_start", "```output")
        self.resp_end = self.config.tags.get("resp_end", "```")

        # Backend detection
        is_hf = (
            os.path.exists(self.model_name)
            or self.model is not None
            or "/" in self.model_name  # Repo names usually have /
            or self.model_name.startswith("qwen")
            or self.model_name.startswith("llama")
        )

        if is_hf:
            self.backend = "huggingface"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if not self.tokenizer:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if not self.model:
                from transformers import AutoModelForCausalLM

                # Get dtype from config or default to bfloat16
                model_dtype = getattr(self.config.model, "dtype", "bfloat16")
                dtype = (
                    getattr(torch, model_dtype) if hasattr(torch, model_dtype) else torch.bfloat16
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    dtype=dtype,
                )
            if self.config.model.compile and hasattr(torch, "compile"):
                self.model = torch.compile(self.model)
        else:
            self.backend = "openai"
            try:
                from openai import AsyncOpenAI

                self.client = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
                    base_url=os.getenv("OPENAI_API_BASE"),
                )
            except ImportError:
                print("Warning: openai package not installed. OpenAI backend will not work.")

    def tokenize_messages(self, messages: list[dict], max_length: int = 4096) -> dict:
        """Tokenize messages incrementally (message-by-message concat).

        This ensures token IDs match what would be produced during generation,
        avoiding the drift issue when re-tokenizing the full conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_length: Maximum sequence length (truncates from beginning)

        Returns:
            dict with 'input_ids', 'attention_mask', 'labels'
        """
        all_ids = []
        all_labels = []

        for i, msg in enumerate(messages):
            # Get IDs for messages up to and including this one
            full_ids = self.tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=True, add_generation_prompt=False
            )

            # Extract just this message's tokens
            if i == 0:
                step_ids = full_ids
            else:
                prev_ids = self.tokenizer.apply_chat_template(
                    messages[:i], tokenize=True, add_generation_prompt=False
                )
                step_ids = full_ids[len(prev_ids) :]

            all_ids.extend(step_ids)

            # Mask non-assistant roles
            if msg.get("role") == "assistant":
                all_labels.extend(step_ids)
            else:
                all_labels.extend([-100] * len(step_ids))

        # Truncate to max_length (keep most recent tokens)
        if len(all_ids) > max_length:
            all_ids = all_ids[-max_length:]
            all_labels = all_labels[-max_length:]

        return {
            "input_ids": all_ids,
            "attention_mask": [1] * len(all_ids),
            "labels": all_labels,
        }

    async def collect_trajectories(self, prompts: list[str]) -> list[Trajectory]:
        tasks = [self._run_rollout(p) for p in prompts]
        return await asyncio.gather(*tasks)

    async def _run_rollout(self, prompt: str | list[Any] | dict[str, Any]) -> Trajectory:
        trajectory = Trajectory()
        messages = []

        def add_turn(role, content, trainable=False, log_probs=None, provided_token_ids=None):
            # Get prefix tokens
            if not messages:
                prefix_tokens = []
            else:
                prefix_tokens = self.tokenizer.apply_chat_template(messages, tokenize=True)

            # Add message
            messages.append({"role": role, "content": content})
            # Get full tokens
            full_tokens = self.tokenizer.apply_chat_template(messages, tokenize=True)
            # Delta
            if provided_token_ids is not None:
                new_tokens = provided_token_ids
            else:
                new_tokens = full_tokens[len(prefix_tokens) :]

            step = Step(
                role=role,
                content=content,
                token_ids=new_tokens,
                log_probs=log_probs,
                trainable=trainable,
            )
            trajectory.add_step(step)
            return step

        # Build initial history from prompt
        has_system = False
        if isinstance(prompt, list):
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                add_turn(role, content)
                if role == "system":
                    has_system = True
        elif isinstance(prompt, dict):
            role = prompt.get("role", "user")
            content = prompt.get("content", str(prompt))
            add_turn(role, content)
            if role == "system":
                has_system = True
        else:
            add_turn("user", str(prompt))

        # Add default system prompt if missing
        if not has_system:
            # We prepend it to the trajectory
            step = Step(
                role="system",
                content=self.DEFAULT_SYSTEM_PROMPT,
                token_ids=self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}], tokenize=True
                ),
                trainable=False,
            )
            trajectory.steps.insert(0, step)
            messages.insert(0, {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT})

        # Sandbox
        sandbox = CodeSandbox()

        # ReTool Loop
        # We allow up to N turns
        for turn in range(self.max_turns):
            # 1. Generate Context using Chat Template
            context = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if self.backend == "openai":
                text, meta = await self._generate_openai(context)
            else:
                text, meta = await asyncio.to_thread(self._generate_hf, context)

            # 2. Add Assistant Step
            add_turn(
                "assistant",
                text,
                trainable=True,
                log_probs=meta.get("logprobs"),
                provided_token_ids=meta.get("token_ids"),
            )

            # 2. Check for Tool Use
            # Robust extraction
            code_block_pattern = (
                f"{re.escape(self.tool_start)}\\s*(.*?)\\s*{re.escape(self.tool_end)}"
            )
            matches = list(re.finditer(code_block_pattern, text, re.DOTALL))

            if matches:
                # Execute the LAST code block? Or all? ReTool usually executes as it goes.
                # If the model generated multiple blocks, it might be hallucinating the output.
                # We take the *first* block that hasn't been "seen" or just the last one?
                # Since we stopped generation at tool_end (hopefully), there is only one.

                last_match = matches[-1]
                code = last_match.group(1).strip()

                # Execute
                obs = sandbox.run(code)

                # Truncate
                if len(obs) > 1000:
                    obs = obs[:1000] + "...[Truncated]"

                # Format Observation
                formatted_obs = f"\n{self.resp_start}\n{obs}\n{self.resp_end}\n"

                # 3. Add Tool Step (as user in history for template, but tool in trajectory)
                add_turn("user", formatted_obs)
                trajectory.steps[-1].role = "tool"

            elif "<answer>" in text or "</answer>" in text:
                # Finished
                break
            else:
                pass

        return trajectory

    async def _generate_openai(self, context: str) -> tuple[str, dict]:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": context}],
            stop=[self.tool_end],
            logprobs=True,
        )
        msg = response.choices[0].message
        text = msg.content or ""

        # Simple logprob extraction
        logprobs = []
        if response.choices[0].logprobs:
            for item in response.choices[0].logprobs.content:
                logprobs.append(item.logprob)

        return text, {"logprobs": logprobs, "token_ids": []}

    def _generate_hf(self, context: str) -> tuple[str, dict]:
        # Setup inputs
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Extract new tokens
        gen_sequences = outputs.sequences[:, input_len:]
        text = self.tokenizer.decode(gen_sequences[0], skip_special_tokens=True)

        # Calculate logprobs from scores
        logprobs = []
        if outputs.scores:
            token_ids = gen_sequences[0].tolist()
            for i, score_tensor in enumerate(outputs.scores):
                if i < len(token_ids):
                    probs = torch.log_softmax(score_tensor, dim=-1)
                    logprobs.append(probs[0, token_ids[i]].item())

        return text, {"logprobs": logprobs, "token_ids": gen_sequences[0].tolist()}

    async def update_model(self, state_dict: dict[str, Any]) -> None:
        # On-policy: Model is shared by reference, handled by Trainer.
        # If distributed, we might need reload.
        # Here we do nothing as self.model references trainer.model
        pass
