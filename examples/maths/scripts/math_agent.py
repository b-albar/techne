"""Example black-box agent implementation for ReTool recipe."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import re
import signal
import traceback
from typing import Any

import torch

from techne.agent import Agent
from techne.config import TechneConfig
from techne.data import Step, Trajectory


class CodeSandbox:
    """A safer, stateful python execution environment."""

    def __init__(self, timeout_seconds: int = 5):
        self.scope = {"__builtins__": __builtins__}
        self.timeout = timeout_seconds

    def run(self, code: str) -> str:
        output = io.StringIO()

        def handler(signum, frame):
            raise TimeoutError("Execution timed out")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.timeout)

        try:
            with contextlib.redirect_stdout(output):
                exec(code, self.scope)
            result = output.getvalue().strip()
            return result if result else "[No Output]"
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

        # Model/tokenizer (shared for on-policy training)
        self.model = model
        self.tokenizer = tokenizer

        # Tool tags
        self.tool_start = config.tags.get("tool_start", "```python")
        self.tool_end = config.tags.get("tool_end", "```")
        self.resp_start = config.tags.get("resp_start", "```output")
        self.resp_end = config.tags.get("resp_end", "```")

        # Backend setup
        self._setup_backend()

        # Async lock for generation serialization
        self.lock = asyncio.Lock()

    def _setup_backend(self):
        """Initialize model backend (HuggingFace or OpenAI)."""
        is_hf = (
            os.path.exists(self.model_name)
            or self.model is not None
            or "/" in self.model_name
            or self.model_name.startswith(("qwen", "llama"))
        )

        if is_hf:
            self.backend = "huggingface"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._setup_hf_model()
        else:
            self.backend = "openai"
            self._setup_openai_client()

    def _setup_hf_model(self):
        """Load HuggingFace model and tokenizer."""
        if not self.tokenizer:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if not self.model:
            from transformers import AutoModelForCausalLM

            dtype = self._get_torch_dtype()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto", dtype=dtype
            )

        if self.config.model.compile and hasattr(torch, "compile"):
            self.model.forward = torch.compile(self.model.forward)

        # Cache special tokens for message construction
        self._im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._nl_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
        # Closing tokens for assistant messages: <|im_end|>\n
        self._msg_close_tokens = [self._im_end_id] + self._nl_tokens

    def _get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from config."""
        model_dtype = getattr(self.config.model, "dtype", "bfloat16")
        if isinstance(model_dtype, torch.dtype):
            return model_dtype
        if isinstance(model_dtype, str) and hasattr(torch, model_dtype):
            return getattr(torch, model_dtype)
        return torch.bfloat16

    def _setup_openai_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
                base_url=os.getenv("OPENAI_API_BASE"),
            )
        except ImportError:
            print("Warning: openai package not installed. OpenAI backend will not work.")

    def _tokenize_message(self, role: str, content: str) -> list[int]:
        """Tokenize a message in <|im_start|>role\ncontent<|im_end|>\n format."""
        role_tokens = self.tokenizer.encode(role, add_special_tokens=False)
        content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
        return (
            [self._im_start_id]
            + role_tokens
            + self._nl_tokens
            + content_tokens
            + [self._im_end_id]
            + self._nl_tokens
        )

    def _get_gen_prompt_tokens(self) -> list[int]:
        """Get tokens for generation prompt: <|im_start|>assistant\n"""
        return (
            [self._im_start_id]
            + self.tokenizer.encode("assistant", add_special_tokens=False)
            + self._nl_tokens
        )

    async def collect_trajectories(
        self, prompts: list[str], use_kv_cache: bool = True
    ) -> list[Trajectory]:
        tasks = [self._run_rollout(p, use_kv_cache=use_kv_cache) for p in prompts]
        return await asyncio.gather(*tasks)

    async def _run_rollout(
        self, prompt: str | list[Any] | dict[str, Any], use_kv_cache: bool = True
    ) -> Trajectory:
        trajectory = Trajectory()
        messages: list[dict] = []
        accumulated_tokens: list[int] = []

        # KV cache state (only for HuggingFace backend)
        past_key_values = None
        cache_len = 0  # Track number of tokens in cache
        pending_tokens: list[int] = []  # Tokens to process in next generate call
        use_cache = use_kv_cache and self.backend == "huggingface"

        def add_turn(
            role: str,
            content: str,
            trainable: bool = False,
            log_probs: list[float] | None = None,
            token_ids: list[int] | None = None,
        ):
            nonlocal accumulated_tokens
            messages.append({"role": role, "content": content})

            # Use provided tokens for assistant, otherwise tokenize manually
            new_tokens = list(token_ids) if token_ids else self._tokenize_message(role, content)
            accumulated_tokens = accumulated_tokens + new_tokens

            trajectory.add_step(
                Step(
                    role=role,
                    content=content,
                    token_ids=new_tokens,
                    log_probs=log_probs,
                    trainable=trainable,
                )
            )
            return new_tokens

        # Build initial history
        has_system = self._add_initial_messages(prompt, add_turn)

        # Add default system prompt if missing
        if not has_system:
            system_tokens = self._tokenize_message("system", self.DEFAULT_SYSTEM_PROMPT)
            accumulated_tokens = system_tokens + accumulated_tokens
            messages.insert(0, {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT})
            trajectory.steps.insert(
                0,
                Step(
                    role="system",
                    content=self.DEFAULT_SYSTEM_PROMPT,
                    token_ids=system_tokens,
                    trainable=False,
                ),
            )

        sandbox = CodeSandbox()
        gen_prompt_tokens = self._get_gen_prompt_tokens()

        for turn in range(self.max_turns):
            if use_cache:
                # KV cache path: accumulate tokens and pass to generate
                if past_key_values is None:
                    input_ids = accumulated_tokens + gen_prompt_tokens
                else:
                    input_ids = pending_tokens + gen_prompt_tokens

                text, meta, past_key_values, cache_len = await self._generate_with_cache(
                    input_ids, past_key_values, cache_len
                )
                pending_tokens = []
            else:
                # No cache: decode full context each time
                context = self.tokenizer.decode(
                    accumulated_tokens + gen_prompt_tokens, skip_special_tokens=False
                )
                text, meta = await self._generate(context)

            # Add assistant response
            add_turn(
                "assistant",
                text,
                trainable=True,
                log_probs=meta.get("logprobs"),
                token_ids=meta.get("token_ids"),
            )

            # Queue closing tokens for next generation if model didn't generate them
            if use_cache:
                gen_token_ids = meta.get("token_ids", [])
                if not gen_token_ids or gen_token_ids[-1] != self._im_end_id:
                    pending_tokens.extend(self._msg_close_tokens)

            # Check for tool use
            code = self._extract_code(text)
            if code:
                obs = sandbox.run(code)
                if len(obs) > 1000:
                    obs = obs[:1000] + "...[Truncated]"
                formatted_obs = f"\n{self.resp_start}\n{obs}\n{self.resp_end}\n"
                user_tokens = add_turn("user", formatted_obs)
                trajectory.steps[-1].role = "tool"

                if use_cache:
                    pending_tokens.extend(user_tokens)

            elif "<answer>" in text or "</answer>" in text:
                break

            elif turn < self.max_turns - 1:
                if turn == self.max_turns - 2:
                    msg = "Please provide your final answer now using the <answer>\\boxed{...}</answer> format."
                else:
                    msg = "Continue."
                user_tokens = add_turn("user", msg)

                if use_cache:
                    pending_tokens.extend(user_tokens)

        return trajectory

    async def _generate_with_cache(
        self, input_ids: list[int], past_key_values: Any | None, cache_len: int
    ) -> tuple[str, dict, Any, int]:
        """Generate with KV cache (HuggingFace only)."""
        async with self.lock:
            return await asyncio.to_thread(
                self._generate_hf_with_cache, input_ids, past_key_values, cache_len
            )

    def _add_initial_messages(self, prompt, add_turn) -> bool:
        """Add initial messages from prompt. Returns True if system message exists."""
        has_system = False

        if isinstance(prompt, list):
            for msg in prompt:
                role = msg.get("role", "user")
                add_turn(role, msg.get("content", ""))
                if role == "system":
                    has_system = True
        elif isinstance(prompt, dict):
            role = prompt.get("role", "user")
            add_turn(role, prompt.get("content", str(prompt)))
            if role == "system":
                has_system = True
        else:
            add_turn("user", str(prompt))

        return has_system

    def _extract_code(self, text: str) -> str | None:
        """Extract code from tool block if present."""
        pattern = f"{re.escape(self.tool_start)}\\s*(.*?)\\s*{re.escape(self.tool_end)}"
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
        return None

    async def _generate(self, context: str) -> tuple[str, dict]:
        """Generate response using appropriate backend."""
        if self.backend == "openai":
            return await self._generate_openai(context)
        async with self.lock:
            return await asyncio.to_thread(self._generate_hf, context)

    async def _generate_openai(self, context: str) -> tuple[str, dict]:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": context}],
            stop=[self.tool_end],
            logprobs=True,
        )
        text = response.choices[0].message.content or ""
        logprobs = []
        if response.choices[0].logprobs:
            logprobs = [item.logprob for item in response.choices[0].logprobs.content]
        return text, {"logprobs": logprobs, "token_ids": []}

    def _generate_hf(self, context: str) -> tuple[str, dict]:
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.rollout.max_new_tokens,
                do_sample=True,
                temperature=self.config.rollout.temperature,
                top_p=self.config.rollout.top_p,
                top_k=self.config.rollout.top_k,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs.sequences[0, input_len:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Calculate logprobs from scores
        logprobs = []
        if outputs.scores:
            token_ids = gen_tokens.tolist()
            for i, scores in enumerate(outputs.scores):
                if i < len(token_ids):
                    log_probs = torch.log_softmax(scores, dim=-1)
                    logprobs.append(log_probs[0, token_ids[i]].item())

        return text, {"logprobs": logprobs, "token_ids": gen_tokens.tolist()}

    def _generate_hf_with_cache(
        self, input_ids: list[int], past_key_values: Any | None, cache_len: int
    ) -> tuple[str, dict, Any, int]:
        """Generate with KV cache support."""
        from transformers import DynamicCache

        input_tensor = torch.tensor([input_ids], device=self.device)
        input_len = len(input_ids)

        # Build generate kwargs
        gen_kwargs = {
            "input_ids": input_tensor,
            "max_new_tokens": self.config.rollout.max_new_tokens,
            "do_sample": True,
            "temperature": self.config.rollout.temperature,
            "top_p": self.config.rollout.top_p,
            "top_k": self.config.rollout.top_k,
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Pass past_key_values, attention_mask, and cache_position if we have cache
        if past_key_values is not None:
            if isinstance(past_key_values, tuple):
                cache = DynamicCache.from_legacy_cache(past_key_values)
            else:
                cache = past_key_values

            actual_cache_len = cache.get_seq_length()
            gen_kwargs["past_key_values"] = cache
            gen_kwargs["attention_mask"] = torch.ones(
                1, actual_cache_len + input_len, device=self.device, dtype=torch.long
            )
            gen_kwargs["cache_position"] = torch.arange(
                actual_cache_len, actual_cache_len + input_len, device=self.device
            )

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        # Generated tokens start after input
        gen_tokens = outputs.sequences[0, input_len:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Calculate logprobs from scores
        logprobs = []
        if outputs.scores:
            token_ids = gen_tokens.tolist()
            for i, scores in enumerate(outputs.scores):
                if i < len(token_ids):
                    log_probs = torch.log_softmax(scores, dim=-1)
                    logprobs.append(log_probs[0, token_ids[i]].item())

        # Get new cache length from cache itself
        new_cache = outputs.past_key_values
        new_cache_len = (
            new_cache.get_seq_length()
            if hasattr(new_cache, "get_seq_length")
            else cache_len + input_len + len(gen_tokens)
        )

        return (
            text,
            {"logprobs": logprobs, "token_ids": gen_tokens.tolist()},
            new_cache,
            new_cache_len,
        )

    async def update_model(self, state_dict: dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python math_agent.py [--no-cache] <prompt>")
        sys.exit(1)

    # Parse args
    use_kv_cache = True
    args = sys.argv[1:]
    if "--no-cache" in args:
        use_kv_cache = False
        args.remove("--no-cache")

    prompt = args[0] if args else ""

    config = TechneConfig(
        **{
            "model": {"name_or_path": "Qwen/Qwen3-0.6B", "dtype": "bfloat16", "compile": False},
            "rollout": {
                "max_turns": 10,
                "max_new_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 50,
            },
        }
    )

    agent = MathToolAgent(config)
    trajectory = asyncio.run(agent._run_rollout(prompt, use_kv_cache=use_kv_cache))
    print(f"[KV Cache: {'enabled' if use_kv_cache else 'disabled'}]\n")

    # Print trajectory
    print("\n" + "=" * 50)
    print("TRAJECTORY")
    print("=" * 50)
    for step in trajectory.steps:
        print(
            f"\n[{step.role.upper()}] (tokens: {len(step.token_ids)}, trainable: {step.trainable})"
        )
        content = step.content[:500] + "..." if len(step.content) > 500 else step.content
        print(content)

    # Print final answer
    print("\n" + "=" * 50)
    print("FINAL ANSWER")
    print("=" * 50)
    for step in reversed(trajectory.steps):
        if step.role == "assistant":
            match = re.search(r"<answer>(.*?)</answer>", step.content, re.DOTALL)
            if match:
                print(match.group(1).strip())
                break
    else:
        print("No answer found in trajectory.")
