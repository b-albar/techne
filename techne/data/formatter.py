"""Dataset formatting utilities."""

from __future__ import annotations

from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizer

from techne.config import TagConfig


class DatasetFormatter:
    """Formats datasets for tool-augmented training.

    Handles:
    - Multi-turn conversation formatting
    - Chat template application
    - Tool call/response tag insertion
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        tags: TagConfig,
        max_length: int = 8192,
    ):
        """Initialize formatter.

        Args:
            tokenizer: Tokenizer for text processing
            tags: Tag configuration
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.tags = tags
        self.max_length = max_length

    def format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format messages using chat template.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted conversation string
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def format_with_tool_calls(
        self,
        messages: list[dict[str, str]],
        tool_calls: list[dict[str, str]] | None = None,
    ) -> str:
        """Format messages with tool call annotations.

        Args:
            messages: List of message dicts
            tool_calls: Optional tool call info to insert

        Returns:
            Formatted string with tool tags
        """
        # Apply chat template first
        formatted = self.format_messages(messages)

        # Tool calls would already be in the message content
        # This method is for cases where we need to add them programmatically
        if tool_calls:
            for tc in tool_calls:
                code = tc.get("code", "")
                result = tc.get("result", "")

                tool_block = self.tags.wrap_tool_call(code)
                response_block = self.tags.wrap_response(result)

                # Append to formatted string
                formatted += f"\n{tool_block}{response_block}"

        return formatted

    def format_dataset(
        self,
        dataset: Dataset,
        input_column: str = "messages",
        output_column: str = "text",
    ) -> Dataset:
        """Format entire dataset.

        Args:
            dataset: Input dataset
            input_column: Column containing messages
            output_column: Column name for formatted text

        Returns:
            Formatted dataset
        """

        def format_example(example: dict[str, Any]) -> dict[str, Any]:
            messages = example.get(input_column) or example.get("conversations", [])

            # Handle different message formats
            if isinstance(messages, list) and len(messages) > 0:
                if isinstance(messages[0], dict):
                    # Already in correct format
                    pass
                elif isinstance(messages[0], str):
                    # Convert list of strings to messages
                    messages = [
                        {"role": "user" if i % 2 == 0 else "assistant", "content": m}
                        for i, m in enumerate(messages)
                    ]
            else:
                # Single turn format
                messages = [
                    {"role": "user", "content": example.get("prompt", example.get("question", ""))},
                    {
                        "role": "assistant",
                        "content": example.get("response", example.get("answer", "")),
                    },
                ]

            formatted = self.format_messages(messages)

            return {output_column: formatted}

        return dataset.map(format_example, remove_columns=dataset.column_names)

    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
    ) -> Dataset:
        """Tokenize formatted dataset.

        Args:
            dataset: Dataset with text column
            text_column: Column containing text

        Returns:
            Tokenized dataset
        """

        def tokenize(examples: dict[str, list]) -> dict[str, list]:
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None,
            )

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=[text_column],
        )

    def prepare_for_training(
        self,
        dataset: Dataset,
        input_column: str = "messages",
    ) -> Dataset:
        """Full pipeline: format and tokenize.

        Args:
            dataset: Raw dataset
            input_column: Column containing messages

        Returns:
            Training-ready dataset
        """
        formatted = self.format_dataset(dataset, input_column)
        return self.tokenize_dataset(formatted)
