"""SFT Trainer for cold-start training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset

from techne.config import TechneConfig
from techne.rollout.parser import TagParser


class SFTTrainer:
    """Supervised Fine-tuning Trainer for cold-start.

    Uses TRL's SFTTrainer with:
    - Multi-turn conversation formatting
    - Interpreter response masking
    - LoRA support via PEFT
    """

    def __init__(self, config: TechneConfig, loss_type: str = "nll"):
        """Initialize SFT trainer.

        Args:
            config: Techne configuration
            loss_type: Loss type for training. Options:
                - "nll": Standard negative log-likelihood (default SFT)
                - "dft": Dynamic Fine-Tuning with reward rectification (better generalization)
        """
        self.config = config
        self.loss_type = loss_type
        self._parser = TagParser(config.tags)
        self._model = None
        self._tokenizer = None
        self._trainer = None

    def _load_model(self) -> None:
        """Load model and tokenizer with LoRA if enabled."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name_or_path,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": self.config.model.trust_remote_code,
            "dtype": self.config.model.torch_dtype,
            "attn_implementation": self.config.model.attn_implementation,
        }

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name_or_path,
            **model_kwargs,
        )

        # Apply LoRA if enabled
        if self.config.model.lora.enabled:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.model.lora.r,
                lora_alpha=self.config.model.lora.alpha,
                lora_dropout=self.config.model.lora.dropout,
                target_modules=self.config.model.lora.target_modules,
                bias=self.config.model.lora.bias,
                task_type=self.config.model.lora.task_type,
            )

            self._model = get_peft_model(self._model, lora_config)
            self._model.print_trainable_parameters()

    def _create_data_collator(self):
        """Create data collator with response masking."""
        from dataclasses import dataclass

        from transformers import DataCollatorForLanguageModeling

        @dataclass
        class MaskedDataCollator(DataCollatorForLanguageModeling):
            """Data collator that masks interpreter responses."""

            parser: TagParser = None

            def __call__(self, features):
                # Let parent handle batching and padding
                batch = super().__call__(features)

                if self.parser is None or "input_ids" not in batch:
                    return batch

                labels = batch["labels"].clone()
                input_ids = batch["input_ids"]

                # Iterate over batch to apply masking
                for i, seq_ids in enumerate(input_ids):
                    # Decode to maintain tag integrity
                    text = self.tokenizer.decode(seq_ids, skip_special_tokens=False)

                    mask_ranges = self.parser.get_response_mask_ranges(text)
                    if not mask_ranges:
                        continue

                    # Get token offsets to map characters to tokens
                    encoding = self.tokenizer(
                        text, return_offsets_mapping=True, add_special_tokens=False
                    )
                    offsets = encoding.offset_mapping

                    # Map valid length
                    seq_len = len(seq_ids)

                    for idx, (start, end) in enumerate(offsets):
                        if idx >= seq_len:
                            break

                        # Mask token if it falls within any interpreter range
                        for m_start, m_end in mask_ranges:
                            # If token overlaps significantly with masked range
                            if max(start, m_start) < min(end, m_end):
                                labels[i, idx] = -100
                                break

                batch["labels"] = labels
                return batch

        return MaskedDataCollator(
            tokenizer=self._tokenizer,
            mlm=False,
            parser=self._parser,
        )

    def _prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training.

        Applies chat template and tokenization.

        Args:
            dataset: Input dataset

        Returns:
            Processed dataset
        """

        def format_conversation(example: dict[str, Any]) -> dict[str, Any]:
            """Format example as conversation."""
            if "messages" in example:
                messages = example["messages"]
            elif "conversations" in example:
                messages = example["conversations"]
            else:
                messages = [
                    {"role": "user", "content": example.get("prompt", example.get("question", ""))},
                    {
                        "role": "assistant",
                        "content": example.get("response", example.get("answer", "")),
                    },
                ]

            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            return {"text": text}

        return dataset.map(format_conversation, remove_columns=dataset.column_names)

    def train(
        self,
        dataset: Dataset | str,
        output_dir: str | None = None,
        **trainer_kwargs: Any,
    ) -> None:
        """Run SFT training.

        Args:
            dataset: Training dataset or path
            output_dir: Output directory (overrides config)
            **trainer_kwargs: Additional kwargs for SFTTrainer
        """
        from datasets import load_dataset
        from trl import SFTConfig
        from trl import SFTTrainer as TRLSFTTrainer

        if self._model is None:
            self._load_model()

        if isinstance(dataset, str):
            dataset = load_dataset(dataset, split="train")

        processed_dataset = self._prepare_dataset(dataset)
        output_dir = output_dir or self.config.output_dir

        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            max_steps=self.config.training.max_steps
            if self.config.training.num_train_epochs is None
            else -1,
            num_train_epochs=self.config.training.num_train_epochs,
            warmup_ratio=self.config.training.warmup_ratio,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            seed=self.config.seed,
            bf16=True,
            gradient_checkpointing=True,
            deepspeed=self.config.training.deepspeed_config,
            # FSDP Arguments
            fsdp=self.config.training.fsdp,
            fsdp_config=self.config.training.fsdp_config,
            # Loss type (sft or dft)
            loss_type=self.loss_type,
            **trainer_kwargs,
        )

        self._trainer = TRLSFTTrainer(
            model=self._model,
            args=training_args,
            train_dataset=processed_dataset,
            data_collator=self._create_data_collator(),
        )

        self._trainer.train()
        self._trainer.save_model(output_dir)
        self._tokenizer.save_pretrained(output_dir)

    def save_model(self, path: str | Path) -> None:
        """Save trained model."""
        if self._model is not None:
            self._model.save_pretrained(path)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(path)
