"""QLoRA finetuning script optimized for consumer GPUs (RTX 3090/4090)."""
import json
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class QLoRAConfig:
    """QLoRA training configuration for consumer GPU."""
    # Model
    base_model: str = "meta-llama/Llama-3.2-3B"  # Smaller model for demo
    output_dir: str = "data/lora_adapters/json_v1"

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None

    # Training parameters
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    max_seq_length: int = 512

    # Optimization for consumer GPU
    load_in_4bit: bool = True
    use_double_quant: bool = True
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True

    # Data
    train_data_path: str = "data/tasks/json_extraction/train.jsonl"
    eval_data_path: str = "data/tasks/json_extraction/validation.jsonl"

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2

    def __post_init__(self):
        if self.lora_target_modules is None:
            # Default LoRA targets for Llama models
            self.lora_target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class QLoRATrainer:
    """QLoRA trainer optimized for consumer GPUs."""

    def __init__(self, config: QLoRAConfig):
        """Initialize trainer with config."""
        self.config = config

        # Check dependencies
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                TrainingArguments,
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from trl import SFTTrainer
        except ImportError as e:
            raise ImportError(
                f"Missing required package: {e}. "
                "Install with: pip install transformers peft trl bitsandbytes accelerate"
            )

        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
        self.BitsAndBytesConfig = BitsAndBytesConfig
        self.TrainingArguments = TrainingArguments
        self.LoraConfig = LoraConfig
        self.get_peft_model = get_peft_model
        self.prepare_model_for_kbit_training = prepare_model_for_kbit_training
        self.SFTTrainer = SFTTrainer

    def load_data(self):
        """Load and format training data."""
        from datasets import Dataset

        def load_jsonl(path):
            """Load JSONL file."""
            data = []
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            return data

        # Load data
        train_data = load_jsonl(self.config.train_data_path)
        eval_data = load_jsonl(self.config.eval_data_path)

        # Format for finetuning (instruction format)
        def format_sample(sample):
            """Format sample as instruction-following example."""
            schema_str = ""
            if "schema" in sample:
                schema_str = f"\n\nExpected JSON Schema:\n{json.dumps(sample['schema'], indent=2)}"

            prompt = f"""Extract the following information and return it as valid JSON:{schema_str}

Text:
{sample['input']}

Return only the JSON output, no additional text."""

            response = json.dumps(sample['expected_output'], indent=2)

            # Format as chat
            text = f"""System: You are a precise data extraction assistant. Extract the requested information from the text and return it as valid JSON. Follow the schema exactly.

User: {prompt}
Assistant: {response}

            return {"text": text}

        # Format datasets
        train_formatted = [format_sample(s) for s in train_data]
        eval_formatted = [format_sample(s) for s in eval_data]

        return Dataset.from_list(train_formatted), Dataset.from_list(eval_formatted)

    def train(self):
        """Run QLoRA finetuning."""
        print(f"Loading base model: {self.config.base_model}")

        # Quantization config (4-bit for consumer GPU)
        bnb_config = self.BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_use_double_quant=self.config.use_double_quant,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load model
        model = self.AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Prepare for k-bit training
        model = self.prepare_model_for_kbit_training(model)

        # LoRA config
        lora_config = self.LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Add LoRA adapters
        model = self.get_peft_model(model, lora_config)

        # Load tokenizer
        tokenizer = self.AutoTokenizer.from_pretrained(self.config.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        # Load data
        print("Loading training data...")
        train_dataset, eval_dataset = self.load_data()
        print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

        # Training arguments (optimized for consumer GPU)
        training_args = self.TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            warmup_ratio=self.config.warmup_ratio,
            optim="paged_adamw_8bit",  # Memory-efficient optimizer
            max_grad_norm=1.0,
            report_to="none"  # Disable wandb/tensorboard
        )

        # Trainer
        trainer = self.SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            tokenizer=tokenizer
        )

        # Train
        print("Starting training...")
        print(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        trainer.train()

        # Save adapter
        print(f"Saving adapter to {self.config.output_dir}")
        model.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)

        print("Training complete!")
        return self.config.output_dir
