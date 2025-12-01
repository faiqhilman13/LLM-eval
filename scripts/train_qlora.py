#!/usr/bin/env python3
"""Train a QLoRA adapter for JSON extraction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_eval.finetune.qlora_trainer import QLoRAConfig, QLoRATrainer


def main():
    """Train JSON extraction adapter."""
    print("="*60)
    print("QLoRA Training - JSON Extraction")
    print("="*60)

    # Create config
    config = QLoRAConfig(
        base_model="meta-llama/Llama-3.2-3B",  # Small model for demo
        output_dir="data/lora_adapters/json_v1",
        num_train_epochs=2,  # Reduced for quick demo
        max_seq_length=512,
    )

    # Print config
    print(f"\nBase model: {config.base_model}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print()

    # Train
    trainer = QLoRATrainer(config)
    adapter_path = trainer.train()

    print(f"\n✅ Adapter saved to: {adapter_path}")
    print("\nTo use this adapter:")
    print(f"  python scripts/run_eval.py --task json --model {adapter_path}")


if __name__ == "__main__":
    main()
