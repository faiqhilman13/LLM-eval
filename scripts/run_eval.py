#!/usr/bin/env python3
"""Command-line interface for LLM Evaluation Harness."""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_eval.models.api_model import OpenAIModel
from llm_eval.models.hf_model import HuggingFaceModel
from llm_eval.runner import EvaluationRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation tasks"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["json", "qa"],
        help="Task to evaluate (json, qa)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model to use as judge for QA task (default: same as --model)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., gpt-4o-mini, meta-llama/Llama-3.2-3B)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["api", "local"],
        default="api",
        help="Model type: 'api' for OpenAI, 'local' for HuggingFace (default: api)"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization (for local models)"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization (for local models)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (default: all)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per generation (default: model default)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to task data (default: data/tasks/{task_name})"
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="data/metrics/eval_results.db",
        help="Path to metrics database"
    )

    args = parser.parse_args()

    # Set default data path
    if args.data_path is None:
        project_root = Path(__file__).parent.parent
        if args.task == "json":
            args.data_path = project_root / "data" / "tasks" / "json_extraction"
        else:
            args.data_path = project_root / "data" / "tasks" / args.task

    # Initialize model
    print(f"Initializing model: {args.model}")
    print(f"Model type: {args.model_type}")

    if args.model_type == "api":
        model = OpenAIModel(model_name=args.model)
    else:  # local
        print(f"Loading local model (this may take a few minutes)...")
        model = HuggingFaceModel(
            model_name=args.model,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit
        )
        print(f"✓ Model loaded on {model.device}")

    # Initialize judge model if needed (for QA task)
    judge_model = None
    if args.task == "qa":
        judge_model_name = args.judge_model or args.model
        print(f"Initializing judge model: {judge_model_name}")
        # Judge is always API for simplicity
        if "gpt" in judge_model_name or "claude" in judge_model_name:
            judge_model = OpenAIModel(model_name=judge_model_name)
        else:
            print("Warning: Using local model as judge (may be slow)")
            judge_model = model  # Reuse same model

    # Initialize runner
    runner = EvaluationRunner(
        model=model,
        task_name=args.task,
        data_path=str(args.data_path),
        storage_path=args.storage_path,
        judge_model=judge_model
    )

    # Run evaluation
    results = runner.run(
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Run ID: {results['run_id']}")
    print(f"Model: {results['model']}")
    print(f"Task: {results['task']}")
    print(f"Samples: {results['sample_count']}")
    print("\nMetrics:")
    for key, value in results['aggregate_metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
