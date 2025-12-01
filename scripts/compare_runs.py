#!/usr/bin/env python3
"""Compare evaluation runs and generate reports."""
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_eval.metrics.storage import MetricsStorage
from llm_eval.ci.thresholds import ThresholdChecker


def load_run(storage: MetricsStorage, run_id: str) -> Dict[str, Any]:
    """Load run data from storage."""
    run_data = storage.get_run(run_id)
    if not run_data:
        print(f"Error: Run {run_id} not found")
        sys.exit(1)
    return run_data


def compare_metrics(baseline: Dict[str, float], current: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Compare two sets of metrics."""
    comparison = {}

    all_keys = set(baseline.keys()) | set(current.keys())

    for key in all_keys:
        base_val = baseline.get(key, 0.0)
        curr_val = current.get(key, 0.0)

        if isinstance(base_val, (int, float)) and isinstance(curr_val, (int, float)):
            diff = curr_val - base_val
            pct_change = (diff / base_val * 100) if base_val != 0 else 0.0

            comparison[key] = {
                "baseline": base_val,
                "current": curr_val,
                "diff": diff,
                "pct_change": pct_change
            }

    return comparison


def generate_markdown_report(baseline: Dict, current: Dict, comparison: Dict, violations: list) -> str:
    """Generate Markdown comparison report."""
    lines = [
        "# Evaluation Comparison Report\n",
        f"**Baseline**: {baseline['run_id']} ({baseline['timestamp']})",
        f"**Current**: {current['run_id']} ({current['timestamp']})\n",
        f"**Task**: {current['task_name']}",
        f"**Model**: {current['model_name']}\n",
    ]

    # Threshold status
    if violations:
        lines.append("## ❌ Threshold Violations\n")
        for v in violations:
            lines.append(f"- {v}\n")
    else:
        lines.append("## ✅ All Thresholds Passed\n")

    # Metrics comparison table
    lines.append("## Metrics Comparison\n")
    lines.append("| Metric | Baseline | Current | Change | % Change |")
    lines.append("|--------|----------|---------|--------|----------|")

    for metric, values in sorted(comparison.items()):
        baseline_val = values["baseline"]
        current_val = values["current"]
        diff = values["diff"]
        pct = values["pct_change"]

        # Format with appropriate symbols
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        pct_str = f"+{pct:.2f}%" if pct > 0 else f"{pct:.2f}%"

        # Add emoji indicators
        if abs(diff) < 0.01:
            indicator = "➖"
        elif diff > 0:
            indicator = "📈"
        else:
            indicator = "📉"

        lines.append(f"| {metric} | {baseline_val:.4f} | {current_val:.4f} | {diff_str} {indicator} | {pct_str} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation runs")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline run ID"
    )
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Current run ID"
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="data/metrics/eval_results.db",
        help="Path to metrics database"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if thresholds violated"
    )
    parser.add_argument(
        "--threshold-config",
        type=str,
        default=None,
        help="Path to threshold configuration YAML"
    )

    args = parser.parse_args()

    # Load runs
    storage = MetricsStorage(args.storage_path)
    baseline_run = load_run(storage, args.baseline)
    current_run = load_run(storage, args.current)

    # Compare metrics
    comparison = compare_metrics(
        baseline_run["aggregate_metrics"],
        current_run["aggregate_metrics"]
    )

    # Check thresholds
    checker = ThresholdChecker(args.threshold_config)
    violations = checker.check(
        current_run["task_name"],
        current_run["aggregate_metrics"]
    )

    # Generate report
    report = generate_markdown_report(baseline_run, current_run, comparison, violations)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)

    # Exit code
    if args.fail_on_regression and violations:
        print(f"\n❌ {len(violations)} threshold violation(s) detected")
        sys.exit(1)

    print(f"\n✅ Comparison complete ({len(comparison)} metrics)")
    sys.exit(0)


if __name__ == "__main__":
    main()
