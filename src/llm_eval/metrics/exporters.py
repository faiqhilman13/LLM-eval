"""Metrics export utilities for analysis."""
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .storage import MetricsStorage


class CSVExporter:
    """Export metrics to CSV format for analysis."""

    def __init__(self, storage: MetricsStorage):
        """Initialize CSV exporter."""
        self.storage = storage

    def export_run(self, run_id: str, output_path: str):
        """
        Export a single run to CSV.

        Args:
            run_id: Run identifier
            output_path: Path to output CSV file
        """
        # Get run metadata and samples
        run = self.storage.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        samples = self.storage.get_sample_results(run_id)

        # Prepare rows
        rows = []
        for sample in samples:
            row = {
                "run_id": run_id,
                "model": run["model_name"],
                "task": run["task_name"],
                "timestamp": run["timestamp"],
                "sample_id": sample["sample_id"],
                "input": sample["input"],
                "expected_output": json.dumps(sample["expected_output"]),
                "model_response": sample["model_response"],
            }

            # Add scores as separate columns
            for score_key, score_value in sample["scores"].items():
                row[f"score_{score_key}"] = score_value

            # Add metadata
            if sample.get("metadata"):
                for meta_key, meta_value in sample["metadata"].items():
                    if isinstance(meta_value, (str, int, float, bool)):
                        row[f"meta_{meta_key}"] = meta_value

            rows.append(row)

        # Write CSV
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return len(rows)

    def export_all_runs(self, output_dir: str, limit: int = 10):
        """
        Export multiple recent runs to separate CSV files.

        Args:
            output_dir: Directory to save CSV files
            limit: Number of recent runs to export

        Returns:
            Number of runs exported
        """
        runs = self.storage.list_runs(limit=limit)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported = 0
        for run in runs:
            run_id = run["run_id"]
            filename = f"{run_id}.csv"
            self.export_run(run_id, str(output_path / filename))
            exported += 1

        return exported

    def export_summary(self, output_path: str, limit: int = 10):
        """
        Export summary of recent runs to CSV.

        Args:
            output_path: Path to output CSV file
            limit: Number of recent runs to include

        Returns:
            Number of runs exported
        """
        runs = self.storage.list_runs(limit=limit)

        # Prepare rows
        rows = []
        for run in runs:
            row = {
                "run_id": run["run_id"],
                "model": run["model_name"],
                "task": run["task_name"],
                "timestamp": run["timestamp"],
            }

            # Add aggregate metrics
            for metric_key, metric_value in run["aggregate_metrics"].items():
                row[metric_key] = metric_value

            rows.append(row)

        # Write CSV
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return len(rows)


class PrometheusExporter:
    """Export metrics in Prometheus format (optional)."""

    def __init__(self, storage: MetricsStorage):
        """Initialize Prometheus exporter."""
        self.storage = storage

    def export_metrics(self, run_id: str) -> str:
        """
        Export run metrics in Prometheus text format.

        Args:
            run_id: Run identifier

        Returns:
            Prometheus-formatted metrics string
        """
        run = self.storage.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        lines = []
        labels = f'{{run_id="{run_id}",model="{run["model_name"]}",task="{run["task_name"]}"}}'

        # Export aggregate metrics
        for metric_name, metric_value in run["aggregate_metrics"].items():
            if isinstance(metric_value, (int, float)):
                safe_name = metric_name.replace("-", "_")
                lines.append(f"# TYPE llm_eval_{safe_name} gauge")
                lines.append(f"llm_eval_{safe_name}{labels} {metric_value}")

        return "\n".join(lines)
