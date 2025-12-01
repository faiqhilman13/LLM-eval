"""Threshold configuration and validation for CI/CD."""
import yaml
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ThresholdViolation:
    """Represents a threshold violation."""
    metric: str
    value: float
    threshold: float
    comparison: str  # 'min' or 'max'

    def __str__(self) -> str:
        symbol = ">=" if self.comparison == "min" else "<="
        return f"{self.metric}: {self.value:.4f} (expected {symbol} {self.threshold:.4f})"


class ThresholdChecker:
    """Checks evaluation metrics against configured thresholds."""

    DEFAULT_THRESHOLDS = {
        "json": {
            "accuracy_min": 0.85,
            "parse_rate_min": 0.95,
            "avg_exact_match_min": 0.85,
        },
        "qa": {
            "avg_overall_min": 4.0,
            "avg_correctness_min": 4.0,
            "pass_rate_4plus_min": 0.70,
        }
    }

    def __init__(self, config_path: str = None):
        """
        Initialize threshold checker.

        Args:
            config_path: Optional path to threshold YAML config
        """
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.thresholds = yaml.safe_load(f)
        else:
            self.thresholds = self.DEFAULT_THRESHOLDS

    def check(self, task: str, metrics: Dict[str, Any]) -> List[ThresholdViolation]:
        """
        Check metrics against thresholds.

        Args:
            task: Task name (e.g., 'json', 'qa')
            metrics: Dictionary of metric values

        Returns:
            List of threshold violations (empty if all pass)
        """
        violations = []

        if task not in self.thresholds:
            return violations

        task_thresholds = self.thresholds[task]

        for key, threshold in task_thresholds.items():
            # Parse threshold key: metric_name + _min/_max
            if key.endswith("_min"):
                metric_name = key[:-4]
                comparison = "min"
                if metric_name in metrics and metrics[metric_name] < threshold:
                    violations.append(ThresholdViolation(
                        metric=metric_name,
                        value=metrics[metric_name],
                        threshold=threshold,
                        comparison=comparison
                    ))
            elif key.endswith("_max"):
                metric_name = key[:-4]
                comparison = "max"
                if metric_name in metrics and metrics[metric_name] > threshold:
                    violations.append(ThresholdViolation(
                        metric=metric_name,
                        value=metrics[metric_name],
                        threshold=threshold,
                        comparison=comparison
                    ))

        return violations

    def passes(self, task: str, metrics: Dict[str, Any]) -> bool:
        """
        Check if all thresholds pass.

        Args:
            task: Task name
            metrics: Dictionary of metric values

        Returns:
            True if all thresholds pass, False otherwise
        """
        return len(self.check(task, metrics)) == 0
