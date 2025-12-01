"""Metrics collection and aggregation."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class EvaluationResult:
    """Result of a single evaluation sample."""
    sample_id: str
    input: str
    expected_output: Any
    model_response: str
    scores: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates evaluation metrics."""

    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize metrics collector.

        Args:
            run_id: Optional run identifier (auto-generated if not provided)
        """
        self.run_id = run_id or self._generate_run_id()
        self.results: List[EvaluationResult] = []

    @staticmethod
    def _generate_run_id() -> str:
        """Generate unique run identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{short_uuid}"

    def add_result(
        self,
        sample_id: str,
        input_text: str,
        expected_output: Any,
        model_response: str,
        scores: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a single evaluation result.

        Args:
            sample_id: Sample identifier
            input_text: Input text
            expected_output: Expected output
            model_response: Model's response
            scores: Score metrics
            metadata: Additional metadata
        """
        result = EvaluationResult(
            sample_id=sample_id,
            input=input_text,
            expected_output=expected_output,
            model_response=model_response,
            scores=scores,
            metadata=metadata or {}
        )
        self.results.append(result)

    def aggregate(self) -> Dict[str, Any]:
        """
        Aggregate metrics across all results.

        Returns:
            Dictionary of aggregated metrics
        """
        if not self.results:
            return {}

        total = len(self.results)

        # Aggregate numeric metrics
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        for result in self.results:
            for key, value in result.scores.items():
                if isinstance(value, (int, float)):
                    metric_sums[key] = metric_sums.get(key, 0.0) + value
                    metric_counts[key] = metric_counts.get(key, 0) + 1

        # Calculate averages
        aggregated = {
            "total_samples": total,
        }

        for key in metric_sums:
            aggregated[f"avg_{key}"] = metric_sums[key] / metric_counts[key]

        # Add pass rate if exact_match exists
        if "exact_match" in metric_sums:
            aggregated["accuracy"] = metric_sums["exact_match"] / total

        # Add parse rate if parse_success exists
        parse_successes = sum(
            1 for r in self.results if r.scores.get("parse_success", False)
        )
        aggregated["parse_rate"] = parse_successes / total

        return aggregated

    def get_failed_samples(self) -> List[EvaluationResult]:
        """
        Get samples that failed (exact_match = 0).

        Returns:
            List of failed evaluation results
        """
        return [
            r for r in self.results
            if r.scores.get("exact_match", 0.0) == 0.0
        ]

    def summary(self) -> str:
        """
        Generate a human-readable summary.

        Returns:
            Summary string
        """
        if not self.results:
            return "No results collected"

        agg = self.aggregate()
        lines = [
            f"Run ID: {self.run_id}",
            f"Total Samples: {agg['total_samples']}",
        ]

        if "accuracy" in agg:
            lines.append(f"Accuracy: {agg['accuracy']:.2%}")

        if "parse_rate" in agg:
            lines.append(f"Parse Rate: {agg['parse_rate']:.2%}")

        failed = len(self.get_failed_samples())
        if failed > 0:
            lines.append(f"Failed Samples: {failed}")

        return "\n".join(lines)
