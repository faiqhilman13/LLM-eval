"""LLM-as-judge scorer for subjective evaluation."""
import json
from typing import Any, Dict, Optional
from dataclasses import dataclass

from ..models.base import BaseModel, Message


@dataclass
class JudgeScore:
    """Score from LLM judge."""
    overall_score: float  # 1-5 scale
    correctness: float
    completeness: float
    format_compliance: float
    rationale: str
    raw_response: str


class LLMJudge:
    """LLM-as-judge scorer for subjective quality evaluation."""

    RUBRIC_PROMPT = """You are an expert evaluator assessing the quality of an AI assistant's response.

Score the response on a scale of 1-5 for each criterion:

**Correctness** (1-5):
- 5: Fully correct, matches expected output
- 4: Mostly correct with minor errors
- 3: Partially correct
- 2: Mostly incorrect
- 1: Completely incorrect

**Completeness** (1-5):
- 5: All required information present
- 4: Most information present, minor gaps
- 3: Some key information missing
- 2: Significant information missing
- 1: Mostly incomplete

**Format Compliance** (1-5):
- 5: Perfect adherence to format requirements
- 4: Minor format deviations
- 3: Some format issues
- 2: Significant format problems
- 1: Wrong format entirely

Expected Output:
{expected}

Actual Output:
{actual}

Respond with a JSON object:
{{
  "correctness": <score 1-5>,
  "completeness": <score 1-5>,
  "format_compliance": <score 1-5>,
  "overall": <average of above three>,
  "rationale": "<brief explanation>"
}}"""

    def __init__(self, judge_model: BaseModel, temperature: float = 0.0):
        """
        Initialize LLM judge.

        Args:
            judge_model: Model to use as judge (e.g., GPT-4o-mini)
            temperature: Generation temperature (default: 0.0 for consistency)
        """
        self.judge_model = judge_model
        self.temperature = temperature

    def score(
        self,
        actual: str,
        expected: Any,
        task_context: Optional[str] = None
    ) -> JudgeScore:
        """
        Score a response using LLM judge.

        Args:
            actual: Model's actual output
            expected: Expected output (can be string or structured)
            task_context: Optional task-specific context

        Returns:
            JudgeScore with detailed ratings
        """
        # Format expected output
        if isinstance(expected, dict):
            expected_str = json.dumps(expected, indent=2)
        else:
            expected_str = str(expected)

        # Build prompt
        prompt = self.RUBRIC_PROMPT.format(
            expected=expected_str,
            actual=actual
        )

        if task_context:
            prompt = f"{task_context}\n\n{prompt}"

        # Get judge response
        messages = [Message(role="user", content=prompt)]
        response = self.judge_model.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=500
        )

        # Parse judge response
        try:
            # Try to extract JSON from response
            raw = response.content
            if "```json" in raw:
                start = raw.find("```json") + 7
                end = raw.find("```", start)
                json_str = raw[start:end].strip()
            else:
                # Try to find JSON object
                start = raw.find("{")
                end = raw.rfind("}") + 1
                json_str = raw[start:end]

            scores = json.loads(json_str)

            return JudgeScore(
                overall_score=scores.get("overall", 0.0),
                correctness=scores.get("correctness", 0.0),
                completeness=scores.get("completeness", 0.0),
                format_compliance=scores.get("format_compliance", 0.0),
                rationale=scores.get("rationale", ""),
                raw_response=response.content
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Failed to parse - return low scores
            return JudgeScore(
                overall_score=0.0,
                correctness=0.0,
                completeness=0.0,
                format_compliance=0.0,
                rationale=f"Failed to parse judge response: {e}",
                raw_response=response.content
            )

    def batch_score(
        self,
        pairs: list[tuple[str, Any]],
        task_context: Optional[str] = None
    ) -> list[JudgeScore]:
        """
        Score multiple response pairs.

        Args:
            pairs: List of (actual, expected) tuples
            task_context: Optional task-specific context

        Returns:
            List of JudgeScore objects
        """
        return [
            self.score(actual, expected, task_context)
            for actual, expected in pairs
        ]


def aggregate_judge_scores(scores: list[JudgeScore]) -> Dict[str, float]:
    """
    Aggregate multiple judge scores.

    Args:
        scores: List of judge scores

    Returns:
        Dictionary of aggregated metrics
    """
    if not scores:
        return {}

    total = len(scores)

    return {
        "avg_overall": sum(s.overall_score for s in scores) / total,
        "avg_correctness": sum(s.correctness for s in scores) / total,
        "avg_completeness": sum(s.completeness for s in scores) / total,
        "avg_format_compliance": sum(s.format_compliance for s in scores) / total,
        "pass_rate_4plus": sum(1 for s in scores if s.overall_score >= 4.0) / total,
        "total_samples": total,
    }
