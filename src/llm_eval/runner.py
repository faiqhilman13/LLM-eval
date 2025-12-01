"""Evaluation runner orchestrator."""
from typing import Any, Dict, Optional
from pathlib import Path
import logging

from .models.base import BaseModel
from .tasks.json_extraction import JSONExtractionTask
from .tasks.qa_task import QATask
from .scorers.deterministic import score_json_extraction
from .scorers.llm_judge import LLMJudge
from .metrics.collector import MetricsCollector
from .metrics.storage import MetricsStorage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Orchestrates evaluation runs."""

    def __init__(
        self,
        model: BaseModel,
        task_name: str,
        data_path: str,
        storage_path: str = "data/metrics/eval_results.db",
        judge_model: Optional[BaseModel] = None
    ):
        """
        Initialize evaluation runner.

        Args:
            model: Model to evaluate
            task_name: Task identifier ('json', 'qa')
            data_path: Path to task data
            storage_path: Path to metrics database
            judge_model: Optional LLM judge model for Q&A scoring
        """
        self.model = model
        self.task_name = task_name
        self.storage = MetricsStorage(storage_path)
        self.judge = LLMJudge(judge_model) if judge_model else None

        # Initialize task
        if task_name == "json":
            self.task = JSONExtractionTask(data_path, split="test")
        elif task_name == "qa":
            self.task = QATask(data_path, split="test")
        else:
            raise ValueError(f"Unknown task: {task_name}")

    def run(
        self,
        limit: Optional[int] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation.

        Args:
            limit: Maximum number of samples to evaluate (None for all)
            temperature: Model temperature
            max_tokens: Maximum tokens per generation
            run_id: Optional run identifier

        Returns:
            Dictionary with results and metrics
        """
        collector = MetricsCollector(run_id=run_id)
        samples = self.task.get_samples(limit=limit)

        logger.info(f"Starting evaluation: {self.task_name}")
        logger.info(f"Model: {self.model.model_name}")
        logger.info(f"Samples: {len(samples)}")
        logger.info(f"Run ID: {collector.run_id}")

        for idx, sample in enumerate(samples, 1):
            # Format prompt
            messages = self.task.format_prompt(sample)

            # Generate response
            logger.info(f"Processing sample {idx}/{len(samples)}: {sample.id}")
            response = self.model.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Score response
            if self.task_name == "json":
                scores = score_json_extraction(
                    response.content,
                    sample.expected_output
                )
                input_text = sample.input
                expected_output = sample.expected_output
                metadata = sample.metadata
            elif self.task_name == "qa":
                if self.judge:
                    judge_score = self.judge.score(response.content, sample.answer)
                    scores = {
                        "overall_score": judge_score.overall_score,
                        "correctness": judge_score.correctness,
                        "completeness": judge_score.completeness,
                        "format_compliance": judge_score.format_compliance,
                        "rationale": judge_score.rationale
                    }
                else:
                    scores = {}
                input_text = f"{sample.question}\n\nContext: {sample.context}"
                expected_output = sample.answer
                metadata = {"slice": sample.slice, "reference_facts": sample.reference_facts}
            else:
                scores = {}
                input_text = str(sample)
                expected_output = ""
                metadata = {}

            # Collect results
            collector.add_result(
                sample_id=sample.id,
                input_text=input_text,
                expected_output=expected_output,
                model_response=response.content,
                scores=scores,
                metadata={
                    **metadata,
                    "usage": response.usage
                }
            )

            # Save to storage
            self.storage.save_sample_result(
                run_id=collector.run_id,
                sample_id=sample.id,
                input_text=sample.input,
                expected_output=sample.expected_output,
                model_response=response.content,
                scores=scores,
                metadata={
                    **sample.metadata,
                    "usage": response.usage
                }
            )

        # Aggregate metrics
        aggregate_metrics = collector.aggregate()

        # Save run metadata
        self.storage.save_run(
            run_id=collector.run_id,
            model_name=self.model.model_name,
            task_name=self.task_name,
            config={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "limit": limit,
            },
            aggregate_metrics=aggregate_metrics
        )

        logger.info("\nEvaluation Complete!")
        logger.info(collector.summary())

        return {
            "run_id": collector.run_id,
            "model": self.model.model_name,
            "task": self.task_name,
            "aggregate_metrics": aggregate_metrics,
            "sample_count": len(samples)
        }
