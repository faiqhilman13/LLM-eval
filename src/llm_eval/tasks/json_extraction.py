"""JSON extraction task implementation."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ..models.base import Message


@dataclass
class TaskSample:
    """Single evaluation sample."""
    id: str
    input: str
    expected_output: Any
    metadata: Optional[Dict[str, Any]] = None


class JSONExtractionTask:
    """Task for evaluating JSON extraction from text."""

    SYSTEM_PROMPT = """You are a precise data extraction assistant. Extract the requested information from the text and return it as valid JSON. Follow the schema exactly."""

    def __init__(self, data_path: str, split: str = "test"):
        """
        Initialize JSON extraction task.

        Args:
            data_path: Path to the task data directory
            split: Dataset split to use ('train', 'validation', 'test')
        """
        self.data_path = Path(data_path)
        self.split = split
        self.samples = self._load_samples()

    def _load_samples(self) -> List[TaskSample]:
        """Load samples from JSONL file."""
        file_path = self.data_path / f"{self.split}.jsonl"
        samples = []

        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                samples.append(TaskSample(
                    id=data["id"],
                    input=data["input"],
                    expected_output=data["expected_output"],
                    metadata={
                        "schema": data.get("schema"),
                        "slice": data.get("slice", []),
                        "difficulty": data.get("difficulty"),
                    }
                ))

        return samples

    def format_prompt(self, sample: TaskSample) -> List[Message]:
        """
        Format a sample into a prompt for the model.

        Args:
            sample: Task sample to format

        Returns:
            List of messages for the model
        """
        schema_str = ""
        if sample.metadata and "schema" in sample.metadata:
            schema_str = f"\n\nExpected JSON Schema:\n{json.dumps(sample.metadata['schema'], indent=2)}"

        user_prompt = f"""Extract the following information and return it as valid JSON:{schema_str}

Text:
{sample.input}

Return only the JSON output, no additional text."""

        return [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=user_prompt)
        ]

    def get_samples(self, limit: Optional[int] = None) -> List[TaskSample]:
        """
        Get task samples.

        Args:
            limit: Maximum number of samples to return (None for all)

        Returns:
            List of task samples
        """
        if limit is not None:
            return self.samples[:limit]
        return self.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"JSONExtractionTask(split='{self.split}', samples={len(self.samples)})"
