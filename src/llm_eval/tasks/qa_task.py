"""Q&A task implementation."""
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from ..models.base import Message


@dataclass
class QASample:
    """Single Q&A evaluation sample."""
    id: str
    question: str
    context: str
    answer: str
    reference_facts: List[str]
    slice: List[str]


class QATask:
    """Task for evaluating question answering capability."""

    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions accurately and concisely. Use the provided context to answer the question. If the answer is not in the context, say so."""

    def __init__(self, data_path: str, split: str = "test"):
        """
        Initialize Q&A task.

        Args:
            data_path: Path to the task data directory
            split: Dataset split to use ('train', 'validation', 'test')
        """
        self.data_path = Path(data_path)
        self.split = split
        self.samples = self._load_samples()

    def _load_samples(self) -> List[QASample]:
        """Load samples from JSONL file."""
        file_path = self.data_path / f"{self.split}.jsonl"
        samples = []

        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                samples.append(QASample(
                    id=data["id"],
                    question=data["question"],
                    context=data["context"],
                    answer=data["answer"],
                    reference_facts=data.get("reference_facts", []),
                    slice=data.get("slice", [])
                ))

        return samples

    def format_prompt(self, sample: QASample) -> List[Message]:
        """
        Format a sample into a prompt for the model.

        Args:
            sample: Task sample to format

        Returns:
            List of messages for the model
        """
        user_prompt = f"""Context:
{sample.context}

Question:
{sample.question}

Answer:"""

        return [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=user_prompt)
        ]

    def get_samples(self, limit: Optional[int] = None, slice_filter: Optional[str] = None) -> List[QASample]:
        """
        Get task samples with optional filtering.

        Args:
            limit: Maximum number of samples to return (None for all)
            slice_filter: Filter by slice category (e.g., 'factual', 'easy')

        Returns:
            List of task samples
        """
        samples = self.samples

        # Filter by slice if specified
        if slice_filter:
            samples = [s for s in samples if slice_filter in s.slice]

        # Apply limit
        if limit is not None:
            samples = samples[:limit]

        return samples

    def get_slices(self) -> dict[str, int]:
        """
        Get count of samples by slice.

        Returns:
            Dictionary mapping slice names to counts
        """
        slice_counts = {}
        for sample in self.samples:
            for slice_name in sample.slice:
                slice_counts[slice_name] = slice_counts.get(slice_name, 0) + 1
        return slice_counts

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        slices = self.get_slices()
        return f"QATask(split='{self.split}', samples={len(self.samples)}, slices={slices})"
