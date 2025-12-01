"""Base model interface for LLM Evaluation Harness."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelResponse:
    """Structured response from an LLM model."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Message:
    """Chat message format."""
    role: str  # 'system', 'user', 'assistant'
    content: str


class BaseModel(ABC):
    """Abstract base class for all LLM models."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the model.

        Args:
            model_name: Identifier for the model
            **kwargs: Additional model-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a response from the model.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            ModelResponse with content and metadata
        """
        pass

    @abstractmethod
    def batch_generate(
        self,
        batch_messages: List[List[Message]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[ModelResponse]:
        """
        Generate responses for multiple inputs in batch.

        Args:
            batch_messages: List of message lists
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            List of ModelResponse objects
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
