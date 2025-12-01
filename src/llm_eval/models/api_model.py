"""OpenAI API model implementation."""
import os
from typing import List, Optional
from openai import OpenAI

from .base import BaseModel, Message, ModelResponse


class OpenAIModel(BaseModel):
    """OpenAI API model implementation."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI model.

        Args:
            model_name: OpenAI model identifier (e.g., 'gpt-4o-mini', 'gpt-4o')
            api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate a single response from OpenAI API."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return ModelResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "id": response.id,
            }
        )

    def batch_generate(
        self,
        batch_messages: List[List[Message]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[ModelResponse]:
        """
        Generate responses for multiple inputs.

        Note: Currently uses sequential generation. Can be optimized with
        asyncio or OpenAI's batch API for production use.
        """
        return [
            self.generate(messages, temperature, max_tokens, **kwargs)
            for messages in batch_messages
        ]
