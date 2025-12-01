"""HuggingFace Transformers model implementation."""
import time
from typing import List, Optional
import torch

from .base import BaseModel, Message, ModelResponse


class HuggingFaceModel(BaseModel):
    """HuggingFace Transformers model implementation."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ):
        """
        Initialize HuggingFace model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ('cuda' or 'cpu')
            load_in_8bit: Load model in 8-bit quantization
            load_in_4bit: Load model in 4-bit quantization (QLoRA)
            **kwargs: Additional model configuration
        """
        super().__init__(model_name, **kwargs)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError("transformers package required. Install with: pip install transformers")

        self.device = device

        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "device_map": "auto" if device == "cuda" else None,
            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        if device == "cpu":
            self.model = self.model.to(device)

    def _format_messages(self, messages: List[Message]) -> str:
        """
        Format messages into a prompt string.

        Args:
            messages: List of messages

        Returns:
            Formatted prompt string
        """
        # Simple chat template - can be customized per model
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\nAssistant: "
            elif msg.role == "assistant":
                prompt += f"{msg.content}\n\n"

        return prompt

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate a response from the model."""
        start_time = time.time()

        # Format messages
        prompt = self._format_messages(messages)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        prompt_tokens = inputs['input_ids'].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or 512,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][prompt_tokens:],
            skip_special_tokens=True
        )

        completion_tokens = outputs.shape[1] - prompt_tokens
        latency_ms = (time.time() - start_time) * 1000

        return ModelResponse(
            content=generated_text.strip(),
            model=self.model_name,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            metadata={
                "latency_ms": latency_ms,
                "device": self.device,
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
        batch processing for better throughput.
        """
        return [
            self.generate(messages, temperature, max_tokens, **kwargs)
            for messages in batch_messages
        ]
