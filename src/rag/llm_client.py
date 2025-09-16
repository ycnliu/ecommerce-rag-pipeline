"""
LLM client for generating responses in the RAG pipeline.
"""
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from huggingface_hub import InferenceClient
from loguru import logger

from ..utils.exceptions import LLMError


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.1,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate response from prompt."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class HuggingFaceLLMClient(BaseLLMClient):
    """Hugging Face Inference API client."""

    def __init__(
        self,
        model_name: str,
        api_token: str,
        base_url: Optional[str] = None
    ):
        """
        Initialize Hugging Face LLM client.

        Args:
            model_name: Model name on Hugging Face
            api_token: Hugging Face API token
            base_url: Optional base URL for custom endpoints
        """
        self.model_name = model_name
        self.api_token = api_token
        self.base_url = base_url

        try:
            self.client = InferenceClient(
                model=model_name,
                token=api_token,
                base_url=base_url
            )
            logger.info(f"Initialized Hugging Face client for model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face client: {e}")
            raise LLMError(f"Failed to initialize Hugging Face client: {e}") from e

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.1,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate response using Hugging Face Inference API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Stop sequences for generation

        Returns:
            Generated response text

        Raises:
            LLMError: If generation fails
        """
        try:
            response = self.client.text_generation(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences or []
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise LLMError(f"Failed to generate response: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "provider": "huggingface",
            "base_url": self.base_url
        }


class OpenAILLMClient(BaseLLMClient):
    """OpenAI API client (placeholder for future implementation)."""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model_name: Model name
        """
        self.api_key = api_key
        self.model_name = model_name
        logger.info(f"Initialized OpenAI client for model: {model_name}")

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.1,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate response using OpenAI API."""
        # Placeholder implementation
        raise NotImplementedError("OpenAI client not yet implemented")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "provider": "openai"
        }


def create_llm_client(
    provider: str,
    model_name: str,
    api_token: str,
    **kwargs
) -> BaseLLMClient:
    """
    Factory function to create LLM clients.

    Args:
        provider: LLM provider ("huggingface", "openai")
        model_name: Model name
        api_token: API token
        **kwargs: Additional provider-specific arguments

    Returns:
        LLM client instance

    Raises:
        ValueError: If provider is not supported
    """
    if provider.lower() == "huggingface":
        return HuggingFaceLLMClient(
            model_name=model_name,
            api_token=api_token,
            **kwargs
        )
    elif provider.lower() == "openai":
        return OpenAILLMClient(
            api_key=api_token,
            model_name=model_name,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")