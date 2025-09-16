"""
LLM client for generating responses in the RAG pipeline.
"""
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import requests
import json
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
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
    """OpenAI API client with full implementation."""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model_name: Model name
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://api.openai.com/v1/chat/completions"
        logger.info(f"Initialized OpenAI client for model: {model_name}")

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.1,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate response using OpenAI API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }

            if stop_sequences:
                payload["stop"] = stop_sequences

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                error_msg = f"OpenAI API error: {response.status_code}"
                if response.status_code == 401:
                    error_msg += " - Invalid API key"
                elif response.status_code == 429:
                    error_msg += " - Rate limit exceeded"
                elif response.status_code == 402:
                    error_msg += " - Insufficient credits"
                else:
                    error_msg += f" - {response.text[:100]}"

                logger.error(error_msg)
                raise LLMError(error_msg)

        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise LLMError(f"OpenAI API request failed: {e}")
        except Exception as e:
            logger.error(f"OpenAI client error: {e}")
            raise LLMError(f"OpenAI client error: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "provider": "openai",
            "base_url": self.base_url
        }


class FreeLLMClient(BaseLLMClient):
    """Client for free LLM APIs like Hugging Face Inference."""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        api_token: Optional[str] = None,
        device: str = "auto"
    ):
        """Initialize free LLM client."""
        self.model_name = model_name
        self.api_token = api_token
        self.device = device
        self.client = None

        # Try Hugging Face Inference API first (free tier available)
        if HF_AVAILABLE and api_token:
            try:
                self.client = InferenceClient(model=model_name, token=api_token)
                self.client_type = "hf_inference"
                logger.info(f"Using Hugging Face Inference API: {model_name}")
            except Exception as e:
                logger.warning(f"HF Inference API failed: {e}")

        # Fallback to local transformers pipeline
        if self.client is None and TRANSFORMERS_AVAILABLE:
            try:
                self.client = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if device == "cuda" else -1,
                    torch_dtype="auto"
                )
                self.client_type = "transformers"
                logger.info(f"Using local transformers pipeline: {model_name}")
            except Exception as e:
                logger.warning(f"Transformers pipeline failed: {e}")

        if self.client is None:
            logger.warning("No LLM client available, using fallback responses")
            self.client_type = "fallback"

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate response using available free LLM."""
        try:
            if self.client_type == "hf_inference":
                response = self.client.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    return_full_text=False
                )
                return response.strip()

            elif self.client_type == "transformers":
                outputs = self.client(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.client.tokenizer.eos_token_id
                )
                response = outputs[0]['generated_text'][len(prompt):].strip()

                # Apply stop sequences
                if stop_sequences:
                    for stop in stop_sequences:
                        if stop in response:
                            response = response.split(stop)[0].strip()
                            break

                return response

            else:
                # Fallback response generation
                return self._generate_fallback_response(prompt)

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_response(prompt)

    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate fallback response when no LLM is available."""
        # Extract search results from prompt for basic response
        if "search results:" in prompt.lower():
            return ("Based on the search results, I found several relevant products that match your query. "
                   "Please review the products listed above for detailed information including prices, "
                   "specifications, and purchasing links.")
        elif "product" in prompt.lower():
            return ("Here are some product recommendations based on your search. "
                   "Each result includes product details, pricing, and direct links to purchase.")
        else:
            return ("I found some relevant results for your query. "
                   "Please check the search results above for more information.")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "client_type": self.client_type,
            "device": self.device,
            "available": self.client is not None
        }


class OllamaLLMClient(BaseLLMClient):
    """Client for local Ollama models."""

    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """Initialize Ollama client."""
        self.model_name = model_name
        self.base_url = base_url
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if self.model_name in model_names:
                    logger.info(f"Ollama model {self.model_name} is available")
                    return True
                else:
                    logger.warning(f"Ollama model {self.model_name} not found. Available: {model_names}")
            return False
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate response using Ollama."""
        if not self.available:
            return "Ollama service is not available. Please start Ollama and ensure the model is installed."

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            if stop_sequences:
                payload["options"]["stop"] = stop_sequences

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                raise LLMError(f"Ollama API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise LLMError(f"Ollama generation failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "available": self.available,
            "type": "ollama"
        }


class OpenSourceLLMClient(BaseLLMClient):
    """Client for various open-source LLM APIs."""

    def __init__(
        self,
        provider: str = "groq",  # groq, together, replicate
        model_name: str = "llama2-70b-4096",
        api_token: Optional[str] = None,
        **kwargs
    ):
        """Initialize open-source LLM client."""
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_token = api_token

        # Provider configurations
        self.configs = {
            "groq": {
                "base_url": "https://api.groq.com/openai/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
            },
            "together": {
                "base_url": "https://api.together.xyz/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
            }
        }

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate response using open-source LLM API."""
        if self.provider not in self.configs:
            return f"Provider {self.provider} not supported"

        config = self.configs[self.provider]

        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            if stop_sequences:
                payload["stop"] = stop_sequences

            response = requests.post(
                config["base_url"],
                headers=config["headers"],
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"{self.provider} API error: {response.status_code}")
                return f"API Error: {response.status_code}"

        except Exception as e:
            logger.error(f"{self.provider} generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "type": "open_source_api"
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
    elif provider.lower() == "free":
        return FreeLLMClient(
            model_name=model_name,
            api_token=api_token,
            **kwargs
        )
    elif provider.lower() == "ollama":
        return OllamaLLMClient(
            model_name=model_name,
            **kwargs
        )
    elif provider.lower() in ["groq", "together", "replicate"]:
        return OpenSourceLLMClient(
            provider=provider,
            model_name=model_name,
            api_token=api_token,
            **kwargs
        )
    else:
        # Default to free LLM client for unknown providers
        logger.warning(f"Unknown provider '{provider}', using free LLM client")
        return FreeLLMClient(
            model_name="microsoft/DialoGPT-medium",
            api_token=api_token,
            **kwargs
        )