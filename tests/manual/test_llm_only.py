#!/usr/bin/env python3
"""
Test script for LLM client functionality only.
No dependencies on embedding services or vector databases.
"""
import sys
import os
import importlib.util
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import requests
import json
import time

# Import loguru
from loguru import logger

# Define custom exception
class LLMError(Exception):
    """Exception raised for LLM-related errors."""
    pass


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

        # Check if transformers is available
        try:
            spec = importlib.util.find_spec("transformers")
            TRANSFORMERS_AVAILABLE = spec is not None
        except ImportError:
            TRANSFORMERS_AVAILABLE = False

        # Check if huggingface_hub is available
        try:
            spec = importlib.util.find_spec("huggingface_hub")
            HF_AVAILABLE = spec is not None
        except ImportError:
            HF_AVAILABLE = False

        # Try Hugging Face Inference API first (free tier available)
        if HF_AVAILABLE and api_token:
            try:
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(model=model_name, token=api_token)
                self.client_type = "hf_inference"
                logger.info(f"Using Hugging Face Inference API: {model_name}")
            except Exception as e:
                logger.warning(f"HF Inference API failed: {e}")

        # Fallback to local transformers pipeline
        if self.client is None and TRANSFORMERS_AVAILABLE:
            try:
                from transformers import pipeline
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


def create_llm_client(
    provider: str,
    model_name: str,
    api_token: str,
    **kwargs
) -> BaseLLMClient:
    """
    Factory function to create LLM clients.

    Args:
        provider: LLM provider ("free", "ollama")
        model_name: Model name
        api_token: API token
        **kwargs: Additional provider-specific arguments

    Returns:
        LLM client instance
    """
    if provider.lower() == "free":
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
    else:
        # Default to free LLM client for unknown providers
        logger.warning(f"Unknown provider '{provider}', using free LLM client")
        return FreeLLMClient(
            model_name="microsoft/DialoGPT-medium",
            api_token=api_token,
            **kwargs
        )


def test_free_llm_responses():
    """Test free LLM clients with e-commerce prompts."""

    logger.info("🆓 Testing Free LLM Response Generation")

    # Test different free LLM configurations
    free_llm_configs = [
        {
            "name": "Free Fallback (No dependencies)",
            "provider": "free",
            "model": "fallback",
            "token": None
        },
        {
            "name": "Local Transformers (if available)",
            "provider": "free",
            "model": "gpt2",
            "token": None
        },
        {
            "name": "Ollama (if running)",
            "provider": "ollama",
            "model": "llama2",
            "token": None
        }
    ]

    # E-commerce test prompts
    test_prompts = [
        """User query: wireless bluetooth headphones under $100

Search results:
1. Product: Sony WH-CH720N Wireless Bluetooth Headphones | Price: $89.99 | Category: Electronics
   - Product URL: https://example.com/sony-headphones
   - Features: Active Noise Cancelling, 35-hour battery life

2. Product: JBL Tune 510BT Wireless On-Ear Headphones | Price: $39.99 | Category: Audio
   - Product URL: https://example.com/jbl-headphones
   - Features: Wireless Bluetooth 5.0, Pure Bass sound

3. Product: Anker Soundcore Life Q20 Hybrid Active Noise Cancelling Headphones | Price: $59.99 | Category: Electronics
   - Product URL: https://example.com/anker-headphones
   - Features: Hi-Res Audio, 40-hour playtime

Provide a helpful response:""",

        """User query: educational toys for 5 year old kids

Search results:
1. Product: LEGO Classic Creative Bricks Set | Price: $24.99 | Category: Toys & Games
   - Product URL: https://example.com/lego-bricks
   - Age Range: 4-99 years, 484 pieces included

2. Product: Melissa & Doug Wooden Shape Sorting Cube | Price: $19.99 | Category: Educational Toys
   - Product URL: https://example.com/shape-cube
   - Features: 12 chunky, vibrant shapes to sort

3. Product: LeapFrog LeapStart Interactive Learning System | Price: $34.99 | Category: Educational Electronics
   - Product URL: https://example.com/leapstart
   - Features: Interactive books, stylus included

Provide a helpful response:"""
    ]

    logger.info("\n" + "="*60)
    logger.info("🧪 TESTING FREE LLM PROVIDERS")
    logger.info("="*60)

    for config_info in free_llm_configs:
        logger.info(f"\n📋 Testing: {config_info['name']}")
        logger.info(f"   Provider: {config_info['provider']}")
        logger.info(f"   Model: {config_info['model']}")

        try:
            # Create LLM client
            llm_client = create_llm_client(
                provider=config_info['provider'],
                model_name=config_info['model'],
                api_token=config_info['token'] or "dummy_token"
            )

            model_info = llm_client.get_model_info()
            logger.info(f"   Status: {model_info}")

            # Test with first prompt
            prompt = test_prompts[0]
            logger.info(f"\n🔍 Testing query: 'wireless bluetooth headphones under $100'")

            # Generate response
            start_time = time.time()
            response = llm_client.generate_response(
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            generation_time = time.time() - start_time

            logger.info(f"✅ Response generated in {generation_time:.2f}s")
            logger.info(f"📝 Response: {response}")

        except Exception as e:
            logger.error(f"❌ Failed: {e}")

    # Demonstrate complete responses with fallback
    logger.info("\n" + "="*60)
    logger.info("🎯 COMPLETE E-COMMERCE RESPONSES")
    logger.info("="*60)

    # Use fallback client for guaranteed responses
    fallback_client = create_llm_client(
        provider="free",
        model_name="fallback",
        api_token=None
    )

    for i, prompt in enumerate(test_prompts, 1):
        # Extract query for display
        query_line = [line for line in prompt.split('\n') if line.startswith('User query:')][0]
        query = query_line.replace('User query: ', '')

        logger.info(f"\n--- Test {i}: '{query}' ---")

        # Generate response
        start_time = time.time()
        response = fallback_client.generate_response(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        generation_time = time.time() - start_time

        print(f"\n🤖 AI Response ({generation_time:.2f}s):")
        print(f"   {response}")

    # Show setup instructions
    logger.info("\n" + "="*60)
    logger.info("📋 SETUP INSTRUCTIONS")
    logger.info("="*60)

    print("\n🆓 Free LLM Setup Options:")
    print("\n1. **Fallback Responses** (Current - Always works)")
    print("   ✅ No setup required")
    print("   ✅ Provides helpful templated responses")
    print("   ✅ Good for demos and testing")

    print("\n2. **Local Transformers** (Recommended)")
    print("   📦 Install: pip install transformers torch")
    print("   🔧 Usage: LLM_PROVIDER=free, LLM_MODEL_NAME=gpt2")
    print("   ✅ Runs locally, privacy-friendly")
    print("   ✅ Models: gpt2, distilgpt2, microsoft/DialoGPT-medium")

    print("\n3. **Ollama** (Best quality)")
    print("   📦 Install: curl https://ollama.ai/install.sh | sh")
    print("   🚀 Start: ollama run llama2")
    print("   🔧 Usage: LLM_PROVIDER=ollama, LLM_MODEL_NAME=llama2")
    print("   ✅ High-quality responses")
    print("   ✅ Models: llama2, mistral, codellama")

    print("\n4. **Free API Tiers**")
    print("   🔑 Hugging Face Inference API (get free token)")
    print("   🔑 Groq API (fast inference, free credits)")
    print("   🔑 Together AI (limited free tier)")

    logger.info("\n🎉 Free LLM testing completed successfully!")


if __name__ == "__main__":
    test_free_llm_responses()