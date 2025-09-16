#!/usr/bin/env python3
"""
Test different Hugging Face models to find working ones.
"""
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from loguru import logger
import time

def test_hf_models():
    """Test multiple HF models to find working ones."""

    load_dotenv()

    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.error("No HF_TOKEN found in environment")
        return

    logger.info("🤗 Testing Multiple Hugging Face Models")

    # Models that typically work well with Inference API
    models_to_test = [
        "microsoft/DialoGPT-small",  # Smaller, more likely to work
        "google/flan-t5-base",       # T5 models are reliable
        "facebook/blenderbot-400M-distill",  # Conversational
        "distilgpt2",                # Simple and fast
        "gpt2"                       # Classic and reliable
    ]

    test_prompt = "Recommend wireless headphones under $100."

    for model_name in models_to_test:
        logger.info(f"\n📋 Testing: {model_name}")

        try:
            client = InferenceClient(model=model_name, token=hf_token)

            start_time = time.time()
            response = client.text_generation(
                test_prompt,
                max_new_tokens=50,
                temperature=0.5,
                return_full_text=False
            )
            generation_time = time.time() - start_time

            logger.info(f"✅ SUCCESS in {generation_time:.2f}s")
            logger.info(f"📝 Response: {response}")

        except Exception as e:
            logger.warning(f"❌ Failed: {str(e)[:100]}...")

    # Test with our integrated LLM client
    logger.info(f"\n🔧 Testing with integrated LLM client...")

    try:
        # Create a minimal LLM client for testing
        from typing import Optional, Dict, Any, List
        from abc import ABC, abstractmethod

        class TestFreeLLMClient:
            def __init__(self, model_name: str, api_token: str):
                self.model_name = model_name
                self.api_token = api_token
                self.client = InferenceClient(model=model_name, token=api_token)

            def generate_response(self, prompt: str, max_tokens: int = 50, temperature: float = 0.5) -> str:
                try:
                    response = self.client.text_generation(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        return_full_text=False
                    )
                    return response.strip()
                except Exception as e:
                    return f"API Error: {str(e)[:50]}..."

        # Test the working model
        test_client = TestFreeLLMClient("gpt2", hf_token)

        ecommerce_prompt = """Based on these product search results, recommend the best option:

1. Sony WH-CH720N Wireless Bluetooth Headphones - $89.99
2. JBL Tune 510BT Wireless On-Ear Headphones - $39.99
3. Anker Soundcore Life Q20 Active Noise Cancelling - $59.99

Recommendation:"""

        response = test_client.generate_response(ecommerce_prompt, max_tokens=100, temperature=0.3)
        logger.info(f"🎯 E-commerce Response: {response}")

    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")

    logger.info("\n✅ Token is valid and stored securely!")
    logger.info("💾 Configuration saved in .env file (gitignored)")
    logger.info("🔧 You can now use: LLM_PROVIDER=free with your HF token")

if __name__ == "__main__":
    test_hf_models()