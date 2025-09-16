#!/usr/bin/env python3
"""
Test Hugging Face API integration with the provided token.
"""
import os
import sys
sys.path.append('src')

from dotenv import load_dotenv
from src.rag.llm_client import create_llm_client
from loguru import logger

def test_huggingface_integration():
    """Test Hugging Face API with the provided token."""

    # Load environment variables
    load_dotenv()

    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.error("No HF_TOKEN found in environment")
        return

    logger.info("🤗 Testing Hugging Face API Integration")
    logger.info(f"Token: {hf_token[:10]}...{hf_token[-4:]}")

    # Test different HF models that work well with the inference API
    test_models = [
        "microsoft/DialoGPT-medium",
        "facebook/blenderbot-400M-distill",
        "google/flan-t5-small"
    ]

    test_prompt = """User query: wireless bluetooth headphones under $100

Search results:
1. Product: Sony WH-CH720N Wireless Bluetooth Headphones | Price: $89.99 | Category: Electronics
2. Product: JBL Tune 510BT Wireless On-Ear Headphones | Price: $39.99 | Category: Audio
3. Product: Anker Soundcore Life Q20 Hybrid Active Noise Cancelling Headphones | Price: $59.99 | Category: Electronics

Provide a helpful response:"""

    for model_name in test_models:
        logger.info(f"\n📋 Testing model: {model_name}")

        try:
            # Create HF client
            llm_client = create_llm_client(
                provider="free",
                model_name=model_name,
                api_token=hf_token
            )

            model_info = llm_client.get_model_info()
            logger.info(f"   Client type: {model_info.get('client_type', 'unknown')}")

            # Generate response
            import time
            start_time = time.time()
            response = llm_client.generate_response(
                prompt=test_prompt,
                max_tokens=100,
                temperature=0.7
            )
            generation_time = time.time() - start_time

            logger.info(f"✅ Response generated in {generation_time:.2f}s")
            logger.info(f"📝 Response: {response}")

        except Exception as e:
            logger.error(f"❌ Failed with {model_name}: {e}")

    logger.info("\n🎉 Hugging Face API testing completed!")

if __name__ == "__main__":
    test_huggingface_integration()