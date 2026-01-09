#!/usr/bin/env python3
"""
Simple test for Hugging Face API integration.
"""
import os
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from loguru import logger


def test_hf_direct():
    """Test Hugging Face Inference API directly."""

    # Load environment variables
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("No HF_TOKEN found in environment")
        return

    logger.info("🤗 Testing Hugging Face Inference API")
    logger.info(f"Token: {hf_token[:10]}...{hf_token[-4:]}")

    # Test with a reliable model
    model_name = "microsoft/DialoGPT-medium"

    try:
        client = InferenceClient(model=model_name, token=hf_token)

        test_prompt = "Based on the following products, recommend the best wireless headphones under $100: Sony WH-CH720N ($89.99), JBL Tune 510BT ($39.99), Anker Soundcore Life Q20 ($59.99)."

        logger.info(f"📋 Testing model: {model_name}")
        logger.info(f"🔍 Prompt: {test_prompt}")

        start_time = time.time()
        response = client.text_generation(
            test_prompt, max_new_tokens=100, temperature=0.7, return_full_text=False
        )
        generation_time = time.time() - start_time

        logger.info(f"✅ Response generated in {generation_time:.2f}s")
        logger.info(f"📝 Response: {response}")

        # Test with a simpler model
        logger.info(f"\n📋 Testing simpler model: google/flan-t5-small")

        client2 = InferenceClient(model="google/flan-t5-small", token=hf_token)

        simple_prompt = "Recommend the best wireless headphones under $100 from: Sony WH-CH720N, JBL Tune 510BT, Anker Soundcore Life Q20"

        start_time = time.time()
        response2 = client2.text_generation(
            simple_prompt, max_new_tokens=50, temperature=0.3
        )
        generation_time2 = time.time() - start_time

        logger.info(f"✅ Response generated in {generation_time2:.2f}s")
        logger.info(f"📝 Response: {response2}")

    except Exception as e:
        logger.error(f"❌ Error: {e}")

        # Show fallback options
        logger.info("\n💡 If API limits are reached, you can still use:")
        logger.info("1. Local transformers: pip install transformers torch")
        logger.info("2. Ollama: curl https://ollama.ai/install.sh | sh")
        logger.info("3. Fallback responses (always available)")

    logger.info("\n🎉 Hugging Face testing completed!")


if __name__ == "__main__":
    test_hf_direct()
