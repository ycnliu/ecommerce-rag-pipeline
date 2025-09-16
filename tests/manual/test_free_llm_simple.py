#!/usr/bin/env python3
"""
Simple test script for free LLM response generation.
This test focuses purely on LLM client functionality without dependencies.
"""
import sys
sys.path.append('src')

from src.rag.llm_client import create_llm_client
from loguru import logger

def test_free_llm_simple():
    """Test free LLM clients with simple prompts."""

    logger.info("🆓 Testing Free LLM Response Generation (Simple)")

    # Test different free LLM configurations
    free_llm_configs = [
        {
            "name": "Free Fallback (No dependencies)",
            "provider": "free",
            "model": "fallback",
            "token": None
        },
        {
            "name": "Ollama (if running)",
            "provider": "ollama",
            "model": "llama2",
            "token": None
        }
    ]

    # Simple test prompts that simulate e-commerce queries
    test_prompts = [
        "User query: wireless bluetooth headphones under $100\n\nSearch results:\n1. Product: Sony WH-CH720N Wireless Bluetooth Headphones | Price: $89.99 | Category: Electronics\n2. Product: JBL Tune 510BT Wireless On-Ear Headphones | Price: $39.99 | Category: Audio\n3. Product: Anker Soundcore Life Q20 Hybrid Active Noise Cancelling Headphones | Price: $59.99 | Category: Electronics\n\nProvide a helpful response:",

        "User query: educational toys for 5 year old kids\n\nSearch results:\n1. Product: LEGO Classic Creative Bricks Set | Price: $24.99 | Category: Toys & Games\n2. Product: Melissa & Doug Wooden Shape Sorting Cube | Price: $19.99 | Category: Educational Toys\n3. Product: LeapFrog LeapStart Interactive Learning System | Price: $34.99 | Category: Educational Electronics\n\nProvide a helpful response:"
    ]

    logger.info("\n" + "="*60)
    logger.info("🧪 TESTING FREE LLM RESPONSE GENERATION")
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
            logger.info(f"\n🔍 Testing query about bluetooth headphones...")

            # Generate response
            import time
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

    # Demonstrate response variations
    logger.info("\n" + "="*60)
    logger.info("🎯 RESPONSE GENERATION EXAMPLES")
    logger.info("="*60)

    # Use fallback client for guaranteed responses
    fallback_client = create_llm_client(
        provider="free",
        model_name="fallback",
        api_token=None
    )

    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\n--- Example {i} ---")

        # Extract query from prompt for display
        query_line = [line for line in prompt.split('\n') if line.startswith('User query:')][0]
        logger.info(f"Query: {query_line.replace('User query: ', '')}")

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

    # Show available options summary
    logger.info("\n" + "="*60)
    logger.info("📋 FREE LLM OPTIONS SUMMARY")
    logger.info("="*60)

    print("\n🆓 Available Free LLM Options:")
    print("1. **Fallback Responses** (Always available)")
    print("   - No setup required")
    print("   - Basic templated responses")
    print("   - Good for testing and demos")

    print("\n2. **Local Transformers** (pip install transformers torch)")
    print("   - Models: GPT-2, DistilGPT-2, etc.")
    print("   - Runs on your hardware")
    print("   - Privacy-friendly")

    print("\n3. **Ollama** (https://ollama.ai)")
    print("   - Install: curl https://ollama.ai/install.sh | sh")
    print("   - Models: llama2, mistral, codellama")
    print("   - Run: ollama run llama2")

    print("\n4. **Free API Tiers**")
    print("   - Hugging Face Inference API (limited free tier)")
    print("   - Groq API (fast inference, some free credits)")
    print("   - Together AI (limited free tier)")

    print("\n5. **Setup Instructions:**")
    print("   # For fallback (no setup needed):")
    print("   LLM_PROVIDER=free")
    print("   LLM_MODEL_NAME=fallback")
    print("")
    print("   # For Ollama:")
    print("   LLM_PROVIDER=ollama")
    print("   LLM_MODEL_NAME=llama2")
    print("")
    print("   # For Transformers:")
    print("   pip install transformers torch")
    print("   LLM_PROVIDER=free")
    print("   LLM_MODEL_NAME=gpt2")

    logger.info("\n🎉 Free LLM testing completed!")

if __name__ == "__main__":
    test_free_llm_simple()