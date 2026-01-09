#!/usr/bin/env python3
"""
Test OpenAI API integration with the provided key.
"""
import os
import sys
import time

sys.path.append("src")

from dotenv import load_dotenv
from src.rag.llm_client import create_llm_client
from loguru import logger


def test_openai_integration():
    """Test OpenAI API with the provided key."""

    # Load environment variables
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("No OPENAI_API_KEY found in environment")
        return

    logger.info("🔑 Testing OpenAI API Integration")
    logger.info(f"Key: {openai_key[:14]}...{openai_key[-4:]}")

    # Test different OpenAI models
    test_models = [
        "gpt-3.5-turbo",  # Most cost-effective
        "gpt-3.5-turbo-0125",  # Latest version
        "gpt-4o-mini",  # If available in your tier
    ]

    # E-commerce specific test prompts
    test_prompts = [
        {
            "title": "Product Recommendation",
            "prompt": """Based on the following e-commerce search results, provide helpful product recommendations:

User Query: wireless bluetooth headphones under $100

Search Results:
1. Sony WH-CH720N Wireless Bluetooth Headphones - $89.99
   Features: Active Noise Cancelling, 35-hour battery, comfortable over-ear design
   Category: Electronics

2. JBL Tune 510BT Wireless On-Ear Headphones - $39.99
   Features: Bluetooth 5.0, Pure Bass sound, 40-hour battery life
   Category: Audio

3. Anker Soundcore Life Q20 Hybrid Active Noise Cancelling - $59.99
   Features: Hi-Res Audio, 40-hour playtime, memory foam ear cups
   Category: Electronics

Please provide a detailed recommendation considering features, price, and value.""",
        },
        {
            "title": "Simple Query",
            "prompt": "What are the key features to look for when buying wireless headphones?",
        },
    ]

    for model_name in test_models:
        logger.info(f"\n📋 Testing model: {model_name}")

        try:
            # Create OpenAI client
            llm_client = create_llm_client(
                provider="openai", model_name=model_name, api_token=openai_key
            )

            model_info = llm_client.get_model_info()
            logger.info(f"   Status: {model_info}")

            # Test with first prompt
            test_case = test_prompts[0]
            logger.info(f"\n🔍 Testing: {test_case['title']}")

            # Generate response
            start_time = time.time()
            response = llm_client.generate_response(
                prompt=test_case["prompt"], max_tokens=200, temperature=0.3
            )
            generation_time = time.time() - start_time

            logger.info(f"✅ Response generated in {generation_time:.2f}s")
            logger.info(f"📝 Response: {response}")

            # Show cost estimate
            estimated_tokens = len(test_case["prompt"].split()) + len(response.split())
            estimated_cost = estimated_tokens * 0.0015 / 1000  # gpt-3.5-turbo pricing
            logger.info(f"💰 Estimated cost: ~${estimated_cost:.4f}")

            break  # Use the first working model

        except Exception as e:
            logger.error(f"❌ Failed with {model_name}: {e}")

    # Test full e-commerce pipeline with OpenAI
    logger.info("\n" + "=" * 60)
    logger.info("🎯 FULL E-COMMERCE RAG WITH OPENAI")
    logger.info("=" * 60)

    try:
        # Use the best available model
        openai_client = create_llm_client(
            provider="openai", model_name="gpt-3.5-turbo", api_token=openai_key
        )

        for i, test_case in enumerate(test_prompts, 1):
            logger.info(f"\n--- Test {i}: {test_case['title']} ---")

            start_time = time.time()
            response = openai_client.generate_response(
                prompt=test_case["prompt"], max_tokens=250, temperature=0.3
            )
            generation_time = time.time() - start_time

            print(f"\n🤖 OpenAI Response ({generation_time:.2f}s):")
            print(f"   {response}")

            # Cost tracking
            estimated_tokens = len(test_case["prompt"].split()) + len(response.split())
            estimated_cost = estimated_tokens * 0.0015 / 1000
            print(f"💰 Estimated cost: ~${estimated_cost:.4f}")

    except Exception as e:
        logger.error(f"❌ Full pipeline test failed: {e}")

    # Show usage summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 OPENAI INTEGRATION SUMMARY")
    logger.info("=" * 60)

    print("\n✅ OpenAI API Setup Complete!")
    print(f"🔑 API Key: {openai_key[:14]}...{openai_key[-4:]}")
    print("💰 Credits: ~$3.00 available")
    print("🎯 Model: gpt-3.5-turbo (recommended for cost-effectiveness)")

    print("\n📋 Cost Estimates:")
    print("   - Simple query: ~$0.001 per response")
    print("   - Complex e-commerce query: ~$0.003-0.005 per response")
    print("   - Your $3 budget: ~600-3000 responses")

    print("\n🚀 Usage Recommendations:")
    print("   1. Use gpt-3.5-turbo for best cost/performance ratio")
    print("   2. Set max_tokens=200-300 to control costs")
    print("   3. Temperature=0.3 for consistent e-commerce responses")
    print("   4. Monitor usage via OpenAI dashboard")

    print("\n🔧 Configuration:")
    print("   LLM_PROVIDER=openai")
    print("   LLM_MODEL_NAME=gpt-3.5-turbo")
    print("   OPENAI_API_KEY=[securely stored in .env]")


if __name__ == "__main__":
    test_openai_integration()
