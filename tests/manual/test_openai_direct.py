#!/usr/bin/env python3
"""
Direct test of OpenAI API without dependencies.
"""
import json
import os
import time

import requests
from dotenv import load_dotenv
from loguru import logger


def test_openai_direct():
    """Test OpenAI API directly."""

    # Load environment variables
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("No OPENAI_API_KEY found in environment")
        return

    logger.info("🔑 Testing OpenAI API Direct Integration")
    logger.info(f"Key: {openai_key[:14]}...{openai_key[-4:]}")

    # Test API with e-commerce specific prompt
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }

    test_prompt = """Based on the following e-commerce search results, provide helpful product recommendations:

User Query: wireless bluetooth headphones under $100

Search Results:
1. Sony WH-CH720N Wireless Bluetooth Headphones - $89.99
   Features: Active Noise Cancelling, 35-hour battery, comfortable over-ear design

2. JBL Tune 510BT Wireless On-Ear Headphones - $39.99
   Features: Bluetooth 5.0, Pure Bass sound, 40-hour battery life

3. Anker Soundcore Life Q20 Hybrid Active Noise Cancelling - $59.99
   Features: Hi-Res Audio, 40-hour playtime, memory foam ear cups

Please provide a detailed recommendation considering features, price, and value."""

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": test_prompt}],
        "max_tokens": 250,
        "temperature": 0.3,
    }

    try:
        logger.info("📋 Testing with gpt-3.5-turbo...")
        logger.info("🔍 Sending e-commerce recommendation request...")

        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        generation_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"].strip()

            logger.info(f"✅ Response generated in {generation_time:.2f}s")
            logger.info(f"📝 Response: {ai_response}")

            # Show usage and cost info
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # GPT-3.5-turbo pricing: $0.0015/1K tokens input, $0.002/1K tokens output
            input_cost = prompt_tokens * 0.0015 / 1000
            output_cost = completion_tokens * 0.002 / 1000
            total_cost = input_cost + output_cost

            logger.info(f"\n📊 Usage Statistics:")
            logger.info(f"   Prompt tokens: {prompt_tokens}")
            logger.info(f"   Completion tokens: {completion_tokens}")
            logger.info(f"   Total tokens: {total_tokens}")
            logger.info(f"   Cost: ${total_cost:.4f}")

        else:
            logger.error(f"❌ API Error: {response.status_code}")
            logger.error(f"   Response: {response.text}")

            if response.status_code == 401:
                logger.error("   → Invalid API key")
            elif response.status_code == 429:
                logger.error("   → Rate limit exceeded")
            elif response.status_code == 402:
                logger.error("   → Insufficient credits")

    except Exception as e:
        logger.error(f"❌ Request failed: {e}")

    # Test with a simpler query to verify basic functionality
    logger.info("\n" + "=" * 50)
    logger.info("🧪 Testing simple query...")

    simple_payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": "What are the top 3 factors to consider when buying wireless headphones?",
            }
        ],
        "max_tokens": 100,
        "temperature": 0.3,
    }

    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=simple_payload, timeout=30)
        generation_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"].strip()

            logger.info(f"✅ Simple query successful in {generation_time:.2f}s")
            logger.info(f"📝 Response: {ai_response}")

            usage = result.get("usage", {})
            total_cost = usage.get("total_tokens", 0) * 0.0015 / 1000
            logger.info(f"💰 Cost: ${total_cost:.4f}")

        else:
            logger.error(f"❌ Simple query failed: {response.status_code}")

    except Exception as e:
        logger.error(f"❌ Simple query error: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 OPENAI API INTEGRATION SUMMARY")
    logger.info("=" * 60)

    print("\n🔑 API Configuration:")
    print(f"   Key: {openai_key[:14]}...{openai_key[-4:]}")
    print(f"   Model: gpt-3.5-turbo")
    print(f"   Budget: ~$3.00")

    print("\n💰 Cost Analysis:")
    print("   - Input: $0.0015 per 1K tokens")
    print("   - Output: $0.002 per 1K tokens")
    print("   - Typical e-commerce query: $0.003-0.008")
    print("   - Your budget allows: ~375-1000 queries")

    print("\n🎯 Recommendations:")
    print("   1. Use temperature=0.3 for consistent responses")
    print("   2. Set max_tokens=200-300 to control costs")
    print("   3. Monitor usage in OpenAI dashboard")
    print("   4. Consider batch processing for efficiency")

    print("\n🚀 Next Steps:")
    print("   1. Update .env: LLM_PROVIDER=openai")
    print("   2. Deploy with OpenAI for better responses")
    print("   3. Monitor costs and usage patterns")


if __name__ == "__main__":
    test_openai_direct()
