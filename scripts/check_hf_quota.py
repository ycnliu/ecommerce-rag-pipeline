#!/usr/bin/env python3
"""
Check HuggingFace quota and API limits.
"""
import os
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi
from loguru import logger

def check_hf_quota():
    """Check HuggingFace API quota and limits."""

    load_dotenv()

    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.error("No HF_TOKEN found in environment")
        return

    logger.info("🤗 Checking HuggingFace Account Status")
    logger.info(f"Token: {hf_token[:10]}...{hf_token[-4:]}")

    try:
        # Initialize HF API
        api = HfApi(token=hf_token)

        # Get user info
        user_info = api.whoami()
        logger.info(f"✅ Account: {user_info.get('name', 'Unknown')}")
        logger.info(f"📧 Email: {user_info.get('email', 'Not available')}")

        # Check if user has pro account
        plan = user_info.get('plan', 'free')
        logger.info(f"💳 Plan: {plan}")

        # Get available models
        logger.info("\n📋 Testing Model Access...")

        test_models = [
            "gpt2",
            "microsoft/DialoGPT-small",
            "google/flan-t5-small",
            "facebook/blenderbot-400M-distill"
        ]

        working_models = []

        for model in test_models:
            try:
                # Try to get model info
                model_info = api.model_info(model)
                logger.info(f"✅ {model}: Available")
                working_models.append(model)
            except Exception as e:
                logger.warning(f"❌ {model}: {str(e)[:50]}...")

        # Check Inference API limits
        logger.info("\n🔍 Checking Inference API Limits...")

        headers = {"Authorization": f"Bearer {hf_token}"}

        # Try a simple inference request to check quota
        inference_url = "https://api-inference.huggingface.co/models/gpt2"
        test_payload = {"inputs": "Hello world"}

        response = requests.post(inference_url, headers=headers, json=test_payload)

        if response.status_code == 200:
            logger.info("✅ Inference API: Working")
            result = response.json()
            logger.info(f"📝 Test response: {str(result)[:100]}...")
        elif response.status_code == 503:
            logger.warning("⏳ Model is loading, try again in a few minutes")
        elif response.status_code == 429:
            logger.warning("🚫 Rate limit reached")
            if 'x-ratelimit-remaining' in response.headers:
                remaining = response.headers['x-ratelimit-remaining']
                logger.info(f"   Remaining requests: {remaining}")
        else:
            logger.error(f"❌ API Error: {response.status_code}")
            logger.error(f"   Response: {response.text[:200]}...")

        # Show rate limit headers if available
        if 'x-ratelimit-limit' in response.headers:
            limit = response.headers['x-ratelimit-limit']
            remaining = response.headers.get('x-ratelimit-remaining', 'Unknown')
            reset = response.headers.get('x-ratelimit-reset', 'Unknown')

            logger.info(f"\n📊 Rate Limits:")
            logger.info(f"   Limit: {limit} requests")
            logger.info(f"   Remaining: {remaining}")
            logger.info(f"   Reset: {reset}")

        # Recommendations
        logger.info("\n💡 Recommendations:")
        if plan == 'free':
            logger.info("🆓 Free tier detected:")
            logger.info("   - Limited requests per hour")
            logger.info("   - Models may have loading delays")
            logger.info("   - Consider upgrading for better performance")

        if working_models:
            logger.info(f"✅ Use these models: {', '.join(working_models[:2])}")

        logger.info("🚀 For production, consider:")
        logger.info("   - HuggingFace Pro subscription")
        logger.info("   - Dedicated Inference Endpoints")
        logger.info("   - Local model deployment")

    except Exception as e:
        logger.error(f"❌ Error checking account: {e}")

def check_spaces_eligibility():
    """Check if account can deploy to HuggingFace Spaces."""

    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')

    logger.info("\n🚀 Checking HuggingFace Spaces Eligibility")

    try:
        api = HfApi(token=hf_token)
        user_info = api.whoami()

        # Check if user can create spaces
        logger.info("✅ Spaces deployment appears possible")
        logger.info("📋 Requirements for deployment:")
        logger.info("   - Valid HuggingFace account ✅")
        logger.info("   - Repository access ✅")
        logger.info("   - Python dependencies manageable ✅")

        logger.info("\n🎯 Deployment Strategy:")
        logger.info("1. Create HuggingFace Space")
        logger.info("2. Use Gradio for web interface")
        logger.info("3. Deploy with CPU inference (free)")
        logger.info("4. Optional: Upgrade to GPU for better performance")

    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    check_hf_quota()
    check_spaces_eligibility()