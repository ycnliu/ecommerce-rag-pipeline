#!/usr/bin/env python3
"""
Test script for free LLM response generation in e-commerce RAG pipeline.
"""
import os
import sys
import time
sys.path.append('src')

from src.embedding.service import CLIPEmbeddingService
from src.vector_db.faiss_service import FAISSVectorDB
from src.rag.llm_client import create_llm_client
from src.rag.prompt_builder import PromptBuilder
from src.utils.logging import setup_logging
from src.utils.config import Config
from loguru import logger

def test_free_llm_responses():
    """Test free LLM response generation with search results."""
    os.environ['DEVICE'] = 'mps'
    config = Config()
    setup_logging(config)

    logger.info("🆓 Testing Free LLM Response Generation")

    # Initialize components
    embedding_service = CLIPEmbeddingService(
        model_name="openai/clip-vit-base-patch32",
        device="mps"
    )
    embedding_service.load_model()

    vector_db = FAISSVectorDB(dimension=512)
    vector_db.load_index("models/test/product_index.faiss", "models/test/product_metadata.pkl")

    prompt_builder = PromptBuilder()

    # Test different free LLM options
    free_llm_configs = [
        {
            "name": "Free Fallback (No API needed)",
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
            "name": "Ollama (if running locally)",
            "provider": "ollama",
            "model": "llama2",
            "token": None
        }
    ]

    # Test queries
    test_queries = [
        "wireless bluetooth headphones under $100",
        "educational toys for 5 year old kids",
        "kitchen appliances for small apartment"
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

            # Test with one query
            query = test_queries[0]
            logger.info(f"\n🔍 Query: '{query}'")

            # Get search results
            query_embedding = embedding_service.get_text_embedding(query)
            metadata_list, distances = vector_db.search(query_embedding, k=3)

            # Build prompt
            prompt = prompt_builder.build_rag_prompt(
                query=query,
                retrieved_metadata=metadata_list
            )

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

    # Demonstrate complete RAG with free LLM
    logger.info("\n" + "="*60)
    logger.info("🎯 COMPLETE RAG PIPELINE WITH FREE LLM")
    logger.info("="*60)

    # Use the best available free option
    llm_client = create_llm_client(
        provider="free",
        model_name="gpt2",
        api_token=None
    )

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n--- Test {i}: '{query}' ---")

        # Search
        query_embedding = embedding_service.get_text_embedding(query)
        metadata_list, distances = vector_db.search(query_embedding, k=5)

        logger.info(f"🔍 Found {len(metadata_list)} results")

        # Show top 3 results
        for j, (metadata, distance) in enumerate(zip(metadata_list[:3], distances[:3]), 1):
            logger.info(f"   {j}. Distance: {distance:.3f}")
            logger.info(f"      Product: {metadata.combined_text[:80]}...")

        # Generate response
        prompt = prompt_builder.build_rag_prompt(
            query=query,
            retrieved_metadata=metadata_list
        )

        start_time = time.time()
        response = llm_client.generate_response(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        generation_time = time.time() - start_time

        print(f"\n🤖 AI Response ({generation_time:.2f}s):")
        print(f"   {response}")

    logger.info("\n🎉 Free LLM testing completed!")

    # Show available options summary
    logger.info("\n" + "="*60)
    logger.info("📋 FREE LLM OPTIONS SUMMARY")
    logger.info("="*60)

    print("\n🆓 Available Free LLM Options:")
    print("1. **Fallback Responses** (Always available)")
    print("   - No setup required")
    print("   - Basic templated responses")
    print("   - Good for testing and demos")

    print("\n2. **Local Transformers** (pip install transformers)")
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
    print("")
    print("   # For Ollama:")
    print("   LLM_PROVIDER=ollama")
    print("   LLM_MODEL_NAME=llama2")
    print("")
    print("   # For Transformers:")
    print("   LLM_PROVIDER=free")
    print("   LLM_MODEL_NAME=gpt2")

if __name__ == "__main__":
    test_free_llm_responses()