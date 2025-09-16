#!/usr/bin/env python3
"""
Demo script for the E-commerce RAG Pipeline.
"""
import os
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config import Config
from src.utils.logging import setup_logging
from src.data.processor import DataProcessor
from src.embedding.service import CLIPEmbeddingService
from src.vector_db.faiss_service import FAISSVectorDB
from src.rag.llm_client import create_llm_client
from src.rag.rag_pipeline import RAGPipeline
from loguru import logger


def create_sample_data():
    """Create sample product data for demo."""
    sample_data = [
        {
            "Uniq Id": "1",
            "Product Name": "Sony WH-1000XM4 Wireless Headphones",
            "Category": "Electronics",
            "Selling Price": "$348.00",
            "Model Number": "WH1000XM4",
            "About Product": "Industry-leading noise canceling with Dual Noise Sensor technology. Up to 30-hour battery life with quick charge.",
            "Product Specification": "Bluetooth 5.0, 40mm drivers, Touch controls",
            "Technical Details": "Frequency Response: 4Hz-40kHz, Weight: 254g",
            "Shipping Weight": "1.5 pounds",
            "Product Dimensions": "9.94 x 7.27 x 3.03 inches",
            "Image": "https://example.com/sony-wh1000xm4.jpg",
            "Variants": "Black, Silver, Blue",
            "Product Url": "https://example.com/products/sony-wh1000xm4",
            "Is Amazon Seller": "Y"
        },
        {
            "Uniq Id": "2",
            "Product Name": "Apple MacBook Pro 16-inch",
            "Category": "Computers",
            "Selling Price": "$2,399.00",
            "Model Number": "MK1E3LL/A",
            "About Product": "Supercharged by M1 Pro chip for groundbreaking performance. 16-inch Liquid Retina XDR display.",
            "Product Specification": "M1 Pro chip, 16GB RAM, 512GB SSD",
            "Technical Details": "Display: 16.2-inch 3456x2234, Touch ID",
            "Shipping Weight": "4.7 pounds",
            "Product Dimensions": "14.01 x 9.77 x 0.66 inches",
            "Image": "https://example.com/macbook-pro-16.jpg",
            "Variants": "Silver, Space Gray",
            "Product Url": "https://example.com/products/macbook-pro-16",
            "Is Amazon Seller": "N"
        },
        {
            "Uniq Id": "3",
            "Product Name": "Logitech MX Master 3 Wireless Mouse",
            "Category": "Electronics",
            "Selling Price": "$99.99",
            "Model Number": "910-005620",
            "About Product": "Advanced wireless mouse designed for power users. Hyper-fast scrolling and app-specific customization.",
            "Product Specification": "Darkfield sensor, 7 buttons, USB-C charging",
            "Technical Details": "DPI: up to 4000, Battery: up to 70 days",
            "Shipping Weight": "0.3 pounds",
            "Product Dimensions": "4.92 x 3.31 x 2.01 inches",
            "Image": "https://example.com/logitech-mx-master-3.jpg",
            "Variants": "Graphite, Mid Grey, Rose",
            "Product Url": "https://example.com/products/logitech-mx-master-3",
            "Is Amazon Seller": "Y"
        },
        {
            "Uniq Id": "4",
            "Product Name": "Nintendo Switch Console",
            "Category": "Video Games",
            "Selling Price": "$299.99",
            "Model Number": "HAC-001(-01)",
            "About Product": "Hybrid gaming console that can be used as a home console or portable device. Play anywhere, anytime.",
            "Product Specification": "Custom NVIDIA Tegra X1, 32GB internal storage",
            "Technical Details": "Display: 6.2-inch capacitive touch screen, WiFi, Bluetooth",
            "Shipping Weight": "2.8 pounds",
            "Product Dimensions": "9.4 x 4.0 x 0.55 inches",
            "Image": "https://example.com/nintendo-switch.jpg",
            "Variants": "Neon Blue/Red, Gray",
            "Product Url": "https://example.com/products/nintendo-switch",
            "Is Amazon Seller": "Y"
        },
        {
            "Uniq Id": "5",
            "Product Name": "Samsung 65-inch 4K Smart TV",
            "Category": "Electronics",
            "Selling Price": "$1,297.99",
            "Model Number": "UN65TU8000FXZA",
            "About Product": "Crystal UHD 4K Smart TV with built-in Alexa. HDR and Crystal Display for enhanced picture quality.",
            "Product Specification": "65-inch 4K UHD, Smart TV with Tizen OS",
            "Technical Details": "HDR10+, 3 HDMI ports, 2 USB ports, WiFi",
            "Shipping Weight": "60.0 pounds",
            "Product Dimensions": "57.1 x 32.7 x 2.4 inches",
            "Image": "https://example.com/samsung-65-4k-tv.jpg",
            "Variants": "Not available",
            "Product Url": "https://example.com/products/samsung-65-4k-tv",
            "Is Amazon Seller": "Y"
        }
    ]

    return sample_data


def setup_demo_environment():
    """Setup demo environment with sample data."""
    print("🔧 Setting up demo environment...")

    # Create demo directory
    demo_dir = Path("demo_data")
    demo_dir.mkdir(exist_ok=True)

    # Create sample CSV
    sample_data = create_sample_data()

    import pandas as pd
    df = pd.DataFrame(sample_data)
    csv_path = demo_dir / "sample_products.csv"
    df.to_csv(csv_path, index=False)

    print(f"✅ Created sample data: {csv_path}")
    return csv_path


def run_demo():
    """Run the complete demo."""
    print("🚀 Starting E-commerce RAG Pipeline Demo")
    print("=" * 50)

    try:
        # Setup configuration for demo
        config = Config(
            debug=True,
            log_level="INFO",
            clip_model_name="openai/clip-vit-base-patch32",
            faiss_index_type="flat",
            llm_provider="huggingface",
            llm_model_name="microsoft/DialoGPT-medium",  # Smaller model for demo
            llm_api_token="demo_token"  # Will use mock for demo
        )

        setup_logging(config)

        # Setup demo data
        csv_path = setup_demo_environment()

        # Step 1: Data Processing
        print("\n📊 Step 1: Processing data...")
        processor = DataProcessor()
        df, metadata_list = processor.process_full_pipeline(str(csv_path))
        print(f"✅ Processed {len(metadata_list)} products")

        # Step 2: Initialize Embedding Service
        print("\n🧠 Step 2: Initializing embedding service...")
        embedding_service = CLIPEmbeddingService(
            model_name=config.clip_model_name,
            device="cpu"  # Force CPU for demo
        )
        embedding_service.load_model()
        print("✅ Embedding service loaded")

        # Step 3: Generate Embeddings
        print("\n🔢 Step 3: Generating embeddings...")
        texts = [meta.combined_text for meta in metadata_list]
        embeddings = embedding_service.batch_text_embeddings(texts, batch_size=16)
        print(f"✅ Generated embeddings: {embeddings.shape}")

        # Step 4: Build Vector Database
        print("\n🗄️  Step 4: Building vector database...")
        vector_db = FAISSVectorDB(
            dimension=embeddings.shape[1],
            index_type="flat"
        )
        vector_db.add_vectors(embeddings, metadata_list)
        print(f"✅ Vector database built with {vector_db.index.ntotal} vectors")

        # Step 5: Setup Mock LLM (for demo without API key)
        print("\n🤖 Step 5: Setting up mock LLM...")

        class MockLLMClient:
            def generate_response(self, prompt, **kwargs):
                return "Based on the available products, I found several relevant options that match your query. These products offer good features and value for the price range."

            def get_model_info(self):
                return {"model_name": "mock_llm", "provider": "demo"}

        llm_client = MockLLMClient()
        print("✅ Mock LLM client ready")

        # Step 6: Create RAG Pipeline
        print("\n🔧 Step 6: Creating RAG pipeline...")
        pipeline = RAGPipeline(
            embedding_service=embedding_service,
            vector_db=vector_db,
            llm_client=llm_client
        )
        print("✅ RAG pipeline ready")

        # Step 7: Demo Queries
        print("\n🔍 Step 7: Running demo queries...")
        demo_queries = [
            "wireless headphones with noise canceling",
            "gaming console for Nintendo games",
            "laptop for professional work",
            "4K TV for home entertainment",
            "wireless mouse for productivity"
        ]

        for i, query in enumerate(demo_queries, 1):
            print(f"\n--- Demo Query {i}: {query} ---")

            start_time = time.time()
            response = pipeline.query(
                text_query=query,
                k=3,
                generate_response=True
            )
            processing_time = time.time() - start_time

            print(f"🤖 AI Response: {response.generated_response}")
            print(f"📊 Found {len(response.results)} results:")

            for j, result in enumerate(response.results, 1):
                print(f"  {j}. {result.metadata.combined_text[:80]}...")
                print(f"     Score: {result.score:.3f}")

            print(f"⏱️  Processing time: {processing_time:.2f}s")

        # Step 8: Demo Statistics
        print(f"\n📈 Demo Statistics:")
        stats = pipeline.get_pipeline_stats()
        print(f"   Total products indexed: {stats['vector_db']['total_vectors']}")
        print(f"   Embedding dimension: {stats['vector_db']['dimension']}")
        print(f"   Index type: {stats['vector_db']['index_type']}")

        print("\n✅ Demo completed successfully!")
        print("\n🎯 Next Steps:")
        print("1. Set up your own data CSV file")
        print("2. Configure your LLM API token in .env")
        print("3. Use the CLI: python -m src.cli --help")
        print("4. Start the API server: python -m src.cli serve")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = run_demo()
    exit(0 if success else 1)