"""
Pytest configuration and fixtures for the e-commerce RAG pipeline.
"""
import os
import tempfile
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from pathlib import Path

from src.utils.config import Config
from src.embedding.service import CLIPEmbeddingService
from src.vector_db.faiss_service import FAISSVectorDB
from src.rag.llm_client import BaseLLMClient
from src.rag.rag_pipeline import RAGPipeline
from src.data.models import ProductMetadata
from src.data.processor import DataProcessor


@pytest.fixture(scope="session")
def test_config():
    """Test configuration with temporary paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config(
            debug=True,
            log_level="DEBUG",
            model_cache_dir=temp_dir,
            faiss_index_path=os.path.join(temp_dir, "test_index.faiss"),
            faiss_metadata_path=os.path.join(temp_dir, "test_metadata.pkl"),
            embeddings_cache_path=os.path.join(temp_dir, "test_embeddings.npy"),
            log_file_path=os.path.join(temp_dir, "test.log"),
            llm_api_token="test_token"
        )
        yield config


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    service = Mock(spec=CLIPEmbeddingService)
    service.get_text_embedding.return_value = np.random.rand(512)
    service.get_image_embedding.return_value = np.random.rand(512)
    service.get_multimodal_embedding.return_value = np.random.rand(512)
    service.get_model_info.return_value = {
        "model_name": "test_model",
        "model_loaded": True,
        "embedding_dim": 512
    }
    return service


@pytest.fixture
def mock_vector_db():
    """Mock vector database for testing."""
    db = Mock(spec=FAISSVectorDB)
    db.search.return_value = ([], [])
    db.get_stats.return_value = Mock(
        total_vectors=100,
        dimension=512,
        index_type="flat"
    )
    return db


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = Mock(spec=BaseLLMClient)
    client.generate_response.return_value = "Test response"
    client.get_model_info.return_value = {
        "model_name": "test_llm",
        "provider": "test"
    }
    return client


@pytest.fixture
def sample_product_metadata():
    """Sample product metadata for testing."""
    return [
        ProductMetadata(
            image_url="https://example.com/image1.jpg",
            product_url="https://example.com/product1",
            variants_products_link="Not available",
            shipping_weight="1 lb",
            product_dimensions="10x5x2 inches",
            product_specification="Test specification",
            technical_details="Test technical details",
            is_amazon_seller="Y",
            combined_text="Product Name: Test Product 1 | Category: Electronics | Price: $99.99"
        ),
        ProductMetadata(
            image_url="https://example.com/image2.jpg",
            product_url="https://example.com/product2",
            variants_products_link="Not available",
            shipping_weight="2 lb",
            product_dimensions="8x4x1 inches",
            product_specification="Test specification 2",
            technical_details="Test technical details 2",
            is_amazon_seller="N",
            combined_text="Product Name: Test Product 2 | Category: Electronics | Price: $149.99"
        )
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return np.random.rand(10, 512).astype('float32')


@pytest.fixture
def mock_rag_pipeline(mock_embedding_service, mock_vector_db, mock_llm_client):
    """Mock RAG pipeline for testing."""
    return RAGPipeline(
        embedding_service=mock_embedding_service,
        vector_db=mock_vector_db,
        llm_client=mock_llm_client
    )


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return """Uniq Id,Product Name,Category,Selling Price,Model Number,About Product,Product Specification,Technical Details,Shipping Weight,Product Dimensions,Image,Variants,Product Url,Is Amazon Seller
1,Test Product 1,Electronics,$99.99,TP1,Great product,Spec 1,Tech 1,1 lb,10x5x2,http://img1.jpg,Not available,http://prod1.com,Y
2,Test Product 2,Electronics,$149.99,TP2,Another product,Spec 2,Tech 2,2 lb,8x4x1,http://img2.jpg,Not available,http://prod2.com,N"""


@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_csv_data)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def data_processor():
    """Data processor instance for testing."""
    return DataProcessor()


# API testing fixtures
@pytest.fixture
def api_client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    # Override dependencies with mocks
    def override_get_rag_pipeline():
        return mock_rag_pipeline

    app.dependency_overrides[get_rag_pipeline] = override_get_rag_pipeline

    with TestClient(app) as client:
        yield client

    # Clean up overrides
    app.dependency_overrides.clear()


# Test data fixtures
@pytest.fixture
def test_queries():
    """Sample test queries."""
    return [
        "What are good wireless headphones?",
        "Show me gaming laptops under $1000",
        "I need a smartphone with good camera",
        "What kitchen appliances do you recommend?"
    ]


@pytest.fixture
def test_reference_responses():
    """Sample reference responses for evaluation."""
    return [
        "Based on the available products, I recommend these wireless headphones...",
        "Here are some gaming laptops under $1000 that offer good performance...",
        "For smartphones with good cameras, consider these options...",
        "I recommend these kitchen appliances for your needs..."
    ]


# Performance testing fixtures
@pytest.fixture
def performance_embeddings():
    """Large set of embeddings for performance testing."""
    return np.random.rand(1000, 512).astype('float32')


@pytest.fixture
def performance_metadata(performance_embeddings):
    """Large set of metadata for performance testing."""
    metadata_list = []
    for i in range(len(performance_embeddings)):
        metadata_list.append(
            ProductMetadata(
                image_url=f"https://example.com/image{i}.jpg",
                product_url=f"https://example.com/product{i}",
                variants_products_link="Not available",
                shipping_weight=f"{i % 10 + 1} lb",
                product_dimensions=f"{i % 20 + 5}x{i % 15 + 3}x{i % 10 + 1} inches",
                product_specification=f"Specification for product {i}",
                technical_details=f"Technical details for product {i}",
                is_amazon_seller="Y" if i % 2 == 0 else "N",
                combined_text=f"Product Name: Test Product {i} | Category: Electronics | Price: ${(i % 500) + 50}.99"
            )
        )
    return metadata_list


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Cleanup temporary files after each test."""
    yield
    # Add any necessary cleanup here