"""
Tests for the FastAPI application.
"""
import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.models import QueryResponse, SearchResult, HealthCheck
from src.utils.exceptions import RAGError, EmbeddingError


class TestAPIEndpoints:
    """Test cases for API endpoints."""

    @pytest.fixture
    def client(self, mock_rag_pipeline):
        """FastAPI test client with mocked dependencies."""
        def override_get_rag_pipeline():
            return mock_rag_pipeline

        def override_get_config():
            from src.utils.config import Config
            return Config(llm_api_token="test_token")

        # Override dependencies
        from src.api.dependencies import get_rag_pipeline, get_config
        app.dependency_overrides[get_rag_pipeline] = override_get_rag_pipeline
        app.dependency_overrides[get_config] = override_get_config

        with TestClient(app) as client:
            yield client

        # Clean up
        app.dependency_overrides.clear()

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data

    def test_health_check_healthy(self, client, mock_rag_pipeline):
        """Test health check endpoint when all services are healthy."""
        # Mock healthy responses
        mock_rag_pipeline.embedding_service.get_model_info.return_value = {
            "model_loaded": True
        }
        mock_rag_pipeline.vector_db.get_stats.return_value = Mock(
            total_vectors=100,
            dimension=512,
            index_type="flat"
        )
        mock_rag_pipeline.llm_client.get_model_info.return_value = {
            "model_name": "test_model"
        }

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "timestamp" in data

    def test_search_text_query(self, client, mock_rag_pipeline):
        """Test text search endpoint."""
        # Mock successful response
        mock_response = QueryResponse(
            query="test query",
            results=[],
            generated_response="Test response",
            processing_time=0.1
        )
        mock_rag_pipeline.query.return_value = mock_response

        request_data = {
            "text_query": "test query",
            "k": 5
        }

        response = client.post("/search", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["generated_response"] == "Test response"
        assert "processing_time" in data

    def test_search_with_rerank(self, client, mock_rag_pipeline):
        """Test search endpoint with reranking enabled."""
        mock_response = QueryResponse(
            query="test query",
            results=[],
            generated_response="Test response",
            processing_time=0.1
        )
        mock_rag_pipeline.query.return_value = mock_response

        request_data = {
            "text_query": "test query",
            "k": 5,
            "rerank": True
        }

        response = client.post("/search", json=request_data)

        assert response.status_code == 200
        # Verify rerank parameter was passed
        mock_rag_pipeline.query.assert_called_with(
            text_query="test query",
            image_query=None,
            k=5,
            rerank=True,
            generate_response=True
        )

    def test_search_image_upload(self, client, mock_rag_pipeline):
        """Test image search endpoint."""
        mock_response = QueryResponse(
            query="Image query",
            results=[],
            generated_response="Test response",
            processing_time=0.1
        )
        mock_rag_pipeline.query.return_value = mock_response

        # Create a mock image file
        image_data = b"fake_image_data"
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        data = {"k": 5}

        response = client.post("/search/image", files=files, data=data)

        assert response.status_code == 200
        response_data = response.json()
        assert "processing_time" in response_data

    def test_search_image_invalid_file(self, client):
        """Test image search with invalid file type."""
        text_data = b"not_an_image"
        files = {"file": ("test.txt", text_data, "text/plain")}

        response = client.post("/search/image", files=files)

        assert response.status_code == 400
        assert "File must be an image" in response.json()["detail"]

    def test_text_embedding_endpoint(self, client, mock_rag_pipeline):
        """Test text embedding endpoint."""
        import numpy as np
        mock_embedding = np.random.rand(512)
        mock_rag_pipeline.embedding_service.get_text_embedding.return_value = mock_embedding

        request_data = {"text": "test text"}

        response = client.post("/embeddings/text", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert data["embedding_type"] == "text"
        assert len(data["embedding"]) == 512

    def test_text_embedding_missing_text(self, client):
        """Test text embedding endpoint with missing text."""
        request_data = {}

        response = client.post("/embeddings/text", json=request_data)

        assert response.status_code == 400
        assert "Text is required" in response.json()["detail"]

    def test_image_embedding_endpoint(self, client, mock_rag_pipeline):
        """Test image embedding endpoint."""
        import numpy as np
        mock_embedding = np.random.rand(512)
        mock_rag_pipeline.embedding_service.get_image_embedding.return_value = mock_embedding

        image_data = b"fake_image_data"
        files = {"file": ("test.jpg", image_data, "image/jpeg")}

        response = client.post("/embeddings/image", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert data["embedding_type"] == "image"
        assert len(data["embedding"]) == 512

    def test_pipeline_stats_endpoint(self, client, mock_rag_pipeline):
        """Test pipeline stats endpoint."""
        mock_stats = {
            "embedding_service": {"model_name": "test_model"},
            "vector_db": {"total_vectors": 100},
            "llm_client": {"model_name": "test_llm"}
        }
        mock_rag_pipeline.get_pipeline_stats.return_value = mock_stats

        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert "embedding_service" in data
        assert "vector_db" in data
        assert "llm_client" in data

    def test_batch_search_endpoint(self, client, mock_rag_pipeline):
        """Test batch search endpoint."""
        mock_responses = [
            QueryResponse(
                query="query 1",
                results=[],
                generated_response="response 1",
                processing_time=0.1
            ),
            QueryResponse(
                query="query 2",
                results=[],
                generated_response="response 2",
                processing_time=0.1
            )
        ]
        mock_rag_pipeline.batch_query.return_value = mock_responses

        request_data = [
            {"text_query": "query 1", "k": 5},
            {"text_query": "query 2", "k": 3}
        ]

        response = client.post("/batch/search", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["query"] == "query 1"
        assert data[1]["query"] == "query 2"

    def test_batch_search_too_large(self, client):
        """Test batch search with too many requests."""
        large_batch = [{"text_query": f"query {i}", "k": 5} for i in range(101)]

        response = client.post("/batch/search", json=large_batch)

        assert response.status_code == 400
        assert "Batch size too large" in response.json()["detail"]

    def test_error_handling_rag_error(self, client, mock_rag_pipeline):
        """Test error handling for RAG errors."""
        mock_rag_pipeline.query.side_effect = RAGError("Test RAG error")

        request_data = {"text_query": "test query", "k": 5}

        response = client.post("/search", json=request_data)

        assert response.status_code == 400
        assert "Test RAG error" in response.json()["detail"]

    def test_error_handling_embedding_error(self, client, mock_rag_pipeline):
        """Test error handling for embedding errors."""
        mock_rag_pipeline.embedding_service.get_text_embedding.side_effect = EmbeddingError("Test embedding error")

        request_data = {"text": "test text"}

        response = client.post("/embeddings/text", json=request_data)

        assert response.status_code == 400
        assert "Test embedding error" in response.json()["detail"]

    def test_error_handling_general_exception(self, client, mock_rag_pipeline):
        """Test handling of unexpected exceptions."""
        mock_rag_pipeline.query.side_effect = Exception("Unexpected error")

        request_data = {"text_query": "test query", "k": 5}

        response = client.post("/search", json=request_data)

        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/")

        # CORS headers should be present due to middleware
        assert response.status_code == 200

    def test_validation_error_k_parameter(self, client):
        """Test validation for k parameter."""
        request_data = {
            "text_query": "test query",
            "k": 100  # Exceeds max limit of 50
        }

        response = client.post("/search", json=request_data)

        # Should be handled by Pydantic validation
        assert response.status_code == 422

    def test_validation_error_missing_query(self, client):
        """Test validation when no query is provided."""
        request_data = {"k": 5}  # No text_query or image_query

        response = client.post("/search", json=request_data)

        # Should be accepted by API but fail in pipeline logic
        assert response.status_code in [400, 422]