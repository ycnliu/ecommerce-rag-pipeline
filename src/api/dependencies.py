"""
Dependency injection for FastAPI endpoints.
"""
import os
from functools import lru_cache
from typing import Optional

from ..utils.config import Config
from ..utils.exceptions import ConfigurationError
from ..embedding.service import CLIPEmbeddingService
from ..vector_db.faiss_service import FAISSVectorDB
from ..rag.llm_client import create_llm_client
from ..rag.rag_pipeline import RAGPipeline
from ..data.processor import DataProcessor


@lru_cache()
def get_config() -> Config:
    """Get application configuration (cached)."""
    return Config()


@lru_cache()
def get_embedding_service() -> CLIPEmbeddingService:
    """Get embedding service instance (cached)."""
    config = get_config()

    service = CLIPEmbeddingService(
        model_name=config.clip_model_name,
        device=config.device,
        cache_dir=config.model_cache_dir
    )

    # Load model
    service.load_model()
    return service


@lru_cache()
def get_vector_db() -> FAISSVectorDB:
    """Get vector database instance (cached)."""
    config = get_config()

    # Initialize vector DB
    vector_db = FAISSVectorDB(
        dimension=config.embedding_dimension,
        index_type=config.faiss_index_type,
        metric=config.faiss_metric
    )

    # Load existing index if available
    if config.faiss_index_path and os.path.exists(config.faiss_index_path):
        vector_db.load_index(
            config.faiss_index_path,
            config.faiss_metadata_path
        )

    return vector_db


@lru_cache()
def get_llm_client():
    """Get LLM client instance (cached)."""
    config = get_config()

    if not config.llm_api_token:
        raise ConfigurationError("LLM API token not configured")

    return create_llm_client(
        provider=config.llm_provider,
        model_name=config.llm_model_name,
        api_token=config.llm_api_token
    )


@lru_cache()
def get_rag_pipeline() -> RAGPipeline:
    """Get RAG pipeline instance (cached)."""
    embedding_service = get_embedding_service()
    vector_db = get_vector_db()
    llm_client = get_llm_client()

    return RAGPipeline(
        embedding_service=embedding_service,
        vector_db=vector_db,
        llm_client=llm_client
    )


def get_data_processor() -> DataProcessor:
    """Get data processor instance."""
    return DataProcessor()