"""
Data models for the e-commerce RAG pipeline.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, HttpUrl


class ProductMetadata(BaseModel):
    """Product metadata structure."""
    image_url: str
    product_url: HttpUrl
    variants_products_link: Optional[str] = None
    shipping_weight: Optional[str] = None
    product_dimensions: Optional[str] = None
    product_specification: Optional[str] = None
    technical_details: Optional[str] = None
    is_amazon_seller: str
    combined_text: str


class QueryRequest(BaseModel):
    """Request model for search queries."""
    text_query: Optional[str] = None
    image_query: Optional[str] = None
    k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    rerank: bool = Field(default=False, description="Whether to rerank results")


class SearchResult(BaseModel):
    """Individual search result."""
    score: float
    metadata: ProductMetadata


class QueryResponse(BaseModel):
    """Response model for search queries."""
    query: str
    results: List[SearchResult]
    generated_response: Optional[str] = None
    processing_time: float


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    text: Optional[str] = None
    image_url: Optional[HttpUrl] = None


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embedding: List[float]
    embedding_type: str


class IndexStats(BaseModel):
    """Statistics about the vector index."""
    total_vectors: int
    dimension: int
    index_type: str
    memory_usage_mb: Optional[float] = None


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    services: Dict[str, str]
    index_stats: Optional[IndexStats] = None