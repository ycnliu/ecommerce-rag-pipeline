"""
Custom exceptions for the e-commerce RAG pipeline.
"""


class EcommerceRAGError(Exception):
    """Base exception for e-commerce RAG pipeline."""
    pass


class ConfigurationError(EcommerceRAGError):
    """Configuration-related errors."""
    pass


class DataProcessingError(EcommerceRAGError):
    """Data processing errors."""
    pass


class EmbeddingError(EcommerceRAGError):
    """Embedding generation errors."""
    pass


class ModelLoadError(EcommerceRAGError):
    """Model loading errors."""
    pass


class VectorDBError(EcommerceRAGError):
    """Vector database errors."""
    pass


class LLMError(EcommerceRAGError):
    """LLM client errors."""
    pass


class RAGError(EcommerceRAGError):
    """RAG pipeline errors."""
    pass


class APIError(EcommerceRAGError):
    """API-related errors."""
    pass


class ValidationError(EcommerceRAGError):
    """Data validation errors."""
    pass