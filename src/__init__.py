"""
E-commerce RAG Pipeline - Industry-level multimodal product search and recommendation system.

This package provides a complete RAG (Retrieval-Augmented Generation) pipeline for
e-commerce applications, featuring:

- Multimodal embeddings using CLIP
- Fast similarity search with FAISS
- LLM-powered response generation
- RESTful API with FastAPI
- Comprehensive evaluation framework
- Production-ready deployment

Example usage:
    >>> from src.rag.rag_pipeline import RAGPipeline
    >>> from src.utils.config import Config
    >>>
    >>> config = Config()
    >>> # Initialize and use pipeline...
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "Industry-level multimodal e-commerce RAG pipeline"

# Package metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]