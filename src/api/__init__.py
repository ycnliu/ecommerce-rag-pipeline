"""
FastAPI application for the e-commerce RAG pipeline.
"""

from .dependencies import get_config, get_rag_pipeline
from .main import app

__all__ = ["app", "get_rag_pipeline", "get_config"]
