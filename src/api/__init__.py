"""
FastAPI application for the e-commerce RAG pipeline.
"""
from .main import app
from .dependencies import get_rag_pipeline, get_config

__all__ = ["app", "get_rag_pipeline", "get_config"]