"""
FastAPI application for the e-commerce RAG pipeline.
"""
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from ..data.models import (
    QueryRequest,
    QueryResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthCheck,
    IndexStats
)
from ..rag.rag_pipeline import RAGPipeline
from ..utils.config import Config
from ..utils.exceptions import RAGError, EmbeddingError, VectorDBError
from .dependencies import get_rag_pipeline, get_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting up E-commerce RAG API")

    # Load configuration and initialize services
    try:
        config = Config()
        # Services will be initialized via dependency injection
        logger.info("API startup completed successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    finally:
        logger.info("Shutting down E-commerce RAG API")


# Create FastAPI app
app = FastAPI(
    title="E-commerce RAG Pipeline API",
    description="Multimodal product search and recommendation API using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "E-commerce RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check(
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    config: Config = Depends(get_config)
):
    """Health check endpoint."""
    try:
        # Check services status
        services = {}

        # Check embedding service
        try:
            model_info = rag_pipeline.embedding_service.get_model_info()
            services["embedding_service"] = "healthy" if model_info["model_loaded"] else "not_loaded"
        except Exception as e:
            services["embedding_service"] = f"error: {str(e)}"

        # Check vector database
        try:
            db_stats = rag_pipeline.vector_db.get_stats()
            services["vector_db"] = "healthy" if db_stats.total_vectors > 0 else "empty"
        except Exception as e:
            services["vector_db"] = f"error: {str(e)}"

        # Check LLM client
        try:
            llm_info = rag_pipeline.llm_client.get_model_info()
            services["llm_client"] = "healthy"
        except Exception as e:
            services["llm_client"] = f"error: {str(e)}"

        # Get index stats
        index_stats = None
        try:
            index_stats = rag_pipeline.vector_db.get_stats()
        except Exception:
            pass

        return HealthCheck(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            services=services,
            index_stats=index_stats
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/search", response_model=QueryResponse)
async def search_products(
    request: QueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Search for products using text and/or image queries.

    Supports multimodal search with text descriptions, image uploads,
    and optional response generation.
    """
    try:
        logger.info(f"Search request: text='{request.text_query}', k={request.k}")

        response = rag_pipeline.query(
            text_query=request.text_query,
            image_query=request.image_query,
            k=request.k,
            rerank=request.rerank,
            generate_response=True
        )

        logger.info(f"Search completed in {response.processing_time:.2f}s")
        return response

    except (RAGError, EmbeddingError, VectorDBError) as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/search/image", response_model=QueryResponse)
async def search_by_image(
    file: UploadFile = File(...),
    text_query: Optional[str] = None,
    k: int = 5,
    rerank: bool = False,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Search for products using an uploaded image.

    Upload an image file and optionally add text description
    for multimodal search.
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image data
        image_data = await file.read()

        logger.info(f"Image search: file={file.filename}, text='{text_query}', k={k}")

        response = rag_pipeline.query(
            text_query=text_query,
            image_query=image_data,
            k=k,
            rerank=rerank,
            generate_response=True
        )

        logger.info(f"Image search completed in {response.processing_time:.2f}s")
        return response

    except (RAGError, EmbeddingError, VectorDBError) as e:
        logger.error(f"Image search error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected image search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/embeddings/text", response_model=EmbeddingResponse)
async def get_text_embedding(
    request: EmbeddingRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Generate embedding for text input."""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")

        embedding = rag_pipeline.embedding_service.get_text_embedding(request.text)

        return EmbeddingResponse(
            embedding=embedding.tolist(),
            embedding_type="text"
        )

    except EmbeddingError as e:
        logger.error(f"Text embedding error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected embedding error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/embeddings/image", response_model=EmbeddingResponse)
async def get_image_embedding(
    file: UploadFile = File(...),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Generate embedding for uploaded image."""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_data = await file.read()
        embedding = rag_pipeline.embedding_service.get_image_embedding(image_data)

        return EmbeddingResponse(
            embedding=embedding.tolist(),
            embedding_type="image"
        )

    except EmbeddingError as e:
        logger.error(f"Image embedding error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected embedding error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/stats", response_model=Dict[str, Any])
async def get_pipeline_stats(
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Get pipeline statistics and information."""
    try:
        stats = rag_pipeline.get_pipeline_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pipeline stats")


@app.post("/batch/search", response_model=List[QueryResponse])
async def batch_search(
    requests: List[QueryRequest],
    background_tasks: BackgroundTasks,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Process multiple search queries in batch.

    Useful for bulk processing of search requests.
    """
    try:
        if len(requests) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large (max 100)")

        logger.info(f"Processing batch of {len(requests)} queries")

        responses = rag_pipeline.batch_query(requests)

        # Log batch completion in background
        background_tasks.add_task(
            logger.info,
            f"Batch processing completed: {len(responses)} responses"
        )

        return responses

    except (RAGError, EmbeddingError, VectorDBError) as e:
        logger.error(f"Batch search error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected batch search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/similar/{product_url:path}", response_model=List[Dict[str, Any]])
async def get_similar_products(
    product_url: str,
    k: int = 5,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Find products similar to a given product URL."""
    try:
        # This would need to be implemented to find the product by URL
        # and then get similar products
        # For now, return a placeholder response

        return [{"message": "Similar products endpoint not fully implemented"}]

    except Exception as e:
        logger.error(f"Similar products error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )