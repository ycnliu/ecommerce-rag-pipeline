"""
Main RAG (Retrieval-Augmented Generation) pipeline.
"""
import time
from typing import List, Optional, Tuple, Union, Dict, Any
from PIL import Image
import numpy as np
from loguru import logger

from ..embedding.service import CLIPEmbeddingService
from ..vector_db.faiss_service import FAISSVectorDB
from ..data.models import ProductMetadata, QueryRequest, QueryResponse, SearchResult
from ..data.processor import TextProcessor
from .llm_client import BaseLLMClient
from .prompt_builder import PromptBuilder
from ..utils.exceptions import RAGError


class RAGPipeline:
    """Complete RAG pipeline for e-commerce product search and response generation."""

    def __init__(
        self,
        embedding_service: CLIPEmbeddingService,
        vector_db: FAISSVectorDB,
        llm_client: BaseLLMClient,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        """
        Initialize RAG pipeline.

        Args:
            embedding_service: Embedding service for encoding queries
            vector_db: Vector database for similarity search
            llm_client: LLM client for response generation
            prompt_builder: Optional custom prompt builder
        """
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()

        logger.info("RAG pipeline initialized successfully")

    def query(
        self,
        text_query: Optional[str] = None,
        image_query: Optional[Union[str, Image.Image, bytes]] = None,
        k: int = 5,
        rerank: bool = False,
        generate_response: bool = True,
        fusion_method: str = "average",
        llm_params: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """
        Process a query through the complete RAG pipeline.

        Args:
            text_query: Optional text query
            image_query: Optional image query (URL, PIL Image, or bytes)
            k: Number of results to retrieve
            rerank: Whether to rerank results using text similarity
            generate_response: Whether to generate LLM response
            fusion_method: Method to combine text and image embeddings
            llm_params: Optional parameters for LLM generation

        Returns:
            Query response with results and optional generated response

        Raises:
            RAGError: If pipeline execution fails
        """
        if not text_query and not image_query:
            raise ValueError("At least one of text_query or image_query must be provided")

        start_time = time.time()
        display_query = text_query or "Image query"

        try:
            logger.info(f"Processing query: {display_query[:100]}...")

            # Step 1: Generate query embedding
            query_embedding = self.embedding_service.get_multimodal_embedding(
                text=text_query,
                image=image_query,
                fusion_method=fusion_method
            )

            # Step 2: Retrieve similar items
            retrieved_metadata, distances = self.vector_db.search(
                query_embedding,
                k=k,
                return_distances=True
            )

            if not retrieved_metadata:
                logger.warning("No results found for query")
                return QueryResponse(
                    query=display_query,
                    results=[],
                    generated_response="I'm sorry, but I couldn't find any relevant products for your query.",
                    processing_time=time.time() - start_time
                )

            # Step 3: Optional reranking
            if rerank and text_query:
                logger.info("Reranking results based on text similarity")
                # Convert metadata to dict format for reranking
                metadata_dicts = [meta.dict() for meta in retrieved_metadata]
                reranked_dicts = TextProcessor.rerank_by_text_similarity(text_query, metadata_dicts)

                # Convert back to ProductMetadata
                retrieved_metadata = [ProductMetadata(**meta_dict) for meta_dict in reranked_dicts]
                # Note: distances are no longer accurate after reranking
                distances = [0.0] * len(retrieved_metadata)

            # Step 4: Create search results
            search_results = []
            for i, (metadata, distance) in enumerate(zip(retrieved_metadata, distances)):
                search_results.append(SearchResult(
                    score=float(distance),
                    metadata=metadata
                ))

            # Step 5: Generate response if requested
            generated_response = None
            if generate_response:
                logger.info("Generating LLM response")
                generated_response = self._generate_response(
                    text_query or "Describe these products",
                    retrieved_metadata,
                    llm_params or {}
                )

            processing_time = time.time() - start_time
            logger.info(f"Query processed in {processing_time:.2f} seconds")

            return QueryResponse(
                query=display_query,
                results=search_results,
                generated_response=generated_response,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            raise RAGError(f"RAG pipeline failed: {e}") from e

    def _generate_response(
        self,
        query: str,
        retrieved_metadata: List[ProductMetadata],
        llm_params: Dict[str, Any]
    ) -> str:
        """
        Generate LLM response from query and retrieved context.

        Args:
            query: User query
            retrieved_metadata: Retrieved product metadata
            llm_params: LLM generation parameters

        Returns:
            Generated response
        """
        try:
            # Build prompt
            prompt = self.prompt_builder.build_rag_prompt(query, retrieved_metadata)

            # Set default LLM parameters
            default_params = {
                "max_tokens": 300,
                "temperature": 0.1,
                "stop_sequences": ["\nUser question:", "\nQ:"]
            }
            default_params.update(llm_params)

            # Generate response
            response = self.llm_client.generate_response(prompt, **default_params)

            return response

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."

    def batch_query(
        self,
        queries: List[QueryRequest],
        generate_responses: bool = True
    ) -> List[QueryResponse]:
        """
        Process multiple queries in batch.

        Args:
            queries: List of query requests
            generate_responses: Whether to generate LLM responses

        Returns:
            List of query responses
        """
        logger.info(f"Processing batch of {len(queries)} queries")

        results = []
        for i, query_request in enumerate(queries):
            try:
                logger.info(f"Processing query {i+1}/{len(queries)}")

                response = self.query(
                    text_query=query_request.text_query,
                    image_query=query_request.image_query,
                    k=query_request.k,
                    rerank=query_request.rerank,
                    generate_response=generate_responses
                )
                results.append(response)

            except Exception as e:
                logger.error(f"Failed to process query {i+1}: {e}")
                # Add error response
                error_response = QueryResponse(
                    query=query_request.text_query or "Image query",
                    results=[],
                    generated_response=f"Error processing query: {str(e)}",
                    processing_time=0.0
                )
                results.append(error_response)

        logger.info(f"Batch processing completed: {len(results)} responses generated")
        return results

    def get_similar_products(
        self,
        product_metadata: ProductMetadata,
        k: int = 5,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find products similar to a given product.

        Args:
            product_metadata: Product to find similar items for
            k: Number of similar products to return
            exclude_self: Whether to exclude the product itself

        Returns:
            List of similar products
        """
        try:
            # Generate embedding for the product
            embedding = self.embedding_service.get_text_embedding(
                product_metadata.combined_text
            )

            # Search for similar products
            retrieved_metadata, distances = self.vector_db.search(
                embedding,
                k=k + (1 if exclude_self else 0),
                return_distances=True
            )

            # Filter out self if needed
            results = []
            for metadata, distance in zip(retrieved_metadata, distances):
                if exclude_self and metadata.product_url == product_metadata.product_url:
                    continue
                results.append(SearchResult(score=float(distance), metadata=metadata))

            return results[:k]

        except Exception as e:
            logger.error(f"Failed to find similar products: {e}")
            raise RAGError(f"Similar product search failed: {e}") from e

    def explain_recommendation(
        self,
        query: str,
        recommended_products: List[ProductMetadata],
        explanation_style: str = "detailed"
    ) -> str:
        """
        Generate explanation for why products were recommended.

        Args:
            query: Original query
            recommended_products: Recommended products
            explanation_style: Style of explanation ("detailed", "brief")

        Returns:
            Explanation text
        """
        try:
            if explanation_style == "detailed":
                custom_instructions = """You are an AI shopping assistant. Explain why these products were recommended for the user's query.

                Focus on:
                1. How each product relates to the user's needs
                2. Key features that make them suitable
                3. Any trade-offs or considerations

                Be helpful and informative."""
            else:
                custom_instructions = """Briefly explain why these products match the user's query. Keep it concise."""

            prompt = self.prompt_builder.build_rag_prompt(
                f"Explain why these products were recommended for: {query}",
                recommended_products,
                include_examples=False,
                custom_instructions=custom_instructions
            )

            explanation = self.llm_client.generate_response(
                prompt,
                max_tokens=200 if explanation_style == "brief" else 400
            )

            return explanation

        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return "Unable to generate explanation for recommendations."

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline components."""
        return {
            "embedding_service": self.embedding_service.get_model_info(),
            "vector_db": self.vector_db.get_stats().dict(),
            "llm_client": self.llm_client.get_model_info(),
            "prompt_builder": {
                "max_chars": self.prompt_builder.max_chars
            }
        }