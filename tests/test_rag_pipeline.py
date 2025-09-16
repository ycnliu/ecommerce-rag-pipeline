"""
Tests for RAG pipeline functionality.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.rag.rag_pipeline import RAGPipeline
from src.rag.evaluation import RAGEvaluator
from src.rag.prompt_builder import PromptBuilder
from src.data.models import QueryRequest, ProductMetadata, SearchResult
from src.utils.exceptions import RAGError


class TestRAGPipeline:
    """Test cases for RAGPipeline class."""

    def test_initialization(self, mock_embedding_service, mock_vector_db, mock_llm_client):
        """Test RAG pipeline initialization."""
        pipeline = RAGPipeline(
            embedding_service=mock_embedding_service,
            vector_db=mock_vector_db,
            llm_client=mock_llm_client
        )

        assert pipeline.embedding_service == mock_embedding_service
        assert pipeline.vector_db == mock_vector_db
        assert pipeline.llm_client == mock_llm_client
        assert pipeline.prompt_builder is not None

    def test_query_text_only(self, mock_rag_pipeline, sample_product_metadata):
        """Test querying with text only."""
        # Setup mocks
        mock_rag_pipeline.embedding_service.get_multimodal_embedding.return_value = np.random.rand(512)
        mock_rag_pipeline.vector_db.search.return_value = (sample_product_metadata[:2], [0.1, 0.2])
        mock_rag_pipeline.llm_client.generate_response.return_value = "Generated response"

        response = mock_rag_pipeline.query(
            text_query="wireless headphones",
            k=5,
            generate_response=True
        )

        assert response.query == "wireless headphones"
        assert len(response.results) == 2
        assert response.generated_response == "Generated response"
        assert response.processing_time > 0

    def test_query_image_only(self, mock_rag_pipeline, sample_product_metadata):
        """Test querying with image only."""
        # Setup mocks
        mock_rag_pipeline.embedding_service.get_multimodal_embedding.return_value = np.random.rand(512)
        mock_rag_pipeline.vector_db.search.return_value = (sample_product_metadata[:1], [0.1])
        mock_rag_pipeline.llm_client.generate_response.return_value = "Generated response"

        image_data = b"fake_image_data"
        response = mock_rag_pipeline.query(
            image_query=image_data,
            k=3,
            generate_response=True
        )

        assert response.query == "Image query"
        assert len(response.results) == 1
        assert response.generated_response == "Generated response"

    def test_query_multimodal(self, mock_rag_pipeline, sample_product_metadata):
        """Test querying with both text and image."""
        # Setup mocks
        mock_rag_pipeline.embedding_service.get_multimodal_embedding.return_value = np.random.rand(512)
        mock_rag_pipeline.vector_db.search.return_value = (sample_product_metadata[:3], [0.1, 0.2, 0.3])
        mock_rag_pipeline.llm_client.generate_response.return_value = "Generated response"

        response = mock_rag_pipeline.query(
            text_query="gaming laptop",
            image_query=b"fake_image_data",
            k=5,
            generate_response=True
        )

        assert response.query == "gaming laptop"
        assert len(response.results) == 3
        assert response.generated_response == "Generated response"

    def test_query_no_input(self, mock_rag_pipeline):
        """Test querying without any input."""
        with pytest.raises(ValueError, match="At least one of text_query or image_query must be provided"):
            mock_rag_pipeline.query()

    def test_query_no_results(self, mock_rag_pipeline):
        """Test querying when no results are found."""
        # Setup mocks to return empty results
        mock_rag_pipeline.embedding_service.get_multimodal_embedding.return_value = np.random.rand(512)
        mock_rag_pipeline.vector_db.search.return_value = ([], [])

        response = mock_rag_pipeline.query(
            text_query="nonexistent product",
            k=5,
            generate_response=True
        )

        assert response.query == "nonexistent product"
        assert len(response.results) == 0
        assert "couldn't find any relevant products" in response.generated_response

    def test_query_with_reranking(self, mock_rag_pipeline, sample_product_metadata):
        """Test querying with reranking enabled."""
        # Setup mocks
        mock_rag_pipeline.embedding_service.get_multimodal_embedding.return_value = np.random.rand(512)
        mock_rag_pipeline.vector_db.search.return_value = (sample_product_metadata[:2], [0.1, 0.2])
        mock_rag_pipeline.llm_client.generate_response.return_value = "Generated response"

        response = mock_rag_pipeline.query(
            text_query="wireless headphones",
            k=5,
            rerank=True,
            generate_response=True
        )

        assert response.query == "wireless headphones"
        assert len(response.results) == 2

    def test_query_without_response_generation(self, mock_rag_pipeline, sample_product_metadata):
        """Test querying without generating LLM response."""
        # Setup mocks
        mock_rag_pipeline.embedding_service.get_multimodal_embedding.return_value = np.random.rand(512)
        mock_rag_pipeline.vector_db.search.return_value = (sample_product_metadata[:1], [0.1])

        response = mock_rag_pipeline.query(
            text_query="wireless headphones",
            k=5,
            generate_response=False
        )

        assert response.query == "wireless headphones"
        assert len(response.results) == 1
        assert response.generated_response is None

    def test_batch_query(self, mock_rag_pipeline, sample_product_metadata):
        """Test batch query processing."""
        # Setup mocks
        mock_rag_pipeline.embedding_service.get_multimodal_embedding.return_value = np.random.rand(512)
        mock_rag_pipeline.vector_db.search.return_value = (sample_product_metadata[:1], [0.1])
        mock_rag_pipeline.llm_client.generate_response.return_value = "Generated response"

        queries = [
            QueryRequest(text_query="wireless headphones", k=5),
            QueryRequest(text_query="gaming laptop", k=3)
        ]

        responses = mock_rag_pipeline.batch_query(queries, generate_responses=True)

        assert len(responses) == 2
        assert all(response.generated_response for response in responses)

    def test_get_similar_products(self, mock_rag_pipeline, sample_product_metadata):
        """Test finding similar products."""
        # Setup mocks
        mock_rag_pipeline.embedding_service.get_text_embedding.return_value = np.random.rand(512)
        mock_rag_pipeline.vector_db.search.return_value = (sample_product_metadata[:2], [0.1, 0.2])

        product = sample_product_metadata[0]
        similar_products = mock_rag_pipeline.get_similar_products(product, k=5)

        assert len(similar_products) <= 2
        assert all(isinstance(result, SearchResult) for result in similar_products)

    def test_explain_recommendation(self, mock_rag_pipeline, sample_product_metadata):
        """Test recommendation explanation generation."""
        # Setup mocks
        mock_rag_pipeline.llm_client.generate_response.return_value = "Explanation of recommendations"

        explanation = mock_rag_pipeline.explain_recommendation(
            "wireless headphones",
            sample_product_metadata[:2],
            explanation_style="detailed"
        )

        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_get_pipeline_stats(self, mock_rag_pipeline):
        """Test getting pipeline statistics."""
        # Setup mocks
        mock_rag_pipeline.embedding_service.get_model_info.return_value = {"model_name": "test"}
        mock_rag_pipeline.vector_db.get_stats.return_value = Mock(
            dict=lambda: {"total_vectors": 100}
        )
        mock_rag_pipeline.llm_client.get_model_info.return_value = {"model_name": "test_llm"}

        stats = mock_rag_pipeline.get_pipeline_stats()

        assert "embedding_service" in stats
        assert "vector_db" in stats
        assert "llm_client" in stats
        assert "prompt_builder" in stats

    def test_error_handling(self, mock_rag_pipeline):
        """Test error handling in RAG pipeline."""
        # Setup mock to raise exception
        mock_rag_pipeline.embedding_service.get_multimodal_embedding.side_effect = Exception("Test error")

        with pytest.raises(RAGError):
            mock_rag_pipeline.query(text_query="test query")


class TestPromptBuilder:
    """Test cases for PromptBuilder class."""

    def test_initialization(self):
        """Test prompt builder initialization."""
        builder = PromptBuilder(max_chars=5000)

        assert builder.max_chars == 5000
        assert builder.few_shot_examples is not None

    def test_build_context_from_metadata(self, sample_product_metadata):
        """Test building context from metadata."""
        builder = PromptBuilder()

        context = builder.build_context_from_metadata(sample_product_metadata[:2])

        assert isinstance(context, str)
        assert len(context) > 0
        assert "Product:" in context

    def test_build_context_max_items(self, sample_product_metadata):
        """Test building context with max items limit."""
        builder = PromptBuilder()

        context = builder.build_context_from_metadata(
            sample_product_metadata * 10,  # Many items
            max_items=2
        )

        # Should only include 2 items
        sections = context.split("\n\n")
        assert len(sections) <= 2

    def test_build_rag_prompt(self, sample_product_metadata):
        """Test building complete RAG prompt."""
        builder = PromptBuilder()

        prompt = builder.build_rag_prompt(
            "wireless headphones",
            sample_product_metadata[:2],
            include_examples=True
        )

        assert isinstance(prompt, str)
        assert "wireless headphones" in prompt
        assert "Product:" in prompt

    def test_build_rag_prompt_no_examples(self, sample_product_metadata):
        """Test building RAG prompt without examples."""
        builder = PromptBuilder()

        prompt = builder.build_rag_prompt(
            "wireless headphones",
            sample_product_metadata[:1],
            include_examples=False
        )

        assert isinstance(prompt, str)
        assert "Q:" not in prompt  # No few-shot examples

    def test_build_evaluation_prompt(self, sample_product_metadata):
        """Test building evaluation prompt."""
        builder = PromptBuilder()

        prompt = builder.build_evaluation_prompt(
            "wireless headphones",
            sample_product_metadata[:1],
            ground_truth="Test ground truth"
        )

        assert isinstance(prompt, str)
        assert "Evaluation Task" in prompt
        assert "Ground Truth Reference" in prompt

    def test_build_comparison_prompt(self, sample_product_metadata):
        """Test building comparison prompt."""
        builder = PromptBuilder()

        product_groups = [
            sample_product_metadata[:1],
            sample_product_metadata[1:2]
        ]
        group_labels = ["Group A", "Group B"]

        prompt = builder.build_comparison_prompt(
            "compare products",
            product_groups,
            group_labels
        )

        assert isinstance(prompt, str)
        assert "Group A" in prompt
        assert "Group B" in prompt
        assert "comparison" in prompt.lower()

    def test_extract_key_features(self, sample_product_metadata):
        """Test extracting key features from metadata."""
        builder = PromptBuilder()

        features = builder.extract_key_features(sample_product_metadata[0])

        assert isinstance(features, dict)
        # Should extract price if present
        if "Price:" in sample_product_metadata[0].combined_text:
            assert "price" in features


class TestRAGEvaluator:
    """Test cases for RAGEvaluator class."""

    def test_initialization(self, mock_rag_pipeline):
        """Test evaluator initialization."""
        evaluator = RAGEvaluator(mock_rag_pipeline)

        assert evaluator.rag_pipeline == mock_rag_pipeline
        assert evaluator.rouge_scorer is not None

    def test_evaluate_bleu(self):
        """Test BLEU score evaluation."""
        evaluator = RAGEvaluator(Mock())

        reference = "The wireless headphones have great sound quality"
        generated = "These wireless headphones offer excellent sound"

        score = evaluator.evaluate_bleu(reference, generated)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_evaluate_rouge(self):
        """Test ROUGE score evaluation."""
        evaluator = RAGEvaluator(Mock())

        reference = "The wireless headphones have great sound quality"
        generated = "These wireless headphones offer excellent sound"

        score = evaluator.evaluate_rouge(reference, generated)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_evaluate_response_quality(self, mock_rag_pipeline, test_queries, test_reference_responses):
        """Test response quality evaluation."""
        # Setup mock to return query responses
        from src.data.models import QueryResponse
        mock_rag_pipeline.query.return_value = QueryResponse(
            query="test",
            results=[],
            generated_response="test response",
            processing_time=0.1
        )

        evaluator = RAGEvaluator(mock_rag_pipeline)

        results = evaluator.evaluate_response_quality(
            test_queries[:2],
            test_reference_responses[:2]
        )

        assert "bleu_mean" in results
        assert "rouge_mean" in results
        assert "bleu_scores" in results
        assert "rouge_scores" in results

    def test_evaluate_latency(self, mock_rag_pipeline, test_queries):
        """Test latency evaluation."""
        # Setup mock to return query responses
        from src.data.models import QueryResponse
        mock_rag_pipeline.query.return_value = QueryResponse(
            query="test",
            results=[],
            generated_response="test response",
            processing_time=0.05  # 50ms
        )

        evaluator = RAGEvaluator(mock_rag_pipeline)

        results = evaluator.evaluate_latency(
            test_queries[:2],
            num_iterations=3,
            k=5
        )

        expected_keys = [
            "mean_latency_ms", "median_latency_ms", "p95_latency_ms",
            "p99_latency_ms", "min_latency_ms", "max_latency_ms", "std_latency_ms"
        ]

        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], float)