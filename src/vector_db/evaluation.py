"""
Evaluation utilities for vector database performance.
"""
from typing import Callable, List, Optional, Dict, Any
import numpy as np
from loguru import logger

from .faiss_service import FAISSVectorDB


class VectorDBEvaluator:
    """Evaluator for vector database recall and performance metrics."""

    def __init__(self, vector_db: FAISSVectorDB):
        """
        Initialize evaluator with vector database.

        Args:
            vector_db: Vector database instance
        """
        self.vector_db = vector_db

    def recall_at_k(
        self,
        k: int,
        embeddings: np.ndarray,
        ground_truth_fn: Callable[[int], set],
        exclude_self: bool = True
    ) -> float:
        """
        Calculate recall@k metric.

        Args:
            k: Number of top results to consider
            embeddings: All embeddings used for queries
            ground_truth_fn: Function that returns ground truth indices for a query index
            exclude_self: Whether to exclude the query itself from results

        Returns:
            Recall@k score
        """
        hits = 0
        total = 0

        for i in range(len(embeddings)):
            # Get ground truth for this query
            gt = ground_truth_fn(i)
            if not gt:  # Skip if no ground truth
                continue

            # Perform search
            query_emb = embeddings[i].reshape(1, -1)
            metadata_results, _ = self.vector_db.search(query_emb, k + (1 if exclude_self else 0))

            # Convert metadata results back to indices
            # Note: This assumes metadata has an index or we track it separately
            retrieved_indices = set()
            for j, meta in enumerate(metadata_results):
                if exclude_self and j == 0:  # Skip first result if it's the query itself
                    continue
                # In a real implementation, you'd need to map metadata back to indices
                # For now, we'll assume the indices match the order
                if j < len(metadata_results):
                    retrieved_indices.add(j)

            # Check if any ground truth items were retrieved
            if retrieved_indices & gt:
                hits += 1
            total += 1

        return hits / total if total > 0 else 0.0

    def evaluate_search_latency(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate search latency performance.

        Args:
            query_embeddings: Sample query embeddings
            k: Number of results per query
            num_iterations: Number of iterations for timing

        Returns:
            Dictionary with latency statistics
        """
        import time

        latencies = []

        for i in range(num_iterations):
            # Select random query
            query_idx = np.random.randint(0, len(query_embeddings))
            query_emb = query_embeddings[query_idx].reshape(1, -1)

            start_time = time.time()
            self.vector_db.search(query_emb, k)
            end_time = time.time()

            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

        latencies = np.array(latencies)

        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "std_latency_ms": float(np.std(latencies))
        }

    def evaluate_memory_usage(self) -> Dict[str, Any]:
        """
        Evaluate memory usage of the vector database.

        Returns:
            Dictionary with memory statistics
        """
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        stats = self.vector_db.get_stats()

        return {
            "process_memory_mb": memory_info.rss / (1024 * 1024),
            "index_memory_mb": stats.memory_usage_mb,
            "total_vectors": stats.total_vectors,
            "memory_per_vector_kb": (memory_info.rss / (1024 * stats.total_vectors)) if stats.total_vectors > 0 else 0
        }

    def compare_index_types(
        self,
        embeddings: np.ndarray,
        metadata: List[Any],
        query_embeddings: np.ndarray,
        index_configs: List[Dict[str, Any]],
        k: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare different index configurations.

        Args:
            embeddings: All embeddings to index
            metadata: Corresponding metadata
            query_embeddings: Query embeddings for testing
            index_configs: List of index configurations to test
            k: Number of results per query

        Returns:
            Comparison results for each configuration
        """
        results = {}

        for config in index_configs:
            config_name = f"{config['index_type']}_{config.get('metric', 'l2')}"
            logger.info(f"Testing configuration: {config_name}")

            try:
                # Create and populate index
                vector_db = FAISSVectorDB(
                    dimension=embeddings.shape[1],
                    **config
                )

                # Measure indexing time
                import time
                start_time = time.time()
                vector_db.add_vectors(embeddings, metadata)
                indexing_time = time.time() - start_time

                # Evaluate search performance
                evaluator = VectorDBEvaluator(vector_db)
                latency_stats = evaluator.evaluate_search_latency(query_embeddings, k)
                memory_stats = evaluator.evaluate_memory_usage()

                results[config_name] = {
                    "config": config,
                    "indexing_time_seconds": indexing_time,
                    "latency_stats": latency_stats,
                    "memory_stats": memory_stats,
                    "index_stats": vector_db.get_stats().dict()
                }

            except Exception as e:
                logger.error(f"Failed to test configuration {config_name}: {e}")
                results[config_name] = {"error": str(e)}

        return results

    def run_comprehensive_evaluation(
        self,
        embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        ground_truth_fn: Optional[Callable[[int], set]] = None,
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of the vector database.

        Args:
            embeddings: All embeddings
            query_embeddings: Query embeddings
            ground_truth_fn: Ground truth function for recall calculation
            k_values: List of k values to evaluate

        Returns:
            Comprehensive evaluation results
        """
        results = {
            "index_stats": self.vector_db.get_stats().dict(),
            "memory_stats": self.evaluate_memory_usage()
        }

        # Evaluate recall if ground truth function provided
        if ground_truth_fn:
            recall_scores = {}
            for k in k_values:
                recall_scores[f"recall@{k}"] = self.recall_at_k(
                    k, embeddings, ground_truth_fn
                )
            results["recall_scores"] = recall_scores

        # Evaluate latency for different k values
        latency_results = {}
        for k in k_values:
            latency_results[f"k={k}"] = self.evaluate_search_latency(
                query_embeddings, k
            )
        results["latency_by_k"] = latency_results

        return results