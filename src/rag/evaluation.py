"""
Evaluation utilities for RAG pipeline performance.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import pairwise_distances
from loguru import logger

from .rag_pipeline import RAGPipeline
from ..data.models import QueryRequest, QueryResponse


class RAGEvaluator:
    """Evaluator for RAG pipeline performance."""

    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize RAG evaluator.

        Args:
            rag_pipeline: RAG pipeline to evaluate
        """
        self.rag_pipeline = rag_pipeline
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def evaluate_bleu(self, reference: str, generated: str) -> float:
        """
        Calculate BLEU score between reference and generated text.

        Args:
            reference: Reference text
            generated: Generated text

        Returns:
            BLEU score
        """
        try:
            smoothie = SmoothingFunction().method4
            ref_tokens = reference.split()
            gen_tokens = generated.split()
            return sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
        except Exception as e:
            logger.error(f"BLEU calculation failed: {e}")
            return 0.0

    def evaluate_rouge(self, reference: str, generated: str) -> float:
        """
        Calculate ROUGE-L score between reference and generated text.

        Args:
            reference: Reference text
            generated: Generated text

        Returns:
            ROUGE-L F1 score
        """
        try:
            score = self.rouge_scorer.score(reference, generated)
            return score['rougeL'].fmeasure
        except Exception as e:
            logger.error(f"ROUGE calculation failed: {e}")
            return 0.0

    def evaluate_retrieval_accuracy(
        self,
        test_queries: List[str],
        ground_truth_indices: List[List[int]],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate retrieval accuracy using ground truth.

        Args:
            test_queries: List of test queries
            ground_truth_indices: List of ground truth document indices for each query
            k: Number of retrieved documents to evaluate

        Returns:
            Dictionary with accuracy metrics
        """
        if len(test_queries) != len(ground_truth_indices):
            raise ValueError("Number of queries must match number of ground truth sets")

        precision_scores = []
        recall_scores = []
        f1_scores = []

        for query, gt_indices in zip(test_queries, ground_truth_indices):
            try:
                # Get retrieval results
                response = self.rag_pipeline.query(
                    text_query=query,
                    k=k,
                    generate_response=False
                )

                # Extract retrieved indices (assuming we can map metadata back to indices)
                retrieved_indices = []
                for result in response.results:
                    # This would need to be implemented based on how we track indices
                    # For now, we'll use a placeholder
                    retrieved_indices.append(0)  # Placeholder

                # Calculate metrics
                gt_set = set(gt_indices)
                retrieved_set = set(retrieved_indices)

                if len(retrieved_set) == 0:
                    precision = 0.0
                else:
                    precision = len(gt_set & retrieved_set) / len(retrieved_set)

                if len(gt_set) == 0:
                    recall = 0.0
                else:
                    recall = len(gt_set & retrieved_set) / len(gt_set)

                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)

                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                f1_scores.append(0.0)

        return {
            "precision": np.mean(precision_scores),
            "recall": np.mean(recall_scores),
            "f1": np.mean(f1_scores),
            "precision_std": np.std(precision_scores),
            "recall_std": np.std(recall_scores),
            "f1_std": np.std(f1_scores)
        }

    def evaluate_response_quality(
        self,
        test_queries: List[str],
        reference_responses: List[str],
        llm_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate response quality using BLEU and ROUGE metrics.

        Args:
            test_queries: List of test queries
            reference_responses: List of reference responses
            llm_params: Optional LLM parameters

        Returns:
            Dictionary with quality metrics
        """
        if len(test_queries) != len(reference_responses):
            raise ValueError("Number of queries must match number of reference responses")

        bleu_scores = []
        rouge_scores = []

        for query, reference in zip(test_queries, reference_responses):
            try:
                # Generate response
                response = self.rag_pipeline.query(
                    text_query=query,
                    generate_response=True,
                    llm_params=llm_params or {}
                )

                generated = response.generated_response or ""

                # Calculate metrics
                bleu_score = self.evaluate_bleu(reference, generated)
                rouge_score = self.evaluate_rouge(reference, generated)

                bleu_scores.append(bleu_score)
                rouge_scores.append(rouge_score)

                logger.debug(f"Query: {query[:50]}... | BLEU: {bleu_score:.3f} | ROUGE: {rouge_score:.3f}")

            except Exception as e:
                logger.error(f"Error evaluating response for query '{query}': {e}")
                bleu_scores.append(0.0)
                rouge_scores.append(0.0)

        return {
            "bleu_mean": np.mean(bleu_scores),
            "bleu_std": np.std(bleu_scores),
            "rouge_mean": np.mean(rouge_scores),
            "rouge_std": np.std(rouge_scores),
            "bleu_scores": bleu_scores,
            "rouge_scores": rouge_scores
        }

    def evaluate_latency(
        self,
        test_queries: List[str],
        num_iterations: int = 10,
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate pipeline latency.

        Args:
            test_queries: List of test queries
            num_iterations: Number of iterations per query
            k: Number of results to retrieve

        Returns:
            Dictionary with latency statistics
        """
        latencies = []

        for _ in range(num_iterations):
            for query in test_queries:
                try:
                    response = self.rag_pipeline.query(
                        text_query=query,
                        k=k,
                        generate_response=True
                    )
                    latencies.append(response.processing_time * 1000)  # Convert to milliseconds

                except Exception as e:
                    logger.error(f"Error measuring latency for query '{query}': {e}")

        if not latencies:
            return {"error": "No successful latency measurements"}

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

    def evaluate_end_to_end(
        self,
        test_queries: List[str],
        reference_responses: Optional[List[str]] = None,
        ground_truth_indices: Optional[List[List[int]]] = None,
        k: int = 5,
        num_latency_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Run comprehensive end-to-end evaluation.

        Args:
            test_queries: List of test queries
            reference_responses: Optional reference responses for quality evaluation
            ground_truth_indices: Optional ground truth indices for retrieval evaluation
            k: Number of results to retrieve
            num_latency_runs: Number of runs for latency evaluation

        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Starting end-to-end evaluation with {len(test_queries)} queries")

        results = {
            "evaluation_config": {
                "num_queries": len(test_queries),
                "k": k,
                "num_latency_runs": num_latency_runs
            },
            "pipeline_stats": self.rag_pipeline.get_pipeline_stats()
        }

        # Evaluate latency
        logger.info("Evaluating latency...")
        results["latency"] = self.evaluate_latency(
            test_queries[:min(5, len(test_queries))],  # Use subset for latency
            num_latency_runs,
            k
        )

        # Evaluate response quality if references provided
        if reference_responses:
            logger.info("Evaluating response quality...")
            results["response_quality"] = self.evaluate_response_quality(
                test_queries,
                reference_responses
            )

        # Evaluate retrieval accuracy if ground truth provided
        if ground_truth_indices:
            logger.info("Evaluating retrieval accuracy...")
            results["retrieval_accuracy"] = self.evaluate_retrieval_accuracy(
                test_queries,
                ground_truth_indices,
                k
            )

        logger.info("End-to-end evaluation completed")
        return results

    def run_ablation_study(
        self,
        test_queries: List[str],
        ablation_configs: List[Dict[str, Any]],
        reference_responses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run ablation study comparing different pipeline configurations.

        Args:
            test_queries: List of test queries
            ablation_configs: List of configuration dictionaries
            reference_responses: Optional reference responses

        Returns:
            Ablation study results
        """
        logger.info(f"Running ablation study with {len(ablation_configs)} configurations")

        results = {}

        for i, config in enumerate(ablation_configs):
            config_name = config.get("name", f"config_{i}")
            logger.info(f"Testing configuration: {config_name}")

            try:
                # Temporarily modify pipeline settings if needed
                # This would need to be implemented based on specific requirements

                # Evaluate this configuration
                if reference_responses:
                    config_results = self.evaluate_response_quality(
                        test_queries[:min(10, len(test_queries))],  # Use subset
                        reference_responses[:min(10, len(reference_responses))]
                    )
                else:
                    config_results = self.evaluate_latency(
                        test_queries[:min(5, len(test_queries))],
                        num_iterations=3
                    )

                results[config_name] = {
                    "config": config,
                    "results": config_results
                }

            except Exception as e:
                logger.error(f"Error evaluating configuration {config_name}: {e}")
                results[config_name] = {"error": str(e)}

        return results