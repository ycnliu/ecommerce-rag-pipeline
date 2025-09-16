"""
Tests for vector database functionality.
"""
import pytest
import numpy as np
import tempfile
import os

from src.vector_db.faiss_service import FAISSVectorDB
from src.vector_db.evaluation import VectorDBEvaluator
from src.data.models import ProductMetadata
from src.utils.exceptions import VectorDBError


class TestFAISSVectorDB:
    """Test cases for FAISSVectorDB class."""

    def test_initialization(self):
        """Test vector database initialization."""
        db = FAISSVectorDB(dimension=128, index_type="flat")

        assert db.dimension == 128
        assert db.index_type == "flat"
        assert db.index is not None
        assert len(db.metadata) == 0

    def test_initialization_with_different_index_types(self):
        """Test initialization with different index types."""
        # Test flat index
        db_flat = FAISSVectorDB(dimension=128, index_type="flat")
        assert db_flat.index_type == "flat"

        # Test IVF index
        db_ivf = FAISSVectorDB(dimension=128, index_type="ivf", nlist=10)
        assert db_ivf.index_type == "ivf"

        # Test HNSW index
        db_hnsw = FAISSVectorDB(dimension=128, index_type="hnsw")
        assert db_hnsw.index_type == "hnsw"

    def test_add_vectors(self, sample_embeddings, sample_product_metadata):
        """Test adding vectors to the database."""
        db = FAISSVectorDB(dimension=512)

        # Use subset of data that matches
        embeddings = sample_embeddings[:2]
        metadata = sample_product_metadata[:2]

        db.add_vectors(embeddings, metadata)

        assert db.index.ntotal == 2
        assert len(db.metadata) == 2

    def test_add_vectors_dimension_mismatch(self, sample_product_metadata):
        """Test adding vectors with wrong dimension."""
        db = FAISSVectorDB(dimension=128)
        wrong_embeddings = np.random.rand(2, 256).astype('float32')

        with pytest.raises(ValueError, match="Embedding dimension"):
            db.add_vectors(wrong_embeddings, sample_product_metadata[:2])

    def test_add_vectors_metadata_mismatch(self, sample_embeddings, sample_product_metadata):
        """Test adding vectors with mismatched metadata count."""
        db = FAISSVectorDB(dimension=512)

        with pytest.raises(ValueError, match="Number of embeddings must match"):
            db.add_vectors(sample_embeddings[:2], sample_product_metadata[:1])

    def test_search_empty_index(self):
        """Test searching in empty index."""
        db = FAISSVectorDB(dimension=512)
        query = np.random.rand(512).astype('float32')

        results, distances = db.search(query, k=5)

        assert len(results) == 0
        assert len(distances) == 0

    def test_search_with_results(self, sample_embeddings, sample_product_metadata):
        """Test searching with populated index."""
        db = FAISSVectorDB(dimension=512)
        embeddings = sample_embeddings[:2]
        metadata = sample_product_metadata[:2]

        db.add_vectors(embeddings, metadata)

        # Search using first embedding
        query = embeddings[0]
        results, distances = db.search(query, k=2)

        assert len(results) == 2
        assert len(distances) == 2
        assert all(isinstance(result, ProductMetadata) for result in results)

    def test_batch_search(self, sample_embeddings, sample_product_metadata):
        """Test batch search functionality."""
        db = FAISSVectorDB(dimension=512)
        embeddings = sample_embeddings[:5]
        metadata = sample_product_metadata * 3  # Repeat to get 6 items

        db.add_vectors(embeddings, metadata[:5])

        # Batch search with multiple queries
        queries = embeddings[:2]
        results = db.batch_search(queries, k=3)

        assert len(results) == 2
        assert all(len(result[0]) <= 3 for result in results)  # metadata
        assert all(len(result[1]) <= 3 for result in results)  # distances

    def test_save_and_load_index(self, sample_embeddings, sample_product_metadata):
        """Test saving and loading index."""
        db = FAISSVectorDB(dimension=512)
        embeddings = sample_embeddings[:2]
        metadata = sample_product_metadata[:2]

        db.add_vectors(embeddings, metadata)

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "test_index.faiss")
            metadata_path = os.path.join(temp_dir, "test_metadata.pkl")

            # Save index
            db.save_index(index_path, metadata_path)

            # Create new database and load
            db2 = FAISSVectorDB(dimension=512)
            db2.load_index(index_path, metadata_path)

            assert db2.index.ntotal == 2
            assert len(db2.metadata) == 2

    def test_get_stats(self, sample_embeddings, sample_product_metadata):
        """Test getting index statistics."""
        db = FAISSVectorDB(dimension=512)
        embeddings = sample_embeddings[:3]
        metadata = sample_product_metadata[:2] + [sample_product_metadata[0]]  # Repeat to get 3

        db.add_vectors(embeddings, metadata)

        stats = db.get_stats()

        assert stats.total_vectors == 3
        assert stats.dimension == 512
        assert stats.index_type == "flat"

    def test_clear_index(self, sample_embeddings, sample_product_metadata):
        """Test clearing the index."""
        db = FAISSVectorDB(dimension=512)
        embeddings = sample_embeddings[:2]
        metadata = sample_product_metadata[:2]

        db.add_vectors(embeddings, metadata)
        assert db.index.ntotal == 2

        db.clear()
        assert db.index.ntotal == 0
        assert len(db.metadata) == 0

    def test_ivf_training(self):
        """Test IVF index training."""
        db = FAISSVectorDB(dimension=128, index_type="ivf", nlist=10)

        # Generate enough data for training
        embeddings = np.random.rand(50, 128).astype('float32')
        metadata = [
            ProductMetadata(
                image_url=f"https://example.com/image{i}.jpg",
                product_url=f"https://example.com/product{i}",
                variants_products_link="Not available",
                shipping_weight="1 lb",
                product_dimensions="10x5x2",
                product_specification="Test spec",
                technical_details="Test details",
                is_amazon_seller="Y",
                combined_text=f"Product {i}"
            ) for i in range(50)
        ]

        db.add_vectors(embeddings, metadata, train_if_needed=True)

        assert db.is_trained
        assert db.index.ntotal == 50

    def test_remove_vectors_flat_index(self):
        """Test removing vectors from flat index."""
        db = FAISSVectorDB(dimension=128, index_type="flat")

        embeddings = np.random.rand(5, 128).astype('float32')
        metadata = [
            ProductMetadata(
                image_url=f"https://example.com/image{i}.jpg",
                product_url=f"https://example.com/product{i}",
                variants_products_link="Not available",
                shipping_weight="1 lb",
                product_dimensions="10x5x2",
                product_specification="Test spec",
                technical_details="Test details",
                is_amazon_seller="Y",
                combined_text=f"Product {i}"
            ) for i in range(5)
        ]

        db.add_vectors(embeddings, metadata)
        assert db.index.ntotal == 5

        # Remove some vectors
        db.remove_vectors([1, 3])
        assert db.index.ntotal == 3
        assert len(db.metadata) == 3

    def test_remove_vectors_unsupported_index(self):
        """Test that removing vectors from unsupported index raises error."""
        db = FAISSVectorDB(dimension=128, index_type="ivf", nlist=10)

        with pytest.raises(VectorDBError, match="Vector removal only supported"):
            db.remove_vectors([0])


class TestVectorDBEvaluator:
    """Test cases for VectorDBEvaluator class."""

    def test_initialization(self):
        """Test evaluator initialization."""
        db = FAISSVectorDB(dimension=128)
        evaluator = VectorDBEvaluator(db)

        assert evaluator.vector_db == db

    def test_evaluate_search_latency(self, sample_embeddings):
        """Test search latency evaluation."""
        db = FAISSVectorDB(dimension=512)
        embeddings = sample_embeddings[:10]
        metadata = [
            ProductMetadata(
                image_url=f"https://example.com/image{i}.jpg",
                product_url=f"https://example.com/product{i}",
                variants_products_link="Not available",
                shipping_weight="1 lb",
                product_dimensions="10x5x2",
                product_specification="Test spec",
                technical_details="Test details",
                is_amazon_seller="Y",
                combined_text=f"Product {i}"
            ) for i in range(10)
        ]

        db.add_vectors(embeddings, metadata)

        evaluator = VectorDBEvaluator(db)
        latency_stats = evaluator.evaluate_search_latency(
            embeddings[:3],
            k=5,
            num_iterations=5
        )

        expected_keys = [
            "mean_latency_ms", "median_latency_ms", "p95_latency_ms",
            "p99_latency_ms", "min_latency_ms", "max_latency_ms", "std_latency_ms"
        ]

        for key in expected_keys:
            assert key in latency_stats
            assert isinstance(latency_stats[key], float)
            assert latency_stats[key] >= 0

    def test_evaluate_memory_usage(self, sample_embeddings):
        """Test memory usage evaluation."""
        db = FAISSVectorDB(dimension=512)
        embeddings = sample_embeddings[:5]
        metadata = [
            ProductMetadata(
                image_url=f"https://example.com/image{i}.jpg",
                product_url=f"https://example.com/product{i}",
                variants_products_link="Not available",
                shipping_weight="1 lb",
                product_dimensions="10x5x2",
                product_specification="Test spec",
                technical_details="Test details",
                is_amazon_seller="Y",
                combined_text=f"Product {i}"
            ) for i in range(5)
        ]

        db.add_vectors(embeddings, metadata)

        evaluator = VectorDBEvaluator(db)
        memory_stats = evaluator.evaluate_memory_usage()

        expected_keys = [
            "process_memory_mb", "total_vectors", "memory_per_vector_kb"
        ]

        for key in expected_keys:
            assert key in memory_stats
            assert isinstance(memory_stats[key], (int, float))

    def test_compare_index_types(self, sample_embeddings):
        """Test comparing different index configurations."""
        embeddings = sample_embeddings[:20]
        metadata = [
            ProductMetadata(
                image_url=f"https://example.com/image{i}.jpg",
                product_url=f"https://example.com/product{i}",
                variants_products_link="Not available",
                shipping_weight="1 lb",
                product_dimensions="10x5x2",
                product_specification="Test spec",
                technical_details="Test details",
                is_amazon_seller="Y",
                combined_text=f"Product {i}"
            ) for i in range(20)
        ]

        # Create dummy evaluator (we'll create new ones in the method)
        dummy_db = FAISSVectorDB(dimension=512)
        evaluator = VectorDBEvaluator(dummy_db)

        configs = [
            {"index_type": "flat", "metric": "l2"},
            {"index_type": "flat", "metric": "ip"}
        ]

        results = evaluator.compare_index_types(
            embeddings,
            metadata,
            embeddings[:3],  # query embeddings
            configs
        )

        assert len(results) == 2
        for config_name, result in results.items():
            if "error" not in result:
                assert "indexing_time_seconds" in result
                assert "latency_stats" in result
                assert "memory_stats" in result
                assert "index_stats" in result