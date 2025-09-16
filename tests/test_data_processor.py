"""
Tests for data processing functionality.
"""
import pytest
import pandas as pd
import tempfile
import os

from src.data.processor import DataProcessor, TextProcessor, GroundTruthGenerator


class TestDataProcessor:
    """Test cases for DataProcessor class."""

    def test_load_data(self, data_processor, temp_csv_file):
        """Test CSV data loading."""
        df = data_processor.load_data(temp_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Product Name" in df.columns
        assert "Category" in df.columns

    def test_clean_data(self, data_processor, temp_csv_file):
        """Test data cleaning functionality."""
        df = data_processor.load_data(temp_csv_file)
        df_cleaned = data_processor.clean_data(df)

        assert isinstance(df_cleaned, pd.DataFrame)
        assert len(df_cleaned) == 2

    def test_build_text_for_embedding(self, data_processor):
        """Test text combination for embeddings."""
        sample_row = pd.Series({
            'Product Name': 'Test Product',
            'Category': 'Electronics',
            'Selling Price': '$99.99',
            'Model Number': 'TP123',
            'About Product': 'Great product'
        })

        result = data_processor.build_text_for_embedding(sample_row)

        assert "Product Name: Test Product" in result
        assert "Category: Electronics" in result
        assert "Price: $99.99" in result
        assert "Model: TP123" in result
        assert "About: Great product" in result

    def test_build_text_for_embedding_excludes_not_available(self, data_processor):
        """Test that 'Not available' values are excluded from embedding text."""
        sample_row = pd.Series({
            'Product Name': 'Test Product',
            'Category': 'Not available',
            'Selling Price': '$99.99',
            'Model Number': 'Not available',
            'About Product': 'Great product'
        })

        result = data_processor.build_text_for_embedding(sample_row)

        assert "Product Name: Test Product" in result
        assert "Category: Not available" not in result
        assert "Price: $99.99" in result
        assert "Model: Not available" not in result
        assert "About: Great product" in result

    def test_create_metadata(self, data_processor, temp_csv_file):
        """Test metadata creation from DataFrame."""
        df = data_processor.load_data(temp_csv_file)
        df_cleaned = data_processor.clean_data(df)
        metadata_list = data_processor.create_metadata(df_cleaned)

        assert len(metadata_list) == 2
        assert all(hasattr(meta, 'combined_text') for meta in metadata_list)
        assert all(hasattr(meta, 'product_url') for meta in metadata_list)
        assert all(hasattr(meta, 'image_url') for meta in metadata_list)

    def test_process_full_pipeline(self, data_processor, temp_csv_file):
        """Test complete data processing pipeline."""
        df, metadata_list = data_processor.process_full_pipeline(temp_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert len(metadata_list) == 2
        assert 'combined_text' in df.columns


class TestTextProcessor:
    """Test cases for TextProcessor class."""

    def test_tokenize(self):
        """Test text tokenization."""
        text = "Hello World! This is a test-123."
        tokens = TextProcessor.tokenize(text)

        expected_tokens = {"hello", "world", "this", "is", "a", "test", "123"}
        assert tokens == expected_tokens

    def test_calculate_text_similarity(self):
        """Test text similarity calculation."""
        tokens1 = {"hello", "world", "test"}
        tokens2 = {"hello", "test", "example"}

        similarity = TextProcessor.calculate_text_similarity(tokens1, tokens2)
        assert similarity == 2  # "hello" and "test" overlap

    def test_rerank_by_text_similarity(self):
        """Test reranking by text similarity."""
        query = "wireless mouse gaming"
        items = [
            {"combined_text": "Gaming mouse wireless RGB"},
            {"combined_text": "Keyboard mechanical blue"},
            {"combined_text": "Mouse pad large gaming"},
            {"combined_text": "Wireless gaming headset"}
        ]

        reranked = TextProcessor.rerank_by_text_similarity(query, items)

        # First item should have highest similarity (wireless, mouse, gaming)
        assert "Gaming mouse wireless RGB" in reranked[0]["combined_text"]


class TestGroundTruthGenerator:
    """Test cases for GroundTruthGenerator class."""

    def test_initialization(self, temp_csv_file):
        """Test ground truth generator initialization."""
        processor = DataProcessor()
        df = processor.load_data(temp_csv_file)
        df_cleaned = processor.clean_data(df)

        generator = GroundTruthGenerator(df_cleaned)

        assert len(generator.preprocessed) == 2
        assert all(len(item) == 3 for item in generator.preprocessed)

    def test_ground_truth_fn(self, temp_csv_file):
        """Test ground truth generation."""
        processor = DataProcessor()
        df = processor.load_data(temp_csv_file)
        df_cleaned = processor.clean_data(df)

        # Create test data with similar products
        test_data = pd.DataFrame({
            'Product Name': [
                'Wireless Mouse Gaming',
                'Gaming Mouse Wireless',
                'Keyboard Mechanical',
                'Mouse Pad Gaming'
            ],
            'Category': [
                'Electronics',
                'Electronics',
                'Electronics',
                'Electronics'
            ]
        })

        generator = GroundTruthGenerator(test_data)
        matches = generator.ground_truth_fn(0)

        # Should find index 1 as a match (similar product name and same category)
        assert 1 in matches

    def test_ground_truth_fn_different_categories(self):
        """Test that ground truth doesn't match different categories."""
        test_data = pd.DataFrame({
            'Product Name': [
                'Wireless Mouse Gaming',
                'Gaming Mouse Wireless'
            ],
            'Category': [
                'Electronics',
                'Home & Garden'
            ]
        })

        generator = GroundTruthGenerator(test_data)
        matches = generator.ground_truth_fn(0)

        # Should not find matches due to different categories
        assert len(matches) == 0

    def test_ground_truth_fn_insufficient_overlap(self):
        """Test that ground truth requires sufficient token overlap."""
        test_data = pd.DataFrame({
            'Product Name': [
                'Wireless Gaming Mouse RGB',
                'Blue Mechanical Keyboard'
            ],
            'Category': [
                'Electronics',
                'Electronics'
            ]
        })

        generator = GroundTruthGenerator(test_data)
        matches = generator.ground_truth_fn(0)

        # Should not find matches due to insufficient token overlap
        assert len(matches) == 0