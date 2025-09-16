"""
Data processing module for e-commerce product data.
"""
import re
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from loguru import logger

from .models import ProductMetadata


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing."""

    def __init__(self):
        self.fill_values = {
            "Category": "Not available",
            "Selling Price": "Not available",
            "Model Number": "Not available",
            "About Product": "Not available",
            "Product Specification": "Not available",
            "Technical Details": "Not available",
            "Shipping Weight": "Not available",
            "Product Dimensions": "Not available",
            "Variants": "Not available"
        }

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and initially clean the dataset."""
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by removing empty columns and filling NaN values."""
        logger.info("Starting data cleaning process")

        # Remove columns that are entirely NaN
        initial_cols = len(df.columns)
        df = df.dropna(axis=1, how="all")
        logger.info(f"Removed {initial_cols - len(df.columns)} empty columns")

        # Remove specific unuseful columns
        columns_to_drop = ["Upc Ean Code"]
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df = df.drop(columns=existing_columns_to_drop)
            logger.info(f"Dropped columns: {existing_columns_to_drop}")

        # Fill NaN values
        df.fillna(value=self.fill_values, inplace=True)
        logger.info("Filled NaN values with default values")

        return df

    def build_text_for_embedding(self, row: pd.Series) -> str:
        """
        Build combined text for embedding, excluding 'Not available' values
        to improve semantic quality.
        """
        parts = [f"Product Name: {row['Product Name']}"]

        if row.get('Category', '') != "Not available":
            parts.append(f"Category: {row['Category']}")
        if row.get('Selling Price', '') != "Not available":
            parts.append(f"Price: {row['Selling Price']}")
        if row.get('Model Number', '') != "Not available":
            parts.append(f"Model: {row['Model Number']}")
        if row.get('About Product', '') != "Not available":
            parts.append(f"About: {row['About Product']}")

        return " | ".join(parts)

    def create_metadata(self, df: pd.DataFrame) -> List[ProductMetadata]:
        """Create metadata list from processed DataFrame."""
        logger.info("Creating metadata list")

        # Add combined text column
        df['combined_text'] = df.apply(self.build_text_for_embedding, axis=1)

        metadata_list = []
        for _, row in df.iterrows():
            try:
                metadata = ProductMetadata(
                    image_url=row["Image"],
                    product_url=row["Product Url"],
                    variants_products_link=row.get("Variants", "Not available"),
                    shipping_weight=row.get("Shipping Weight", "Not available"),
                    product_dimensions=row.get("Product Dimensions", "Not available"),
                    product_specification=row.get("Product Specification", "Not available"),
                    technical_details=row.get("Technical Details", "Not available"),
                    is_amazon_seller=row["Is Amazon Seller"],
                    combined_text=row["combined_text"]
                )
                metadata_list.append(metadata)
            except Exception as e:
                logger.error(f"Error creating metadata for row {row.name}: {e}")
                continue

        logger.info(f"Created {len(metadata_list)} metadata entries")
        return metadata_list

    def process_full_pipeline(self, csv_path: str) -> Tuple[pd.DataFrame, List[ProductMetadata]]:
        """Run the complete data processing pipeline."""
        df = self.load_data(csv_path)
        df_cleaned = self.clean_data(df)
        metadata_list = self.create_metadata(df_cleaned)

        logger.info("Data processing pipeline completed successfully")
        return df_cleaned, metadata_list


class TextProcessor:
    """Utilities for text processing and similarity calculations."""

    @staticmethod
    def tokenize(text: str) -> set:
        """Tokenize text into lowercase alphanumeric tokens."""
        return set(re.findall(r"\w+", str(text).lower()))

    @staticmethod
    def calculate_text_similarity(query_tokens: set, item_tokens: set) -> int:
        """Calculate text similarity score based on token overlap."""
        return len(query_tokens & item_tokens)

    @staticmethod
    def rerank_by_text_similarity(query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank items based on text similarity with query."""
        query_tokens = TextProcessor.tokenize(query)
        scored = []

        for item in items:
            item_tokens = TextProcessor.tokenize(item.get("combined_text", ""))
            score = TextProcessor.calculate_text_similarity(query_tokens, item_tokens)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]


class GroundTruthGenerator:
    """Generate ground truth for evaluation purposes."""

    def __init__(self, df: pd.DataFrame):
        self.preprocessed = self._preprocess_data(df)

    def _preprocess_data(self, df: pd.DataFrame) -> List[Tuple[int, set, str]]:
        """Preprocess data for ground truth generation."""
        preprocessed = []
        for i, row in df.iterrows():
            name_tokens = TextProcessor.tokenize(row["Product Name"])
            category = row.get("Category", "")
            preprocessed.append((i, name_tokens, category))
        return preprocessed

    def ground_truth_fn(self, idx: int) -> set:
        """
        Generate ground truth matches for a given index.
        Matches must be in same category and share at least 2 name tokens.
        """
        query_tokens = self.preprocessed[idx][1]
        query_category = self.preprocessed[idx][2]

        matches = set()
        for i, tokens, category in self.preprocessed:
            if i == idx:
                continue
            if category == query_category and len(query_tokens & tokens) >= 2:
                matches.add(i)

        return matches