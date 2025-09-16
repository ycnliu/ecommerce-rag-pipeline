"""
Prompt building utilities for the RAG pipeline.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from loguru import logger

from ..data.models import ProductMetadata


class PromptBuilder:
    """Builds prompts for the RAG system."""

    def __init__(self, max_chars: int = 6000):
        """
        Initialize prompt builder.

        Args:
            max_chars: Maximum characters for context
        """
        self.max_chars = max_chars
        self.few_shot_examples = self._get_few_shot_examples()

    def _get_few_shot_examples(self) -> str:
        """Get few-shot examples for the prompt."""
        return """
Q: What are the features of the Samsung Galaxy S21?
A: The Samsung Galaxy S21 comes with a 6.2-inch AMOLED display, a triple-camera setup, and a 4000mAh battery.

Q: Can you show me a picture of the Apple AirPods Pro?
A: Sure! The AirPods Pro feature active noise cancellation, a customizable fit, and water resistance. You can view it here: [Image URL].

Q: What gaming laptops do you recommend under $1000?
A: Based on the available products, I can recommend several gaming laptops under $1000 that offer good performance for their price range.
"""

    def build_context_from_metadata(
        self,
        metadata_list: List[ProductMetadata],
        max_items: Optional[int] = None
    ) -> str:
        """
        Build context string from product metadata.

        Args:
            metadata_list: List of product metadata
            max_items: Maximum number of items to include

        Returns:
            Formatted context string
        """
        if max_items:
            metadata_list = metadata_list[:max_items]

        prompt_parts = []
        char_count = 0

        for i, metadata in enumerate(metadata_list):
            section = f"{i+1}."

            # Add combined text (main product info)
            if metadata.combined_text:
                section += f"\n  - Product: {metadata.combined_text}"

            # Add additional details if available and not "Not available"
            details = [
                ("Shipping Weight", metadata.shipping_weight),
                ("Product Dimensions", metadata.product_dimensions),
                ("Product Specification", metadata.product_specification),
                ("Technical Details", metadata.technical_details),
                ("Product URL", str(metadata.product_url)),
                ("Image URL", metadata.image_url),
            ]

            for label, value in details:
                if value and str(value).strip() and str(value) != "Not available":
                    # Truncate very long values
                    display_value = str(value)
                    if len(display_value) > 200:
                        display_value = display_value[:200] + "..."
                    section += f"\n  - {label}: {display_value}"

            # Check if adding this section would exceed character limit
            if char_count + len(section) > self.max_chars:
                logger.info(f"Context truncated at item {i} due to character limit")
                break

            prompt_parts.append(section)
            char_count += len(section)

        return "\n\n".join(prompt_parts)

    def build_rag_prompt(
        self,
        query: str,
        retrieved_metadata: List[ProductMetadata],
        include_examples: bool = True,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Build complete RAG prompt with query and retrieved context.

        Args:
            query: User query
            retrieved_metadata: Retrieved product metadata
            include_examples: Whether to include few-shot examples
            custom_instructions: Custom system instructions

        Returns:
            Complete prompt string
        """
        # Build context from retrieved items
        context = self.build_context_from_metadata(retrieved_metadata)

        # Build instruction section
        instructions = custom_instructions or """You are a helpful AI assistant for an e-commerce website. Your job is to answer customer questions based on available product details.

Guidelines:
- Provide informative and accurate responses using the product information provided
- If multiple items are relevant, mention them and highlight their differences
- Include product URLs when recommending specific items
- If you don't have enough information to answer the question, say "I'm not sure based on the available product data"
- Be concise but comprehensive in your responses
- Focus on the most relevant products for the user's query"""

        # Combine all parts
        prompt_parts = []

        if include_examples:
            prompt_parts.append(self.few_shot_examples.strip())

        prompt_parts.extend([
            instructions,
            f"\nUser question:\n{query}",
            f"\nHere are some product descriptions that may be relevant:\n{context}",
            "\nProvide an informative and accurate response:"
        ])

        full_prompt = "\n\n".join(prompt_parts)

        # Ensure we don't exceed character limit
        if len(full_prompt) > self.max_chars:
            logger.warning(f"Prompt length {len(full_prompt)} exceeds limit {self.max_chars}")
            full_prompt = full_prompt[:self.max_chars].strip()

        return full_prompt

    def build_evaluation_prompt(
        self,
        query: str,
        retrieved_metadata: List[ProductMetadata],
        ground_truth: Optional[str] = None
    ) -> str:
        """
        Build prompt for evaluation purposes.

        Args:
            query: User query
            retrieved_metadata: Retrieved metadata
            ground_truth: Optional ground truth answer

        Returns:
            Evaluation prompt
        """
        context = self.build_context_from_metadata(retrieved_metadata)

        prompt = f"""Evaluation Task: Answer the following question based on the provided product information.

Question: {query}

Retrieved Products:
{context}

Answer:"""

        if ground_truth:
            prompt += f"\n\nGround Truth Reference: {ground_truth}"

        return prompt

    def build_comparison_prompt(
        self,
        query: str,
        product_groups: List[List[ProductMetadata]],
        group_labels: Optional[List[str]] = None
    ) -> str:
        """
        Build prompt for comparing different product groups.

        Args:
            query: User query
            product_groups: Groups of products to compare
            group_labels: Optional labels for each group

        Returns:
            Comparison prompt
        """
        prompt_parts = [
            "You are a product comparison expert. Compare the following product groups based on the user's query.",
            f"\nUser Query: {query}\n"
        ]

        for i, group in enumerate(product_groups):
            label = group_labels[i] if group_labels and i < len(group_labels) else f"Group {i+1}"
            context = self.build_context_from_metadata(group, max_items=3)
            prompt_parts.append(f"{label}:\n{context}")

        prompt_parts.append("\nProvide a detailed comparison highlighting the key differences and similarities:")

        return "\n\n".join(prompt_parts)

    def extract_key_features(self, metadata: ProductMetadata) -> Dict[str, str]:
        """
        Extract key features from product metadata.

        Args:
            metadata: Product metadata

        Returns:
            Dictionary of key features
        """
        features = {}

        # Parse combined text for key information
        combined_text = metadata.combined_text
        if "Price:" in combined_text:
            price_part = combined_text.split("Price:")[1].split("|")[0].strip()
            features["price"] = price_part

        if "Category:" in combined_text:
            category_part = combined_text.split("Category:")[1].split("|")[0].strip()
            features["category"] = category_part

        # Add other metadata
        if metadata.shipping_weight and metadata.shipping_weight != "Not available":
            features["shipping_weight"] = metadata.shipping_weight

        if metadata.product_dimensions and metadata.product_dimensions != "Not available":
            features["dimensions"] = metadata.product_dimensions

        return features