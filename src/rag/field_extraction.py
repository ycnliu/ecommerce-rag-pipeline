"""
Field extraction patterns for structured queries in e-commerce.
"""
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class FieldPattern:
    """Field extraction pattern definition."""
    keywords: Set[str]
    pattern: re.Pattern
    field_name: str
    description: str
    priority: int = 1


class FieldExtractor:
    """Extract specific fields from product text using pattern matching."""

    def __init__(self):
        """Initialize field extractor with predefined patterns."""
        self.patterns = self._build_patterns()
        logger.info(f"Initialized field extractor with {len(self.patterns)} patterns")

    def _build_patterns(self) -> List[FieldPattern]:
        """Build predefined field extraction patterns."""
        patterns = [
            # Price patterns
            FieldPattern(
                keywords={"price", "cost", "pricing", "amount", "dollar", "usd"},
                pattern=re.compile(
                    r"(?:Price|Cost|Selling\s+Price):\s*\$?([0-9]+(?:\.[0-9]+)?)",
                    re.IGNORECASE
                ),
                field_name="price",
                description="Product price in USD",
                priority=1
            ),

            # Dimensions patterns
            FieldPattern(
                keywords={"dimension", "dimensions", "size", "width", "height", "length"},
                pattern=re.compile(
                    r"(?:Product\s*Dimensions?|Size):\s*([0-9.]+\s*x\s*[0-9.]+(?:\s*x\s*[0-9.]+)?(?:\s*(?:inches?|in|cm|mm))?)",
                    re.IGNORECASE
                ),
                field_name="dimensions",
                description="Product dimensions",
                priority=1
            ),

            # Weight patterns
            FieldPattern(
                keywords={"weight", "heavy", "pounds", "lbs", "kg", "ounces", "oz"},
                pattern=re.compile(
                    r"(?:Item\s*Weight|Shipping\s*Weight):\s*([0-9.]+\s*(?:pounds?|lbs?|kg|ounces?|oz))",
                    re.IGNORECASE
                ),
                field_name="weight",
                description="Product weight",
                priority=1
            ),

            # ASIN patterns
            FieldPattern(
                keywords={"asin", "identifier", "id"},
                pattern=re.compile(
                    r"ASIN:\s*([A-Za-z0-9]+)",
                    re.IGNORECASE
                ),
                field_name="asin",
                description="Amazon ASIN identifier",
                priority=2
            ),

            # Model number patterns
            FieldPattern(
                keywords={"model", "model number", "part number", "sku"},
                pattern=re.compile(
                    r"(?:Item\s*Model\s*Number|Model):\s*([A-Za-z0-9\-_]+)",
                    re.IGNORECASE
                ),
                field_name="model",
                description="Product model number",
                priority=2
            ),

            # Age recommendation patterns
            FieldPattern(
                keywords={"age", "recommended age", "years", "children", "kids"},
                pattern=re.compile(
                    r"(?:Manufacturer\s*Recommended\s*Age|Ages?):\s*([0-9\-\+\s]*(?:years?|months?)[^|]*)",
                    re.IGNORECASE
                ),
                field_name="age",
                description="Recommended age range",
                priority=2
            ),

            # Shipping info patterns
            FieldPattern(
                keywords={"shipping", "delivery", "international", "domestic"},
                pattern=re.compile(
                    r"(?:Domestic\s*Shipping|International\s*Shipping):\s*([^|]+)",
                    re.IGNORECASE
                ),
                field_name="shipping",
                description="Shipping information",
                priority=3
            ),

            # Brand patterns
            FieldPattern(
                keywords={"brand", "manufacturer", "company", "made by"},
                pattern=re.compile(
                    r"(?:Brand|Manufacturer):\s*([^|]+)",
                    re.IGNORECASE
                ),
                field_name="brand",
                description="Product brand",
                priority=2
            ),

            # Color patterns
            FieldPattern(
                keywords={"color", "colour", "colors", "colours"},
                pattern=re.compile(
                    r"(?:Color|Colour):\s*([^|]+)",
                    re.IGNORECASE
                ),
                field_name="color",
                description="Product color",
                priority=3
            ),

            # Battery patterns
            FieldPattern(
                keywords={"battery", "batteries", "power", "aa", "aaa"},
                pattern=re.compile(
                    r"(?:Batteries?|Power):\s*([^|]+)",
                    re.IGNORECASE
                ),
                field_name="battery",
                description="Battery requirements",
                priority=3
            ),

            # Material patterns
            FieldPattern(
                keywords={"material", "made of", "fabric", "plastic", "wood", "metal"},
                pattern=re.compile(
                    r"(?:Material|Made\s+of|Fabric):\s*([^|]+)",
                    re.IGNORECASE
                ),
                field_name="material",
                description="Product material",
                priority=3
            ),

            # Rating patterns
            FieldPattern(
                keywords={"rating", "stars", "review", "score"},
                pattern=re.compile(
                    r"(?:Rating|Score):\s*([0-9.]+(?:\s*out\s*of\s*[0-9]+)?(?:\s*stars?)?)",
                    re.IGNORECASE
                ),
                field_name="rating",
                description="Product rating",
                priority=3
            )
        ]

        return patterns

    def extract_field(self, text: str, query: str) -> Optional[str]:
        """
        Extract specific field from text based on query intent.

        Args:
            text: Product text to search in
            query: User query indicating what field to extract

        Returns:
            Extracted field value or None
        """
        query_lower = query.lower()

        # Find matching patterns based on query keywords
        matching_patterns = []
        for pattern in self.patterns:
            if any(keyword in query_lower for keyword in pattern.keywords):
                matching_patterns.append(pattern)

        # Sort by priority (lower number = higher priority)
        matching_patterns.sort(key=lambda x: x.priority)

        # Try each matching pattern
        for pattern in matching_patterns:
            match = pattern.pattern.search(text)
            if match:
                extracted_value = match.group(1).strip()
                logger.debug(f"Extracted {pattern.field_name}: {extracted_value}")
                return extracted_value

        return None

    def extract_all_fields(self, text: str) -> Dict[str, str]:
        """
        Extract all available fields from text.

        Args:
            text: Product text to search in

        Returns:
            Dictionary of field_name -> extracted_value
        """
        extracted_fields = {}

        for pattern in self.patterns:
            match = pattern.pattern.search(text)
            if match:
                extracted_value = match.group(1).strip()
                extracted_fields[pattern.field_name] = extracted_value

        return extracted_fields

    def is_field_query(self, query: str) -> bool:
        """
        Check if query is asking for a specific field.

        Args:
            query: User query

        Returns:
            True if query appears to be asking for a specific field
        """
        query_lower = query.lower()

        # Check for direct field requests
        field_indicators = [
            "what is the", "what's the", "tell me the", "show me the",
            "how much", "what size", "what color", "what brand",
            "price", "cost", "dimensions", "weight", "model"
        ]

        return any(indicator in query_lower for indicator in field_indicators)

    def get_field_suggestions(self, text: str) -> List[Dict[str, str]]:
        """
        Get suggestions for fields that can be extracted from text.

        Args:
            text: Product text

        Returns:
            List of field suggestions with descriptions
        """
        suggestions = []
        extracted_fields = self.extract_all_fields(text)

        for pattern in self.patterns:
            if pattern.field_name in extracted_fields:
                suggestions.append({
                    "field": pattern.field_name,
                    "description": pattern.description,
                    "value": extracted_fields[pattern.field_name],
                    "keywords": list(pattern.keywords)[:3]  # First 3 keywords
                })

        return suggestions

    def add_custom_pattern(
        self,
        keywords: Set[str],
        pattern_str: str,
        field_name: str,
        description: str,
        priority: int = 2
    ) -> None:
        """
        Add a custom field extraction pattern.

        Args:
            keywords: Keywords that trigger this pattern
            pattern_str: Regular expression pattern
            field_name: Name of the field to extract
            description: Description of the field
            priority: Priority (1=highest, 3=lowest)
        """
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            field_pattern = FieldPattern(
                keywords=keywords,
                pattern=pattern,
                field_name=field_name,
                description=description,
                priority=priority
            )
            self.patterns.append(field_pattern)
            logger.info(f"Added custom pattern for field: {field_name}")

        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern_str}': {e}")
            raise ValueError(f"Invalid regex pattern: {e}")

    def validate_patterns(self) -> Dict[str, Any]:
        """
        Validate all patterns and return diagnostics.

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "total_patterns": len(self.patterns),
            "patterns_by_priority": {},
            "keyword_coverage": set(),
            "field_coverage": set(),
            "duplicate_fields": []
        }

        # Count patterns by priority
        for pattern in self.patterns:
            priority = pattern.priority
            if priority not in validation_results["patterns_by_priority"]:
                validation_results["patterns_by_priority"][priority] = 0
            validation_results["patterns_by_priority"][priority] += 1

            # Collect keywords and fields
            validation_results["keyword_coverage"].update(pattern.keywords)
            validation_results["field_coverage"].add(pattern.field_name)

        # Check for duplicate fields
        field_counts = {}
        for pattern in self.patterns:
            field_counts[pattern.field_name] = field_counts.get(pattern.field_name, 0) + 1

        validation_results["duplicate_fields"] = [
            field for field, count in field_counts.items() if count > 1
        ]

        # Convert sets to lists for JSON serialization
        validation_results["keyword_coverage"] = list(validation_results["keyword_coverage"])
        validation_results["field_coverage"] = list(validation_results["field_coverage"])

        return validation_results

    def test_extraction(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        Test field extraction on sample texts.

        Args:
            sample_texts: List of sample product texts

        Returns:
            Test results with extraction statistics
        """
        results = {
            "total_texts": len(sample_texts),
            "successful_extractions": 0,
            "field_extraction_counts": {},
            "sample_extractions": []
        }

        for i, text in enumerate(sample_texts):
            extracted_fields = self.extract_all_fields(text)

            if extracted_fields:
                results["successful_extractions"] += 1

            # Count field extractions
            for field in extracted_fields:
                if field not in results["field_extraction_counts"]:
                    results["field_extraction_counts"][field] = 0
                results["field_extraction_counts"][field] += 1

            # Store sample extractions
            if i < 5:  # First 5 samples
                results["sample_extractions"].append({
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "extracted_fields": extracted_fields
                })

        results["extraction_rate"] = results["successful_extractions"] / len(sample_texts)

        return results


class SmartFieldExtractor(FieldExtractor):
    """Enhanced field extractor with fuzzy matching and context awareness."""

    def __init__(self):
        """Initialize smart field extractor."""
        super().__init__()
        self.fuzzy_threshold = 0.8

    def fuzzy_extract_field(self, text: str, query: str, threshold: float = 0.8) -> Optional[Tuple[str, float]]:
        """
        Extract field using fuzzy matching for better recall.

        Args:
            text: Product text
            query: User query
            threshold: Minimum similarity threshold

        Returns:
            Tuple of (extracted_value, confidence) or None
        """
        from difflib import SequenceMatcher

        best_match = None
        best_confidence = 0.0

        query_words = set(query.lower().split())

        for pattern in self.patterns:
            # Calculate keyword similarity
            pattern_words = pattern.keywords
            similarity = len(query_words & pattern_words) / len(query_words | pattern_words)

            if similarity >= threshold:
                match = pattern.pattern.search(text)
                if match and similarity > best_confidence:
                    best_match = match.group(1).strip()
                    best_confidence = similarity

        return (best_match, best_confidence) if best_match else None

    def context_aware_extraction(self, text: str, query: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Extract field considering additional context.

        Args:
            text: Product text
            query: User query
            context: Additional context (category, brand, etc.)

        Returns:
            Extracted field value or None
        """
        # First try regular extraction
        result = self.extract_field(text, query)
        if result:
            return result

        # Try fuzzy extraction
        fuzzy_result = self.fuzzy_extract_field(text, query)
        if fuzzy_result and fuzzy_result[1] >= self.fuzzy_threshold:
            return fuzzy_result[0]

        # Context-based extraction (e.g., infer price range from category)
        if context.get("category") and "price" in query.lower():
            return self._infer_price_from_context(text, context)

        return None

    def _infer_price_from_context(self, text: str, context: Dict[str, Any]) -> Optional[str]:
        """Infer price information from context."""
        # This is a placeholder for more sophisticated context-based inference
        category = context.get("category", "").lower()

        # Look for any price-like patterns in text
        price_patterns = [
            r"\$([0-9]+(?:\.[0-9]+)?)",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:dollars?|usd)",
        ]

        for pattern_str in price_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            match = pattern.search(text)
            if match:
                return f"${match.group(1)}"

        return None