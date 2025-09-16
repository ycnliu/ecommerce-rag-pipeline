"""
Embedding service using CLIP model for text and image embeddings.
"""
import io
from typing import Optional, Union, List
from PIL import Image
import torch
import numpy as np
import requests
from transformers import CLIPProcessor, CLIPModel
from loguru import logger

from ..utils.exceptions import EmbeddingError, ModelLoadError


class CLIPEmbeddingService:
    """Service for generating embeddings using CLIP model."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize CLIP embedding service.

        Args:
            model_name: HuggingFace model name
            device: Device to run model on (auto-detected if None)
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None

        logger.info(f"Initializing CLIP service with device: {self.device}")

    def load_model(self) -> None:
        """Load CLIP model and processor."""
        try:
            logger.info(f"Loading CLIP model: {self.model_name}")

            self.model = CLIPModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)

            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            logger.info("CLIP model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise ModelLoadError(f"Failed to load CLIP model: {e}") from e

    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded before use."""
        if self.model is None or self.processor is None:
            self.load_model()

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text input.

        Args:
            text: Input text

        Returns:
            Numpy array containing the embedding

        Raises:
            EmbeddingError: If embedding generation fails
        """
        self._ensure_model_loaded()

        try:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                embedding = self.model.get_text_features(**inputs).squeeze().cpu().numpy()

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            raise EmbeddingError(f"Failed to generate text embedding: {e}") from e

    def get_image_embedding(self, image_input: Union[str, Image.Image, bytes]) -> np.ndarray:
        """
        Generate embedding for image input.

        Args:
            image_input: Can be URL string, PIL Image, or bytes

        Returns:
            Numpy array containing the embedding

        Raises:
            EmbeddingError: If embedding generation fails
        """
        self._ensure_model_loaded()

        try:
            # Handle different input types
            if isinstance(image_input, str):
                # Assume it's a URL
                image = self._load_image_from_url(image_input)
            elif isinstance(image_input, bytes):
                # Bytes input
                image = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = image_input.convert("RGB")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs).squeeze().cpu().numpy()

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}")
            raise EmbeddingError(f"Failed to generate image embedding: {e}") from e

    def _load_image_from_url(self, url: str, timeout: int = 10) -> Image.Image:
        """
        Load image from URL.

        Args:
            url: Image URL
            timeout: Request timeout in seconds

        Returns:
            PIL Image object

        Raises:
            EmbeddingError: If image loading fails
        """
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
            return image

        except Exception as e:
            logger.error(f"Failed to load image from URL {url}: {e}")
            raise EmbeddingError(f"Failed to load image from URL: {e}") from e

    def get_multimodal_embedding(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Image.Image, bytes]] = None,
        fusion_method: str = "average"
    ) -> np.ndarray:
        """
        Generate combined embedding from text and/or image.

        Args:
            text: Optional text input
            image: Optional image input
            fusion_method: How to combine embeddings ("average", "concatenate")

        Returns:
            Combined embedding

        Raises:
            ValueError: If no inputs provided or invalid fusion method
            EmbeddingError: If embedding generation fails
        """
        if text is None and image is None:
            raise ValueError("At least one of text or image must be provided")

        embeddings = []

        if text is not None:
            text_emb = self.get_text_embedding(text)
            embeddings.append(text_emb)

        if image is not None:
            image_emb = self.get_image_embedding(image)
            embeddings.append(image_emb)

        if len(embeddings) == 1:
            return embeddings[0]

        # Combine embeddings
        if fusion_method == "average":
            return np.mean(embeddings, axis=0)
        elif fusion_method == "concatenate":
            return np.concatenate(embeddings)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def batch_text_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts
            batch_size: Processing batch size

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        self._ensure_model_loaded()

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            try:
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    batch_embeddings = self.model.get_text_features(**inputs).cpu().numpy()
                    embeddings.append(batch_embeddings)

            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                raise EmbeddingError(f"Batch processing failed: {e}") from e

        return np.vstack(embeddings)

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": self.model.config.projection_dim if self.model else None,
            "model_loaded": self.model is not None
        }