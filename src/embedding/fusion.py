"""
Advanced embedding fusion strategies for multimodal search.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from loguru import logger

from ..utils.exceptions import EmbeddingError
from .service import CLIPEmbeddingService


class CrossModalAttention(nn.Module):
    """Neural attention module for cross-modal fusion."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Apply cross-modal attention."""
        batch_size = query.size(0)

        # Project to Q, K, V
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention weights
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )

        # Output projection with residual connection
        output = self.out_proj(attn_output)
        return self.layer_norm(output + query)


class DynamicWeightingModule(nn.Module):
    """Neural module for dynamic fusion weight prediction."""

    def __init__(self, embed_dim: int, num_modalities: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        self.fusion_net = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_modalities),
            nn.Softmax(dim=-1)
        )

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Predict dynamic weights for fusion."""
        # Concatenate all embeddings
        concat_emb = torch.cat(embeddings, dim=-1)
        weights = self.fusion_net(concat_emb)
        return weights


class AdvancedEmbeddingFusion:
    """Advanced embedding fusion combining CLIP and SentenceTransformer embeddings."""

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        sentence_model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        target_dim: int = 512,
        use_neural_fusion: bool = True
    ):
        """
        Initialize advanced embedding fusion.

        Args:
            clip_model_name: CLIP model name
            sentence_model_name: SentenceTransformer model name
            device: Device to run models on
            target_dim: Target embedding dimension
            use_neural_fusion: Whether to use neural fusion modules
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_dim = target_dim
        self.use_neural_fusion = use_neural_fusion

        # Initialize CLIP service
        self.clip_service = CLIPEmbeddingService(
            model_name=clip_model_name,
            device=self.device
        )

        # Initialize SentenceTransformer
        self.sentence_model = SentenceTransformer(sentence_model_name, device=self.device)

        # Initialize neural fusion modules if enabled
        if self.use_neural_fusion:
            self.cross_attention = CrossModalAttention(target_dim).to(self.device)
            self.dynamic_weighting = DynamicWeightingModule(target_dim).to(self.device)
            self.projection_layers = nn.ModuleDict({
                'clip_proj': nn.Linear(512, target_dim),  # CLIP output is typically 512
                'sentence_proj': nn.Linear(384, target_dim)  # MiniLM output is 384
            }).to(self.device)

        logger.info(f"Initialized advanced fusion with CLIP: {clip_model_name}, Sentence: {sentence_model_name}")
        if self.use_neural_fusion:
            logger.info("Neural fusion modules enabled")

    def load_models(self) -> None:
        """Load all models."""
        self.clip_service.load_model()
        logger.info("All embedding models loaded successfully")

    def normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit vector."""
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def pad_to_target_dim(self, embedding: np.ndarray) -> np.ndarray:
        """Pad or truncate embedding to target dimension."""
        if embedding.shape[0] >= self.target_dim:
            return embedding[:self.target_dim]
        return np.pad(embedding, (0, self.target_dim - embedding.shape[0]), mode="constant")

    def fuse_embeddings(
        self,
        embeddings: List[np.ndarray],
        weights: Optional[List[float]] = None,
        fusion_method: str = "weighted_average"
    ) -> np.ndarray:
        """
        Fuse multiple embeddings using specified method.

        Args:
            embeddings: List of embeddings to fuse
            weights: Optional weights for each embedding
            fusion_method: Fusion strategy

        Returns:
            Fused embedding
        """
        if not embeddings:
            raise ValueError("No embeddings provided for fusion")

        if weights is None:
            weights = [1.0 / len(embeddings)] * len(embeddings)

        if len(weights) != len(embeddings):
            raise ValueError("Number of weights must match number of embeddings")

        # Normalize all embeddings
        norm_embeddings = [self.normalize(emb) for emb in embeddings]

        if fusion_method == "weighted_average":
            fused = np.zeros_like(norm_embeddings[0])
            for emb, weight in zip(norm_embeddings, weights):
                fused += weight * emb
            return self.normalize(fused)

        elif fusion_method == "concatenate":
            return np.concatenate(norm_embeddings)

        elif fusion_method == "max_pooling":
            stacked = np.stack(norm_embeddings, axis=0)
            return np.max(stacked, axis=0)

        elif fusion_method == "attention_weighted":
            return self._attention_fusion(norm_embeddings, weights)

        elif fusion_method == "neural_attention" and self.use_neural_fusion:
            return self._neural_attention_fusion(norm_embeddings)

        elif fusion_method == "dynamic_weighted" and self.use_neural_fusion:
            return self._dynamic_weighted_fusion(norm_embeddings)

        elif fusion_method == "cross_modal_attention" and self.use_neural_fusion:
            return self._cross_modal_attention_fusion(norm_embeddings)

        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def _attention_fusion(self, embeddings: List[np.ndarray], base_weights: List[float]) -> np.ndarray:
        """Apply attention-based fusion."""
        # Simple attention mechanism
        scores = []
        for i, emb1 in enumerate(embeddings):
            score = 0
            for j, emb2 in enumerate(embeddings):
                if i != j:
                    score += np.dot(emb1, emb2) * base_weights[j]
            scores.append(score)

        # Softmax normalization
        exp_scores = np.exp(np.array(scores) - np.max(scores))
        attention_weights = exp_scores / np.sum(exp_scores)

        # Apply attention weights
        fused = np.zeros_like(embeddings[0])
        for emb, weight in zip(embeddings, attention_weights):
            fused += weight * emb

        return self.normalize(fused)

    def _neural_attention_fusion(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Apply neural attention-based fusion."""
        if not self.use_neural_fusion:
            raise ValueError("Neural fusion not enabled")

        # Convert to tensors and project to target dimension
        tensors = []
        for i, emb in enumerate(embeddings):
            tensor = torch.from_numpy(emb).float().unsqueeze(0).to(self.device)

            if i == 0:  # CLIP embedding
                projected = self.projection_layers['clip_proj'](tensor)
            else:  # Sentence embedding
                projected = self.projection_layers['sentence_proj'](tensor)

            tensors.append(projected)

        # Apply cross-modal attention
        if len(tensors) == 2:
            attended = self.cross_attention(tensors[0], tensors[1], tensors[1])
            result = (attended + tensors[0]) / 2  # Residual connection
        else:
            result = tensors[0]

        return self.normalize(result.squeeze(0).cpu().numpy())

    def _dynamic_weighted_fusion(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Apply dynamic weighting fusion."""
        if not self.use_neural_fusion:
            raise ValueError("Neural fusion not enabled")

        # Convert to tensors and project
        tensors = []
        for i, emb in enumerate(embeddings):
            tensor = torch.from_numpy(emb).float().to(self.device)

            if i == 0:  # CLIP embedding
                projected = self.projection_layers['clip_proj'](tensor)
            else:  # Sentence embedding
                projected = self.projection_layers['sentence_proj'](tensor)

            tensors.append(projected)

        # Predict dynamic weights
        weights = self.dynamic_weighting(tensors)

        # Apply weighted fusion
        fused = torch.zeros_like(tensors[0])
        for tensor, weight in zip(tensors, weights[0]):
            fused += weight * tensor

        return self.normalize(fused.cpu().numpy())

    def _cross_modal_attention_fusion(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Apply bidirectional cross-modal attention fusion."""
        if not self.use_neural_fusion or len(embeddings) < 2:
            return embeddings[0]

        # Convert to tensors and project
        clip_tensor = torch.from_numpy(embeddings[0]).float().unsqueeze(0).to(self.device)
        sentence_tensor = torch.from_numpy(embeddings[1]).float().unsqueeze(0).to(self.device)

        clip_proj = self.projection_layers['clip_proj'](clip_tensor)
        sentence_proj = self.projection_layers['sentence_proj'](sentence_tensor)

        # Bidirectional attention
        clip_attended = self.cross_attention(clip_proj, sentence_proj, sentence_proj)
        sentence_attended = self.cross_attention(sentence_proj, clip_proj, clip_proj)

        # Combine with residual connections
        fused = (clip_attended + sentence_attended + clip_proj + sentence_proj) / 4

        return self.normalize(fused.squeeze(0).cpu().numpy())

    def get_hybrid_text_embedding(self, text: str, fusion_weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate hybrid text embedding using both CLIP and SentenceTransformer.

        Args:
            text: Input text
            fusion_weights: Weights for [clip, sentence_transformer]

        Returns:
            Fused text embedding
        """
        if fusion_weights is None:
            fusion_weights = [0.6, 0.4]  # Favor CLIP for multimodal compatibility

        try:
            # Get CLIP text embedding
            clip_emb = self.clip_service.get_text_embedding(text)
            clip_emb = self.normalize(clip_emb)

            # Get SentenceTransformer embedding
            st_emb = self.sentence_model.encode([text], normalize_embeddings=True)[0]
            st_emb = self.pad_to_target_dim(st_emb)
            st_emb = self.normalize(st_emb)

            # Fuse embeddings
            fused_emb = self.fuse_embeddings(
                [clip_emb, st_emb],
                weights=fusion_weights,
                fusion_method="weighted_average"
            )

            return fused_emb

        except Exception as e:
            logger.error(f"Failed to generate hybrid text embedding: {e}")
            raise EmbeddingError(f"Hybrid text embedding failed: {e}") from e

    def get_hybrid_image_embedding(self, image_input) -> np.ndarray:
        """
        Generate image embedding using CLIP.

        Args:
            image_input: Image input (URL, PIL Image, or bytes)

        Returns:
            Image embedding padded to target dimension
        """
        try:
            clip_emb = self.clip_service.get_image_embedding(image_input)
            return self.normalize(clip_emb)

        except Exception as e:
            logger.error(f"Failed to generate hybrid image embedding: {e}")
            raise EmbeddingError(f"Hybrid image embedding failed: {e}") from e

    def get_multimodal_embedding(
        self,
        text: Optional[str] = None,
        image_input=None,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
        fusion_method: str = "weighted_average"
    ) -> np.ndarray:
        """
        Generate multimodal embedding from text and/or image.

        Args:
            text: Optional text input
            image_input: Optional image input
            text_weight: Weight for text embedding
            image_weight: Weight for image embedding
            fusion_method: Fusion strategy

        Returns:
            Multimodal embedding
        """
        if text is None and image_input is None:
            raise ValueError("At least one of text or image_input must be provided")

        embeddings = []
        weights = []

        if text is not None:
            text_emb = self.get_hybrid_text_embedding(text)
            embeddings.append(text_emb)
            weights.append(text_weight)

        if image_input is not None:
            image_emb = self.get_hybrid_image_embedding(image_input)
            embeddings.append(image_emb)
            weights.append(image_weight)

        if len(embeddings) == 1:
            return embeddings[0]

        return self.fuse_embeddings(embeddings, weights, fusion_method)

    def batch_hybrid_text_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        fusion_weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Generate hybrid text embeddings for multiple texts in batches.

        Args:
            texts: List of texts
            batch_size: Processing batch size
            fusion_weights: Weights for fusion

        Returns:
            Array of hybrid embeddings
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                emb = self.get_hybrid_text_embedding(text, fusion_weights)
                batch_embeddings.append(emb)

            embeddings.extend(batch_embeddings)

        return np.vstack(embeddings)

    def create_product_embeddings(
        self,
        product_texts: List[str],
        product_images: Optional[List[str]] = None,
        text_weight: float = 0.7,
        image_weight: float = 0.3
    ) -> np.ndarray:
        """
        Create embeddings for products using both text and image data.

        Args:
            product_texts: List of product text descriptions
            product_images: Optional list of product image URLs
            text_weight: Weight for text embeddings
            image_weight: Weight for image embeddings

        Returns:
            Array of product embeddings
        """
        embeddings = []

        for i, text in enumerate(product_texts):
            image_url = product_images[i] if product_images and i < len(product_images) else None

            try:
                emb = self.get_multimodal_embedding(
                    text=text,
                    image_input=image_url,
                    text_weight=text_weight,
                    image_weight=image_weight
                )
                embeddings.append(emb)

            except Exception as e:
                logger.warning(f"Failed to create embedding for product {i}: {e}")
                # Fallback to text-only embedding
                text_emb = self.get_hybrid_text_embedding(text)
                embeddings.append(text_emb)

        return np.vstack(embeddings)

    def get_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray, metric: str = "cosine") -> float:
        """
        Calculate similarity between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding
            metric: Similarity metric ("cosine", "euclidean", "dot")

        Returns:
            Similarity score
        """
        if metric == "cosine":
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        elif metric == "euclidean":
            return 1.0 / (1.0 + np.linalg.norm(emb1 - emb2))
        elif metric == "dot":
            return np.dot(emb1, emb2)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")

    def analyze_embedding_quality(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Analyze quality metrics of embeddings.

        Args:
            embeddings: Array of embeddings to analyze

        Returns:
            Dictionary with quality metrics
        """
        # Calculate statistics
        norms = np.linalg.norm(embeddings, axis=1)
        pairwise_sims = np.dot(embeddings, embeddings.T)

        # Remove diagonal (self-similarity)
        mask = ~np.eye(len(embeddings), dtype=bool)
        off_diagonal_sims = pairwise_sims[mask]

        return {
            "num_embeddings": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "norm_stats": {
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms)),
                "min": float(np.min(norms)),
                "max": float(np.max(norms))
            },
            "similarity_stats": {
                "mean": float(np.mean(off_diagonal_sims)),
                "std": float(np.std(off_diagonal_sims)),
                "min": float(np.min(off_diagonal_sims)),
                "max": float(np.max(off_diagonal_sims))
            },
            "diversity_score": 1.0 - float(np.mean(off_diagonal_sims))  # Higher is more diverse
        }

    def save_fusion_config(self, config_path: str) -> None:
        """Save fusion configuration to file."""
        config = {
            "clip_model": self.clip_service.model_name,
            "text_model": self.text_model._modules["0"].auto_model.name_or_path,
            "target_dim": self.target_dim,
            "device": self.device
        }

        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Fusion configuration saved to {config_path}")

    def train_neural_fusion(
        self,
        training_data: List[Tuple[str, str, float]],
        epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 32
    ) -> None:
        """
        Train neural fusion modules on similarity data.

        Args:
            training_data: List of (text1, text2, similarity_score) tuples
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
        """
        if not self.use_neural_fusion:
            raise ValueError("Neural fusion not enabled")

        # Set modules to training mode
        self.cross_attention.train()
        self.dynamic_weighting.train()
        self.projection_layers.train()

        # Initialize optimizer
        params = list(self.cross_attention.parameters()) + \
                list(self.dynamic_weighting.parameters()) + \
                list(self.projection_layers.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        logger.info(f"Training neural fusion on {len(training_data)} examples for {epochs} epochs")

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                batch_loss = 0.0

                for text1, text2, target_sim in batch:
                    # Get embeddings
                    emb1 = self.get_hybrid_text_embedding(text1)
                    emb2 = self.get_hybrid_text_embedding(text2)

                    # Convert to tensors
                    emb1_tensor = torch.from_numpy(emb1).float().to(self.device)
                    emb2_tensor = torch.from_numpy(emb2).float().to(self.device)

                    # Compute predicted similarity
                    pred_sim = torch.cosine_similarity(emb1_tensor, emb2_tensor, dim=0)

                    # Compute loss
                    target_tensor = torch.tensor(target_sim, device=self.device)
                    loss = F.mse_loss(pred_sim, target_tensor)
                    batch_loss += loss

                # Backward pass
                batch_loss /= len(batch)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Set back to eval mode
        self.cross_attention.eval()
        self.dynamic_weighting.eval()
        self.projection_layers.eval()

        logger.info("Neural fusion training completed")

    def save_model(self, save_path: str) -> None:
        """Save the complete fusion model."""
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save configuration
        config = {
            "clip_model": self.clip_service.model_name,
            "sentence_model": getattr(self.sentence_model, '_model_name', 'all-MiniLM-L6-v2'),
            "target_dim": self.target_dim,
            "device": self.device,
            "use_neural_fusion": self.use_neural_fusion
        }

        import json
        with open(os.path.join(save_path, "fusion_config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        # Save neural modules if enabled
        if self.use_neural_fusion:
            torch.save({
                'cross_attention': self.cross_attention.state_dict(),
                'dynamic_weighting': self.dynamic_weighting.state_dict(),
                'projection_layers': self.projection_layers.state_dict()
            }, os.path.join(save_path, "neural_modules.pt"))

        logger.info(f"Fusion model saved to {save_path}")

    def load_model(self, load_path: str) -> None:
        """Load a saved fusion model."""
        import os
        import json

        # Load configuration
        config_path = os.path.join(load_path, "fusion_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load neural modules if they exist
        neural_path = os.path.join(load_path, "neural_modules.pt")
        if os.path.exists(neural_path) and self.use_neural_fusion:
            state_dict = torch.load(neural_path, map_location=self.device)
            self.cross_attention.load_state_dict(state_dict['cross_attention'])
            self.dynamic_weighting.load_state_dict(state_dict['dynamic_weighting'])
            self.projection_layers.load_state_dict(state_dict['projection_layers'])

            # Set to eval mode
            self.cross_attention.eval()
            self.dynamic_weighting.eval()
            self.projection_layers.eval()

        logger.info(f"Fusion model loaded from {load_path}")

    def get_available_fusion_methods(self) -> List[str]:
        """Get list of available fusion methods."""
        methods = [
            "weighted_average",
            "concatenate",
            "max_pooling",
            "attention_weighted"
        ]

        if self.use_neural_fusion:
            methods.extend([
                "neural_attention",
                "dynamic_weighted",
                "cross_modal_attention"
            ])

        return methods

    def evaluate_fusion_quality(
        self,
        test_queries: List[str],
        test_documents: List[str],
        relevance_scores: List[float],
        fusion_method: str = "cross_modal_attention"
    ) -> Dict[str, float]:
        """
        Evaluate fusion quality on test data.

        Args:
            test_queries: List of test queries
            test_documents: List of test documents
            relevance_scores: Ground truth relevance scores
            fusion_method: Fusion method to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        if len(test_queries) != len(test_documents) or len(test_queries) != len(relevance_scores):
            raise ValueError("All input lists must have the same length")

        predicted_scores = []

        for query, doc in zip(test_queries, test_documents):
            # Get embeddings using specified fusion method
            query_emb = self.get_hybrid_text_embedding(query)
            doc_emb = self.get_hybrid_text_embedding(doc)

            # Compute similarity
            similarity = self.get_embedding_similarity(query_emb, doc_emb, "cosine")
            predicted_scores.append(similarity)

        # Calculate metrics
        predicted_scores = np.array(predicted_scores)
        relevance_scores = np.array(relevance_scores)

        # Pearson correlation
        correlation = np.corrcoef(predicted_scores, relevance_scores)[0, 1]

        # Mean squared error
        mse = np.mean((predicted_scores - relevance_scores) ** 2)

        # Mean absolute error
        mae = np.mean(np.abs(predicted_scores - relevance_scores))

        return {
            "correlation": float(correlation),
            "mse": float(mse),
            "mae": float(mae),
            "fusion_method": fusion_method
        }

    @classmethod
    def load_fusion_config(cls, config_path: str) -> 'AdvancedEmbeddingFusion':
        """Load fusion configuration from file."""
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)

        return cls(
            clip_model_name=config["clip_model"],
            sentence_model_name=config.get("sentence_model", config.get("text_model", "all-MiniLM-L6-v2")),
            device=config["device"],
            target_dim=config["target_dim"],
            use_neural_fusion=config.get("use_neural_fusion", True)
        )