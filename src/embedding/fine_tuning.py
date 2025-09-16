"""
CLIP fine-tuning module for domain adaptation on e-commerce data.
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from transformers import get_linear_schedule_with_warmup
from loguru import logger
import wandb
from tqdm import tqdm

from ..utils.exceptions import ModelLoadError, EmbeddingError
from ..data.models import ProductMetadata


class EcommerceDataset(Dataset):
    """Dataset for e-commerce CLIP fine-tuning."""

    def __init__(
        self,
        products: List[ProductMetadata],
        processor: CLIPProcessor,
        max_text_length: int = 77,
        image_timeout: int = 10
    ):
        """
        Initialize e-commerce dataset.

        Args:
            products: List of product metadata
            processor: CLIP processor
            max_text_length: Maximum text sequence length
            image_timeout: Timeout for image downloads
        """
        self.products = products
        self.processor = processor
        self.max_text_length = max_text_length
        self.image_timeout = image_timeout

        # Filter products with valid images and text
        self.valid_products = self._filter_valid_products()
        logger.info(f"Dataset initialized with {len(self.valid_products)} valid products")

    def _filter_valid_products(self) -> List[ProductMetadata]:
        """Filter products with valid text and accessible images."""
        valid_products = []

        for product in self.products:
            if (
                product.combined_text and
                len(product.combined_text.strip()) > 10 and
                product.image_url and
                self._is_valid_image_url(product.image_url)
            ):
                valid_products.append(product)

        return valid_products

    def _is_valid_image_url(self, url: str) -> bool:
        """Check if image URL is valid and accessible."""
        try:
            # Take first URL if multiple are provided
            url = url.split('|')[0].strip()
            if not url or 'transparent-pixel' in url.lower():
                return False

            response = requests.head(url, timeout=5)
            return response.status_code == 200 and 'image' in response.headers.get('content-type', '')

        except Exception:
            return False

    def _load_image(self, url: str) -> Optional[Image.Image]:
        """Load image from URL with error handling."""
        try:
            # Take first URL if multiple are provided
            url = url.split('|')[0].strip()
            response = requests.get(url, timeout=self.image_timeout)
            response.raise_for_status()
            return Image.open(requests.get(url, stream=True).raw).convert("RGB")

        except Exception as e:
            logger.warning(f"Failed to load image from {url}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.valid_products)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        product = self.valid_products[idx]

        # Load image
        image = self._load_image(product.image_url)
        if image is None:
            # Return dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='white')

        # Process text and image
        inputs = self.processor(
            text=product.combined_text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'product_id': product.product_url
        }


class ContrastiveLoss(nn.Module):
    """Contrastive loss for CLIP training."""

    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.

        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            text_features: Text embeddings (batch_size, embed_dim)
            image_features: Image embeddings (batch_size, embed_dim)

        Returns:
            Contrastive loss
        """
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        logits = torch.matmul(text_features, image_features.T) / self.temperature

        # Labels are diagonal (each text matches its corresponding image)
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        # Compute cross-entropy loss in both directions
        loss_text_to_image = nn.CrossEntropyLoss()(logits, labels)
        loss_image_to_text = nn.CrossEntropyLoss()(logits.T, labels)

        return (loss_text_to_image + loss_image_to_text) / 2


class DomainAdaptationLoss(nn.Module):
    """Domain adaptation loss for e-commerce fine-tuning."""

    def __init__(self, alpha: float = 0.1):
        """
        Initialize domain adaptation loss.

        Args:
            alpha: Weight for domain adaptation component
        """
        super().__init__()
        self.alpha = alpha
        self.contrastive_loss = ContrastiveLoss()

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        category_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            text_features: Text embeddings
            image_features: Image embeddings
            category_labels: Optional category labels for additional supervision

        Returns:
            Dictionary with loss components
        """
        # Primary contrastive loss
        contrastive_loss = self.contrastive_loss(text_features, image_features)

        losses = {
            'contrastive_loss': contrastive_loss,
            'total_loss': contrastive_loss
        }

        # Add category-based clustering loss if labels provided
        if category_labels is not None:
            category_loss = self._compute_category_loss(text_features, category_labels)
            losses['category_loss'] = category_loss
            losses['total_loss'] = contrastive_loss + self.alpha * category_loss

        return losses

    def _compute_category_loss(
        self,
        features: torch.Tensor,
        category_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute category-based clustering loss."""
        # Simple implementation: encourage same-category items to be closer
        unique_categories = torch.unique(category_labels)
        category_loss = 0.0

        for category in unique_categories:
            mask = category_labels == category
            if mask.sum() > 1:
                category_features = features[mask]
                # Compute pairwise distances within category
                distances = torch.cdist(category_features, category_features)
                # Minimize mean distance within category
                category_loss += distances.mean()

        return category_loss / len(unique_categories)


class CLIPFineTuner:
    """CLIP fine-tuning manager for e-commerce domain adaptation."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        output_dir: str = "fine_tuned_models"
    ):
        """
        Initialize CLIP fine-tuner.

        Args:
            model_name: Pre-trained CLIP model name
            device: Device to use for training
            output_dir: Directory to save fine-tuned models
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model and processor
        self.model = None
        self.processor = None
        self.optimizer = None
        self.scheduler = None

        logger.info(f"Initialized CLIP fine-tuner for {model_name} on {self.device}")

    def load_model(self) -> None:
        """Load pre-trained CLIP model."""
        try:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

            # Enable gradient computation
            for param in self.model.parameters():
                param.requires_grad = True

            logger.info(f"Loaded CLIP model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise ModelLoadError(f"Failed to load CLIP model: {e}") from e

    def prepare_data(
        self,
        products: List[ProductMetadata],
        train_ratio: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders.

        Args:
            products: List of product metadata
            train_ratio: Ratio of data to use for training
            batch_size: Batch size for training
            num_workers: Number of data loader workers

        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Split data
        train_size = int(len(products) * train_ratio)
        train_products = products[:train_size]
        val_products = products[train_size:]

        # Create datasets
        train_dataset = EcommerceDataset(train_products, self.processor)
        val_dataset = EcommerceDataset(val_products, self.processor)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        logger.info(f"Prepared data: {len(train_dataset)} train, {len(val_dataset)} val samples")
        return train_loader, val_loader

    def setup_training(
        self,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        total_steps: Optional[int] = None
    ) -> None:
        """
        Setup optimizer and scheduler for training.

        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Setup scheduler if total_steps provided
        if total_steps:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

        logger.info(f"Setup training with lr={learning_rate}, weight_decay={weight_decay}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            loss_fn: Loss function
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            pixel_values = batch['pixel_values'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )

            # Compute loss
            loss_dict = loss_fn(outputs.text_embeds, outputs.image_embeds)
            loss = loss_dict['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        return {
            'train_loss': total_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def validate(
        self,
        val_loader: DataLoader,
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader
            loss_fn: Loss function

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )

                # Compute loss
                loss_dict = loss_fn(outputs.text_embeds, outputs.image_embeds)
                loss = loss_dict['total_loss']

                total_loss += loss.item()
                num_batches += 1

        return {'val_loss': total_loss / num_batches}

    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 5,
        save_best: bool = True,
        use_wandb: bool = False,
        project_name: str = "clip-ecommerce-finetuning"
    ) -> Dict[str, Any]:
        """
        Fine-tune CLIP model on e-commerce data.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            save_best: Whether to save the best model
            use_wandb: Whether to use Weights & Biases logging
            project_name: W&B project name

        Returns:
            Training history
        """
        if self.model is None or self.optimizer is None:
            raise ValueError("Model and optimizer not setup. Call load_model() and setup_training() first.")

        # Initialize wandb if requested
        if use_wandb:
            wandb.init(project=project_name, config={
                'model_name': self.model_name,
                'num_epochs': num_epochs,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })

        # Setup loss function
        loss_fn = DomainAdaptationLoss()

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader, loss_fn, epoch + 1)

            # Validate
            val_metrics = self.validate(val_loader, loss_fn)

            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['learning_rate'].append(train_metrics['learning_rate'])

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_metrics['train_loss']:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}"
            )

            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    **train_metrics,
                    **val_metrics
                })

            # Save best model
            if save_best and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_model(f"best_model_epoch_{epoch + 1}")
                logger.info(f"Saved best model with val_loss={best_val_loss:.4f}")

        # Save final model
        self.save_model("final_model")

        if use_wandb:
            wandb.finish()

        logger.info("Fine-tuning completed successfully")
        return history

    def save_model(self, model_name: str) -> str:
        """
        Save fine-tuned model.

        Args:
            model_name: Name for the saved model

        Returns:
            Path to saved model
        """
        save_path = self.output_dir / model_name
        save_path.mkdir(exist_ok=True)

        # Save model and processor
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

        # Save training config
        config = {
            'base_model': self.model_name,
            'fine_tuned_on': 'ecommerce_data',
            'device': self.device
        }

        with open(save_path / "fine_tuning_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {save_path}")
        return str(save_path)

    def load_fine_tuned_model(self, model_path: str) -> None:
        """
        Load a fine-tuned model.

        Args:
            model_path: Path to fine-tuned model
        """
        try:
            self.model = CLIPModel.from_pretrained(model_path).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_path)

            logger.info(f"Loaded fine-tuned model from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            raise ModelLoadError(f"Failed to load fine-tuned model: {e}") from e

    def evaluate_similarity(
        self,
        text_queries: List[str],
        image_urls: List[str],
        ground_truth_pairs: List[Tuple[int, int]]
    ) -> Dict[str, float]:
        """
        Evaluate text-image similarity performance.

        Args:
            text_queries: List of text queries
            image_urls: List of image URLs
            ground_truth_pairs: List of (text_idx, image_idx) ground truth pairs

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        self.model.eval()

        # Get embeddings
        text_embeddings = []
        image_embeddings = []

        with torch.no_grad():
            # Process texts
            for text in text_queries:
                inputs = self.processor(text=text, return_tensors="pt").to(self.device)
                text_emb = self.model.get_text_features(**inputs)
                text_embeddings.append(text_emb.cpu().numpy())

            # Process images
            for url in image_urls:
                try:
                    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    image_emb = self.model.get_image_features(**inputs)
                    image_embeddings.append(image_emb.cpu().numpy())
                except Exception:
                    # Use zero embedding for failed images
                    image_embeddings.append(np.zeros((1, self.model.config.projection_dim)))

        text_embeddings = np.vstack(text_embeddings)
        image_embeddings = np.vstack(image_embeddings)

        # Compute similarities
        similarities = np.dot(text_embeddings, image_embeddings.T)

        # Evaluate performance
        correct_top1 = 0
        correct_top5 = 0

        for text_idx, image_idx in ground_truth_pairs:
            text_similarities = similarities[text_idx]
            sorted_indices = np.argsort(text_similarities)[::-1]

            if sorted_indices[0] == image_idx:
                correct_top1 += 1
            if image_idx in sorted_indices[:5]:
                correct_top5 += 1

        return {
            'accuracy_top1': correct_top1 / len(ground_truth_pairs),
            'accuracy_top5': correct_top5 / len(ground_truth_pairs),
            'num_pairs': len(ground_truth_pairs)
        }