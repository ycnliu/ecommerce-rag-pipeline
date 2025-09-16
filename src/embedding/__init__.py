"""
Embedding services for the e-commerce RAG pipeline.
"""
from .service import CLIPEmbeddingService
from .fusion import AdvancedEmbeddingFusion
from .fine_tuning import CLIPFineTuner, EcommerceDataset, ContrastiveLoss, DomainAdaptationLoss

__all__ = [
    "CLIPEmbeddingService",
    "AdvancedEmbeddingFusion",
    "CLIPFineTuner",
    "EcommerceDataset",
    "ContrastiveLoss",
    "DomainAdaptationLoss"
]