import torch
import torch.nn as nn
import numpy as np


class JAQLoss(nn.Module):
    """Joint Alignment & Quality (JAQ) Loss.
    
    A reference-free objective combining text-image compatibility and aesthetic score
    evaluated on calibration prompts without requiring ground-truth reference images.
    """

    def __init__(self, lambda_align: float = 0.5, lambda_quality: float = 0.5):
        super().__init__()
        self.lambda_align = lambda_align
        self.lambda_quality = lambda_quality

    def compute_text_compatibility(self, text_embeddings: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """Reference-free cosine alignment score between text prompts and generated feature maps."""
        text_norm = text_embeddings / (text_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        img_norm = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = torch.sum(text_norm * img_norm, dim=-1)
        return similarity.mean()

    def compute_image_quality(self, image_features: torch.Tensor) -> torch.Tensor:
        """Reference-free quality metric based on feature variance and high-frequency energy ratio."""
        var = torch.var(image_features, dim=-1).mean()
        norm_ratio = torch.sigmoid(var)
        return norm_ratio

    def forward(self, text_embeddings: torch.Tensor, image_features: torch.Tensor) -> dict:
        align_score = self.compute_text_compatibility(text_embeddings, image_features)
        quality_score = self.compute_image_quality(image_features)
        
        total_loss = - (self.lambda_align * align_score + self.lambda_quality * quality_score)
        
        return {
            "total_loss": float(total_loss.item()),
            "alignment_score": float(align_score.item()),
            "quality_score": float(quality_score.item()),
        }
