import torch
import torch.nn as nn


class CLAPProjection(nn.Module):
    """
    Project SELD class logits into LAION CLAP embedding space.

    Trained with MSE loss against CLAP text embeddings for each class label,
    enabling Audio-Language Cross-Attention to operate in a shared semantic space.
    """

    def __init__(self, num_classes: int = 13, clap_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_classes, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, clap_dim),
        )

    def forward(self, class_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            class_logits: (B, N, C) or (B, C) - raw class logits
        Returns:
            clap_embeds:  (B, N, clap_dim) or (B, clap_dim)
        """
        return self.proj(class_logits)
