import numpy as np
import torch
import torch.nn as nn


class AudioAttentionMapGenerator(nn.Module):
    """
    Splat audio-language attention weights onto image space using Gaussian blobs.

    Output: (B, H, W) attention map where high values indicate regions
    whose sound source is most relevant to the language command.
    """

    def __init__(self, sigma: float = 20.0, learnable_sigma: bool = True):
        super().__init__()
        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma)))
        else:
            self.register_buffer("log_sigma", torch.tensor(np.log(sigma)))

    def forward(
        self,
        pixel_coords: torch.Tensor,
        attn_weights: torch.Tensor,
        in_frame_mask: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Args:
            pixel_coords:  (B, N, 2) - [u, v] pixel coordinates
            attn_weights:  (B, N)    - per-source attention weight
            in_frame_mask: (B, N)    - bool, sources inside the image frame
            H, W:          output map resolution (matches visual feature grid)
        Returns:
            attn_map: (B, H, W) - normalised attention map in [0, 1]
        """
        B, N = attn_weights.shape
        sigma = torch.exp(self.log_sigma)

        # Zero out contributions from sources outside the frame
        weights = attn_weights * in_frame_mask.float()  # (B, N)

        # Build pixel grid: (H, W, 2) with [u, v] = [x, y]
        device = pixel_coords.device
        grid_y = torch.arange(H, device=device, dtype=torch.float32)
        grid_x = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

        # Gaussian splatting
        # pixel_coords: (B, N, 2) → (B, N, 1, 1, 2)
        centers = pixel_coords.unsqueeze(2).unsqueeze(3)
        # grid: (H, W, 2) → (1, 1, H, W, 2)
        grid = grid.unsqueeze(0).unsqueeze(0)

        diff = grid - centers               # (B, N, H, W, 2)
        sq_dist = (diff ** 2).sum(dim=-1)   # (B, N, H, W)
        gaussians = torch.exp(-0.5 * sq_dist / (sigma ** 2 + 1e-6))  # (B, N, H, W)

        # Weighted sum over sources
        w = weights.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        attn_map = (w * gaussians).sum(dim=1)    # (B, H, W)

        # Normalise to [0, 1]
        attn_map = attn_map / (attn_map.amax(dim=(-1, -2), keepdim=True) + 1e-6)

        return attn_map
