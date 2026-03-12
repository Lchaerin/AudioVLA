from dataclasses import dataclass
import torch


@dataclass
class SELDOutput:
    """Output of the SELD model."""
    peak_coords: torch.Tensor    # (B, N_max, 2) - azimuth, elevation in radians
    class_logits: torch.Tensor   # (B, N_max, C) - C = number of sound classes
    energy: torch.Tensor         # (B, N_max, 1) - energy/confidence per peak
    valid_mask: torch.Tensor     # (B, N_max) - bool, padding mask
    heatmap: torch.Tensor = None # (B, T_frames, H_az, W_el) - optional, for visualization
