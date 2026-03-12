import numpy as np
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Encode (azimuth, elevation) 2D coordinates using sinusoidal encoding.
    Similar to NeRF positional encoding.
    """

    def __init__(self, d_model: int = 256, num_frequencies: int = 64):
        super().__init__()
        self.num_frequencies = num_frequencies
        # 2 (az, el) × num_frequencies × 2 (sin, cos) → projected to d_model
        self.proj = nn.Linear(2 * num_frequencies * 2, d_model)

        # Log-spaced frequency bands
        freqs = torch.logspace(0, np.log10(100), num_frequencies)
        self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 2) - azimuth, elevation in radians
        Returns:
            encoding: (B, N, d_model)
        """
        B, N, _ = coords.shape
        # (B, N, 2) → (B, N, 2, 1) * (num_freq,) → (B, N, 2, num_freq)
        scaled = coords.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # sin & cos → (B, N, 2, num_freq, 2) → flatten → (B, N, 2*num_freq*2)
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        encoded = encoded.reshape(B, N, -1)
        return self.proj(encoded)


class SoundTokenEncoder(nn.Module):
    """
    Convert SELD output into sound_tokens of shape (B, N, D_s).
    Each token = spatial_embed + class_embed + energy_embed.
    """

    def __init__(self, D_s: int = 256, num_classes: int = 13, num_frequencies: int = 64):
        super().__init__()
        self.D_s = D_s

        # Spatial encoding: (az, el) → D_s via sinusoidal positional encoding
        self.spatial_enc = SinusoidalPositionalEncoding(d_model=D_s, num_frequencies=num_frequencies)

        # Class embedding: soft logits → D_s (robust to uncertain predictions)
        self.class_proj = nn.Sequential(
            nn.Linear(num_classes, D_s),
            nn.GELU(),
            nn.Linear(D_s, D_s),
        )

        # Energy embedding: scalar → D_s
        self.energy_proj = nn.Linear(1, D_s)

        # Fusion: concatenate 3 embeddings → D_s
        self.fuse = nn.Sequential(
            nn.Linear(D_s * 3, D_s),
            nn.LayerNorm(D_s),
            nn.GELU(),
            nn.Linear(D_s, D_s),
        )

    def forward(
        self,
        peak_coords: torch.Tensor,
        class_logits: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            peak_coords:  (B, N, 2) - azimuth, elevation in radians
            class_logits: (B, N, C) - raw logits (before softmax)
            energy:       (B, N, 1) - energy/confidence
        Returns:
            sound_tokens: (B, N, D_s)
        """
        s = self.spatial_enc(peak_coords)   # (B, N, D_s)
        c = self.class_proj(class_logits)   # (B, N, D_s)
        e = self.energy_proj(energy)        # (B, N, D_s)
        return self.fuse(torch.cat([s, c, e], dim=-1))  # (B, N, D_s)
