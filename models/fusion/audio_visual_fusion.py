import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioVisualFusion(nn.Module):
    """
    Apply audio attention map as a gated modulation on visual features.

    Design principle:
      - Regions with no audio signal → visual features unchanged
      - Regions with relevant audio  → visual features emphasised
    This preserves the VLA's original performance when audio is uninformative.
    """

    def __init__(self, D_v: int = 576, audio_context_dim: int = 256):
        super().__init__()
        self.D_v = D_v

        # Gate: (visual feature + attention scalar + audio context) → gate
        self.gate_net = nn.Sequential(
            nn.Linear(D_v + 1 + audio_context_dim, D_v),
            nn.GELU(),
            nn.Linear(D_v, D_v),
            nn.Sigmoid(),
        )

        # Project audio context to visual feature dimension
        self.audio_context_proj = nn.Linear(audio_context_dim, D_v)

        # Learnable residual scale (initialised to 0 for training stability)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_attn_map: torch.Tensor,
        audio_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (B, num_tokens, D_v)  - SmolVLA visual tokens (64 tokens)
            audio_attn_map:  (B, H, W)              - audio attention map
            audio_context:   (B, audio_context_dim) - weighted audio feature
        Returns:
            fused_features: (B, num_tokens, D_v)
        """
        B, num_tokens, D_v = visual_features.shape

        # Interpret tokens as a square spatial grid (8×8 for SmolVLA's 64 tokens)
        H_grid = W_grid = int(num_tokens ** 0.5)

        # Resize audio attention map to match the visual token grid
        attn_resized = F.interpolate(
            audio_attn_map.unsqueeze(1),           # (B, 1, H, W)
            size=(H_grid, W_grid),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, H_grid, W_grid)
        attn_flat = attn_resized.view(B, num_tokens, 1)  # (B, num_tokens, 1)

        # Broadcast audio context to all tokens
        ac = self.audio_context_proj(audio_context)          # (B, D_v)
        ac = ac.unsqueeze(1).expand(-1, num_tokens, -1)      # (B, num_tokens, D_v)

        # Compute gate
        gate_input = torch.cat([visual_features, attn_flat, ac], dim=-1)  # (B, num_tokens, D_v+1+ctx)
        gate = self.gate_net(gate_input)  # (B, num_tokens, D_v)

        # Gated modulation with learnable residual scale
        modulation = gate * attn_flat  # activated only where attention is high
        fused = visual_features + torch.sigmoid(self.alpha) * modulation * visual_features

        return fused
