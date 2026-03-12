import torch
import torch.nn as nn


class AudioLanguageCrossAttention(nn.Module):
    """
    Cross-attention between sound tokens (from SELD) and language tokens
    (from SmolVLA language encoder). Identifies which sound token is most
    relevant to the language command.

    Inputs:
        sound_tokens: (B, N, D_s)  - sound tokens from SELD
        lang_tokens:  (B, T, D_l)  - language tokens from SmolVLA encoder
    Outputs:
        attn_weights: (B, N)       - per-sound-token importance (softmax)
        audio_context: (B, D_hidden) - weighted audio feature
    """

    def __init__(
        self,
        D_s: int = 256,
        D_l: int = 576,
        D_hidden: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        assert D_hidden % num_heads == 0, "D_hidden must be divisible by num_heads"
        self.num_heads = num_heads
        self.D_hidden = D_hidden
        self.head_dim = D_hidden // num_heads
        self.scale = self.head_dim ** -0.5

        # Sound tokens → Key & Value
        self.proj_k = nn.Linear(D_s, D_hidden)
        self.proj_v = nn.Linear(D_s, D_hidden)
        # Language tokens → Query
        self.proj_q = nn.Linear(D_l, D_hidden)

        self.out_proj = nn.Linear(D_hidden, D_hidden)
        self.norm = nn.LayerNorm(D_hidden)

    def forward(
        self,
        sound_tokens: torch.Tensor,
        lang_tokens: torch.Tensor,
        sound_mask: torch.Tensor = None,
    ):
        """
        Args:
            sound_tokens: (B, N, D_s)
            lang_tokens:  (B, T, D_l)
            sound_mask:   (B, N) bool — True for valid sound tokens
        Returns:
            attn_weights: (B, N)
            audio_context: (B, D_hidden)
        """
        B, N, _ = sound_tokens.shape
        _, T, _ = lang_tokens.shape

        Q = self.proj_q(lang_tokens)   # (B, T, D_hidden)
        K = self.proj_k(sound_tokens)  # (B, N, D_hidden)
        V = self.proj_v(sound_tokens)  # (B, N, D_hidden)

        # Reshape for multi-head attention
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, T, d_k)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N, d_k)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N, d_k)

        # Attention scores: (B, h, T, N)
        attn = (Q @ K.transpose(-1, -2)) * self.scale

        if sound_mask is not None:
            # Mask padding: (B, N) → (B, 1, 1, N)
            attn = attn.masked_fill(~sound_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = attn.softmax(dim=-1)  # (B, h, T, N)

        # Per-token importance: average over heads and text tokens → (B, N)
        attn_weights = attn.mean(dim=1).mean(dim=1)  # (B, N)
        attn_weights = attn_weights.softmax(dim=-1)

        # Weighted audio context: (B, h, T, d_k) → (B, T, D_hidden)
        out = (attn @ V)  # (B, h, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.D_hidden)
        audio_context = self.norm(self.out_proj(out)).mean(dim=1)  # (B, D_hidden)

        return attn_weights, audio_context
