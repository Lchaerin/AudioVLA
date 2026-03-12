"""
SELD model: ResNet-Conformer + EINv2 architecture.
Based on DCASE 2024/2025 baseline.
Input:  Binaural audio (2ch) → log-mel spectrogram + ILD/IPD features
Output: Multi-ACCDOA (simultaneous detection of multiple sound events)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from .seld_output import SELDOutput


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class BinauralFeatureExtractor(nn.Module):
    """
    Extract log-mel spectrogram + ILD/IPD from binaural (2-channel) audio.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 64,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, 2, T_samples) - binaural waveform
        Returns:
            features: (B, C_feat, T_frames, n_mels)
                where C_feat = n_mels*2 (log-mel L & R) + n_fft//2+1 (ILD) + n_fft//2+1 (IPD)
        """
        B = audio.shape[0]
        left = audio[:, 0, :]   # (B, T)
        right = audio[:, 1, :]  # (B, T)

        # Log-mel spectrograms for each channel
        mel_l = torch.log1p(self.mel_spec(left))   # (B, n_mels, T_frames)
        mel_r = torch.log1p(self.mel_spec(right))  # (B, n_mels, T_frames)

        # ILD / IPD via STFT
        stft_l = torch.stft(left, self.n_fft, self.hop_length, return_complex=True)   # (B, F, T_frames)
        stft_r = torch.stft(right, self.n_fft, self.hop_length, return_complex=True)  # (B, F, T_frames)

        ild = 20 * torch.log10((stft_r.abs() + 1e-8) / (stft_l.abs() + 1e-8))  # (B, F, T)
        ipd = torch.angle(stft_l * stft_r.conj())                                # (B, F, T)

        # Permute to (B, C, T, F) layout
        mel_l = mel_l.permute(0, 2, 1)  # (B, T, n_mels)
        mel_r = mel_r.permute(0, 2, 1)
        ild   = ild.permute(0, 2, 1)    # (B, T, F)
        ipd   = ipd.permute(0, 2, 1)

        # Concatenate along feature axis then unsqueeze channel dim
        features = torch.cat([mel_l, mel_r, ild, ipd], dim=-1)  # (B, T, C_feat)
        return features


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class ConformerBlock(nn.Module):
    """Simplified Conformer block (Feed-Forward → MHSA → Conv → Feed-Forward)."""

    def __init__(self, d_model: int, num_heads: int = 4, ff_mult: int = 4, kernel_size: int = 31):
        super().__init__()
        self.ff1   = self._ff(d_model, ff_mult)
        self.attn  = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1),
            # Depthwise conv applied on the time axis
        )
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.conv_norm = nn.LayerNorm(d_model)
        self.ff2   = self._ff(d_model, ff_mult)
        self.norm_out = nn.LayerNorm(d_model)

    @staticmethod
    def _ff(d: int, mult: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * mult),
            nn.SiLU(),
            nn.Linear(d * mult, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + 0.5 * self.ff1(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Conformer conv module
        conv_in = self.conv(x)  # (B, T, D)
        conv_in = self.dw_conv(conv_in.transpose(1, 2)).transpose(1, 2)  # depthwise
        x = x + self.conv_norm(conv_in)
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x)


# ---------------------------------------------------------------------------
# EINv2-style output head (Multi-ACCDOA)
# ---------------------------------------------------------------------------

class EINv2Head(nn.Module):
    """
    Multi-ACCDOA output head.
    Produces per-track activity + direction vectors.
    """

    def __init__(self, d_model: int, num_classes: int, n_tracks: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.n_tracks = n_tracks
        # For each track: predict (x, y, z) activity-weighted direction + energy
        self.heads = nn.ModuleList([
            nn.Linear(d_model, num_classes * 3 + 1)  # xyz per class + energy
            for _ in range(n_tracks)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, D)
        Returns:
            List[dict] with per-track tensors, each shaped (B, T, num_classes, ...)
        """
        outputs = []
        for head in self.heads:
            out = head(x)  # (B, T, num_classes*3 + 1)
            xyz = out[..., :self.num_classes * 3].view(*out.shape[:-1], self.num_classes, 3)
            energy = out[..., -1:]  # (B, T, 1)
            outputs.append({"xyz": xyz, "energy": energy})
        return outputs


# ---------------------------------------------------------------------------
# Full SELD Model
# ---------------------------------------------------------------------------

class ResNetConformerSELD(nn.Module):
    """
    ResNet-Conformer + EINv2 SELD model.
    Input:  (B, 2, T_samples) binaural audio
    Output: SELDOutput with peak_coords, class_logits, energy, valid_mask
    """

    def __init__(
        self,
        num_classes: int = 13,
        n_max_sources: int = 8,
        sample_rate: int = 24000,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 256,
        d_model: int = 256,
        num_conformer_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_max_sources = n_max_sources

        # Feature extraction
        self.feature_extractor = BinauralFeatureExtractor(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        in_feat = n_mels * 2 + (n_fft // 2 + 1) * 2  # log-mel L+R + ILD + IPD

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_feat, d_model),
            nn.LayerNorm(d_model),
        )

        # ResNet-style 2D encoder (treats (T, F) as the 2D space)
        self.resnet = nn.Sequential(
            ResidualBlock2D(1, 32, stride=(1, 2)),
            ResidualBlock2D(32, 64, stride=(1, 2)),
            ResidualBlock2D(64, 128, stride=(1, 2)),
        )
        # After 3 halving strides on the frequency axis:
        # freq_out = n_mels // 8  (if using mel channels from input_proj)
        # Use adaptive pooling so we get a fixed-size frequency dim
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 8))  # (B, C, T, 8)
        self.flat_proj = nn.Linear(128 * 8, d_model)

        # Conformer layers
        self.conformer = nn.Sequential(
            *[ConformerBlock(d_model, num_heads=num_heads) for _ in range(num_conformer_layers)]
        )

        # EINv2-style output head (multi-track ACCDOA)
        self.output_head = EINv2Head(d_model, num_classes, n_tracks=n_max_sources)

    def forward(self, audio: torch.Tensor) -> SELDOutput:
        """
        Args:
            audio: (B, 2, T_samples)
        Returns:
            SELDOutput
        """
        B = audio.shape[0]

        # Feature extraction: (B, T, C_feat)
        feats = self.feature_extractor(audio)  # (B, T, C_feat)
        feats = self.input_proj(feats)          # (B, T, d_model)

        # Treat as 2D (time × model_dim) for ResNet
        # Reshape: (B, 1, T, d_model) → ResNet → (B, C, T, F)
        x2d = feats.unsqueeze(1)               # (B, 1, T, d_model)
        x2d = self.resnet(x2d)                 # (B, 128, T, F')
        x2d = self.freq_pool(x2d)              # (B, 128, T, 8)
        T_out = x2d.shape[2]
        x2d = x2d.permute(0, 2, 1, 3).reshape(B, T_out, -1)  # (B, T, 128*8)
        x = self.flat_proj(x2d)               # (B, T, d_model)

        # Conformer
        x = self.conformer(x)  # (B, T, d_model)

        # Per-track output heads → aggregate over time → peaks
        track_outputs = self.output_head(x)   # list of {xyz, energy}

        # Aggregate each track over time with max-energy frame selection
        peak_coords_list = []
        class_logits_list = []
        energy_list = []

        for track in track_outputs:
            xyz    = track["xyz"]     # (B, T, C, 3)
            energy = track["energy"]  # (B, T, 1)

            # Select frame with highest energy for this track
            t_idx = energy.squeeze(-1).argmax(dim=1)  # (B,)
            best_xyz    = xyz[torch.arange(B), t_idx]      # (B, C, 3)
            best_energy = energy[torch.arange(B), t_idx]   # (B, 1)

            # Dominant class
            activity = best_xyz.norm(dim=-1)  # (B, C) - activity per class
            class_logits_list.append(activity)  # use activity as logits

            # Weighted direction → (az, el)
            dominant_class = activity.argmax(dim=-1)  # (B,)
            direction = best_xyz[torch.arange(B), dominant_class]  # (B, 3)
            direction = F.normalize(direction, dim=-1)

            az = torch.atan2(direction[:, 0], direction[:, 2])  # (B,)
            el = torch.asin((-direction[:, 1]).clamp(-1, 1))     # (B,)

            peak_coords_list.append(torch.stack([az, el], dim=-1))  # (B, 2)
            energy_list.append(best_energy)

        # Stack to (B, N_max, ...)
        peak_coords  = torch.stack(peak_coords_list,  dim=1)  # (B, N, 2)
        class_logits = torch.stack(class_logits_list, dim=1)  # (B, N, C)
        energy_out   = torch.stack(energy_list,       dim=1)  # (B, N, 1)

        # Valid mask: tracks with energy above threshold
        valid_mask = energy_out.squeeze(-1) > 0.1  # (B, N)

        return SELDOutput(
            peak_coords=peak_coords,
            class_logits=class_logits,
            energy=energy_out,
            valid_mask=valid_mask,
        )
