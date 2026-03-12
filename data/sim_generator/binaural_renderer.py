"""
Binaural audio renderer using Room Impulse Response (RIR) convolution.

Given a mono source waveform and a target (azimuth, elevation) in the scene,
this module generates a binaural (2-channel) signal using Head-Related Transfer
Functions (HRTFs) or simulated RIRs via SpatialScaper / pyroomacoustics.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def az_el_to_unit_vector(az_rad: float, el_rad: float) -> np.ndarray:
    """Convert (azimuth, elevation) in radians to a unit vector (x, y, z)."""
    x = math.cos(el_rad) * math.sin(az_rad)
    y = -math.sin(el_rad)
    z = math.cos(el_rad) * math.cos(az_rad)
    return np.array([x, y, z], dtype=np.float32)


class SimpleBinauralRenderer:
    """
    A lightweight binaural renderer using ITD/ILD approximations.

    For real experiments, replace with:
      - SpatialScaper (https://github.com/iranroman/SpatialScaper)
      - pyroomacoustics (https://github.com/LCAV/pyroomacoustics)
      - HRIR convolution with a measured HRTF database (e.g. CIPIC)

    This implementation approximates:
      - ITD (Interaural Time Difference) from head radius model
      - ILD (Interaural Level Difference) from cosine rule
    """

    HEAD_RADIUS_M = 0.0875   # average human head radius in metres
    SOUND_SPEED_MS = 343.0   # m/s

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def _compute_itd_samples(self, az_rad: float) -> int:
        """Estimate ITD (in samples) using Woodworth's formula."""
        # Woodworth: ITD = (r/c) * (az + sin(az))
        itd_sec = (self.HEAD_RADIUS_M / self.SOUND_SPEED_MS) * (az_rad + math.sin(az_rad))
        return int(itd_sec * self.sample_rate)

    def _compute_ild_db(self, az_rad: float, el_rad: float) -> float:
        """Approximate ILD in dB using a simple cosine model."""
        # Sound from the right (az > 0) is louder in right ear
        return 6.0 * math.sin(az_rad) * math.cos(el_rad)

    def render(
        self,
        mono_waveform: torch.Tensor,
        az_rad: float,
        el_rad: float = 0.0,
        room_rir: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Render a mono waveform to binaural using ITD + ILD.

        Args:
            mono_waveform: (T,) mono waveform
            az_rad:        azimuth in radians  (-π to π, 0 = forward)
            el_rad:        elevation in radians (-π/2 to π/2)
            room_rir:      optional (2, T_rir) room impulse response for each ear;
                           if provided, used instead of simple delay/gain model
        Returns:
            binaural: (2, T) left and right channels
        """
        T = mono_waveform.shape[-1]

        if room_rir is not None:
            # Convolve with measured/simulated RIR
            rir_len = room_rir.shape[-1]
            src = mono_waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            left_rir  = room_rir[0].unsqueeze(0).unsqueeze(0)  # (1, 1, T_rir)
            right_rir = room_rir[1].unsqueeze(0).unsqueeze(0)

            left  = F.conv1d(src, left_rir.flip(-1),  padding=rir_len - 1)[0, 0, :T]
            right = F.conv1d(src, right_rir.flip(-1), padding=rir_len - 1)[0, 0, :T]
            return torch.stack([left, right], dim=0)

        # Simple ITD / ILD model
        itd = self._compute_itd_samples(az_rad)
        ild_db = self._compute_ild_db(az_rad, el_rad)
        ild_linear = 10 ** (ild_db / 20.0)

        # Positive ITD → sound from right → left ear delayed
        left  = mono_waveform.clone()
        right = mono_waveform.clone()

        if itd > 0:
            # Right-side source: delay left ear by |itd|, amplify right ear
            left  = F.pad(left, (itd, 0))[..., :T]
            right = right * ild_linear
        elif itd < 0:
            itd = abs(itd)
            right = F.pad(right, (itd, 0))[..., :T]
            left  = left * ild_linear

        return torch.stack([left, right], dim=0)  # (2, T)


class MultisourceBinauralMixer:
    """
    Mix multiple mono sources at different spatial positions into a binaural signal.
    """

    def __init__(self, sample_rate: int = 24000):
        self.renderer = SimpleBinauralRenderer(sample_rate=sample_rate)

    def mix(
        self,
        sources: list[dict],
        target_samples: int,
    ) -> torch.Tensor:
        """
        Args:
            sources: list of dicts with keys:
                waveform: (T,) tensor
                az:       float azimuth in radians
                el:       float elevation in radians (default 0)
                gain:     float linear gain (default 1.0)
            target_samples: desired output length
        Returns:
            mix: (2, target_samples)
        """
        mix = torch.zeros(2, target_samples)
        for src in sources:
            wav  = src["waveform"]
            az   = src.get("az", 0.0)
            el   = src.get("el", 0.0)
            gain = src.get("gain", 1.0)

            # Crop or pad mono source
            if wav.shape[-1] < target_samples:
                wav = F.pad(wav, (0, target_samples - wav.shape[-1]))
            else:
                wav = wav[:target_samples]

            binaural = self.renderer.render(wav, az, el) * gain  # (2, T)
            mix += binaural

        # Normalise to prevent clipping
        peak = mix.abs().max()
        if peak > 1.0:
            mix = mix / peak

        return mix
