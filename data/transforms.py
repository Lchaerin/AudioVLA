"""
Audio and image augmentation transforms for Audio-VLA training.
"""

import random

import torch
import torchaudio.functional as AF


class RandomAudioGain:
    """Randomly scale audio amplitude by ±6 dB."""

    def __init__(self, gain_db_range: tuple[float, float] = (-6.0, 6.0)):
        self.lo, self.hi = gain_db_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        gain_db = random.uniform(self.lo, self.hi)
        gain    = 10 ** (gain_db / 20.0)
        return waveform * gain


class RandomAudioNoise:
    """Add Gaussian noise at a random SNR level."""

    def __init__(self, snr_db_range: tuple[float, float] = (20.0, 40.0)):
        self.lo, self.hi = snr_db_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        snr_db  = random.uniform(self.lo, self.hi)
        signal_power = waveform.pow(2).mean()
        noise_power  = signal_power / (10 ** (snr_db / 10.0))
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return waveform + noise


class RandomAudioCrop:
    """Randomly crop a segment of fixed duration from a longer audio clip."""

    def __init__(self, target_samples: int):
        self.target_samples = target_samples

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        T = waveform.shape[-1]
        if T <= self.target_samples:
            # Pad with zeros if shorter
            pad = self.target_samples - T
            return torch.nn.functional.pad(waveform, (0, pad))
        start = random.randint(0, T - self.target_samples)
        return waveform[..., start : start + self.target_samples]


class ChannelSwap:
    """Randomly swap left/right channels of binaural audio (data augmentation)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (2, T)
        if random.random() < self.p:
            return waveform.flip(0)
        return waveform


class AudioCompose:
    """Compose a list of audio transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            waveform = t(waveform)
        return waveform


def get_audio_transforms(augment: bool, target_samples: int) -> AudioCompose:
    """Return a composed audio transform pipeline."""
    base = [RandomAudioCrop(target_samples)]
    if augment:
        base += [
            RandomAudioGain((-6.0, 6.0)),
            RandomAudioNoise((25.0, 45.0)),
            ChannelSwap(p=0.3),
        ]
    return AudioCompose(base)
