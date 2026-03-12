"""
PyTorch Dataset classes for Audio-VLA training.

DummyAudioVLADataset  — synthetic random tensors for pipeline testing
AudioVLADataset       — loads real/simulated episodes from disk
SELDDataset           — audio-only dataset for SELD model training (이미지 불필요)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms as T


# ---------------------------------------------------------------------------
# Dummy dataset (for pipeline / unit testing)
# ---------------------------------------------------------------------------

class DummyAudioVLADataset(Dataset):
    """
    Random tensors in the correct shapes for quickly testing the training loop.
    Not for real training — replace with AudioVLADataset.
    """

    def __init__(
        self,
        num_samples: int = 256,
        audio_sr: int = 24000,
        audio_duration: float = 5.0,
        img_size: int = 512,
        N_max: int = 8,
        num_classes: int = 13,
        D_l: int = 576,
        D_v: int = 576,
        action_dim: int = 7,
        action_chunk: int = 50,
    ):
        self.n = num_samples
        self.T_samples = int(audio_sr * audio_duration)
        self.img_size  = img_size
        self.N_max     = N_max
        self.num_classes = num_classes
        self.D_l  = D_l
        self.D_v  = D_v
        self.action_dim   = action_dim
        self.action_chunk = action_chunk
        # Dummy camera intrinsic (pinhole, fx=fy=512, principal at centre)
        self.K = torch.tensor([
            [512.0,   0.0, img_size / 2],
            [  0.0, 512.0, img_size / 2],
            [  0.0,   0.0,          1.0],
        ])

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "audio":            torch.randn(2, self.T_samples),
            "image":            torch.randn(3, self.img_size, self.img_size),
            "camera_intrinsic": self.K.clone(),
            "lang_tokens":      torch.randn(20, self.D_l),   # T=20 text tokens
            "visual_features":  torch.randn(64, self.D_v),   # 64 visual tokens
            "target_sound_idx": torch.randint(0, self.N_max, (1,)).squeeze(),
            "target_actions":   torch.randn(self.action_chunk, self.action_dim),
            "command":          "Pick up the object making the siren sound.",
        }


# ---------------------------------------------------------------------------
# Real / simulated episode dataset
# ---------------------------------------------------------------------------

class AudioVLADataset(Dataset):
    """
    Load pre-collected simulation or real episodes from disk.

    Expected episode directory layout:
        episodes/
            0000/
                audio.wav          (2ch, 24 kHz)
                image.png          (RGB)
                meta.json          {command, target_sound_idx, target_actions,
                                    camera_intrinsic, camera_extrinsic,
                                    peak_coords, class_logits, energy}
            0001/
            ...

    meta.json fields:
        command:          str
        target_sound_idx: int  — index of the target sound source
        target_actions:   list[list[float]]  — shape (50, action_dim)
        camera_intrinsic: list[list[float]]  — shape (3, 3)
        camera_extrinsic: list[list[float]] or null  — shape (4, 4)
        lang_tokens:      optional; if absent, will be computed at load time
        visual_features:  optional; if absent, will be computed at load time
    """

    def __init__(
        self,
        data_root: str,
        audio_sr: int = 24000,
        img_size: int = 512,
        D_l: int = 576,
        D_v: int = 576,
        use_augmentation: bool = False,
        split: str = "train",
        train_fraction: float = 0.9,
    ):
        self.data_root = Path(data_root)
        self.audio_sr  = audio_sr
        self.img_size  = img_size
        self.D_l = D_l
        self.D_v = D_v

        all_eps = sorted(self.data_root.glob("*/meta.json"))
        n_train = int(len(all_eps) * train_fraction)
        if split == "train":
            self.episodes = all_eps[:n_train]
        else:
            self.episodes = all_eps[n_train:]

        # Image augmentation
        aug_list = [T.Resize((img_size, img_size)), T.ToTensor()]
        if use_augmentation:
            aug_list = [
                T.Resize((img_size + 32, img_size + 32)),
                T.RandomCrop(img_size),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor(),
            ]
        self.img_transform = T.Compose(aug_list)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        meta_path = self.episodes[idx]
        ep_dir    = meta_path.parent

        # Load meta
        with open(meta_path) as f:
            meta = json.load(f)

        # Audio
        audio_path = ep_dir / "audio.wav"
        waveform, sr = torchaudio.load(str(audio_path))  # (2, T)
        if sr != self.audio_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.audio_sr)

        # Image
        from PIL import Image
        img_path = ep_dir / "image.png"
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.img_transform(img)  # (3, H, W)

        # Camera
        K = torch.tensor(meta["camera_intrinsic"], dtype=torch.float32)  # (3, 3)
        extr = None
        if meta.get("camera_extrinsic"):
            extr = torch.tensor(meta["camera_extrinsic"], dtype=torch.float32)

        # Pre-extracted features (if available) or placeholder zeros
        if "lang_tokens" in meta:
            lang_tokens = torch.tensor(meta["lang_tokens"], dtype=torch.float32)
        else:
            lang_tokens = torch.zeros(20, self.D_l)

        if "visual_features" in meta:
            visual_features = torch.tensor(meta["visual_features"], dtype=torch.float32)
        else:
            visual_features = torch.zeros(64, self.D_v)

        target_actions = torch.tensor(meta["target_actions"], dtype=torch.float32)

        item = {
            "audio":            waveform,
            "image":            img_tensor,
            "camera_intrinsic": K,
            "lang_tokens":      lang_tokens,
            "visual_features":  visual_features,
            "target_sound_idx": torch.tensor(meta["target_sound_idx"], dtype=torch.long),
            "target_actions":   target_actions,
            "command":          meta["command"],
        }
        if extr is not None:
            item["camera_extrinsic"] = extr

        return item


# ---------------------------------------------------------------------------
# SELD-only dataset (이미지 불필요 — audio + spatial labels만 사용)
# ---------------------------------------------------------------------------

class SELDDataset(Dataset):
    """
    SELD 모델 학습용 오디오 전용 데이터셋.

    episode_collector가 생성한 에피소드에서 audio.wav와 meta.json만 읽습니다.
    이미지는 로드하지 않아 SELD 학습 시 불필요한 I/O를 제거합니다.

    타겟 포맷: EINv2 / ACCDOA
      shape: (N_max, num_classes, 3)
      각 track i, class c에 대해:
        - source i의 클래스가 c이면: target[i, c] = [cos(el)*sin(az), -sin(el), cos(el)*cos(az)]
        - 그 외: [0, 0, 0]
    """

    def __init__(
        self,
        data_root: str,
        audio_sr: int = 24000,
        num_classes: int = 200,
        N_max: int = 8,
        split: str = "train",
        train_fraction: float = 0.9,
    ):
        self.data_root   = Path(data_root)
        self.audio_sr    = audio_sr
        self.num_classes = num_classes
        self.N_max       = N_max

        all_eps = sorted(self.data_root.glob("*/meta.json"))
        n_train = int(len(all_eps) * train_fraction)
        self.episodes = all_eps[:n_train] if split == "train" else all_eps[n_train:]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        meta_path = self.episodes[idx]
        ep_dir    = meta_path.parent

        with open(meta_path) as f:
            meta = json.load(f)

        # ── 오디오 로드 ────────────────────────────────────────────────────────
        waveform, sr = torchaudio.load(str(ep_dir / "audio.wav"))  # (2, T)
        if sr != self.audio_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.audio_sr)

        # ── ACCDOA 타겟 생성 ───────────────────────────────────────────────────
        # objects: [{class_idx, az_rad, el_rad}, ...]
        objects = meta.get("objects", [])
        target  = torch.zeros(self.N_max, self.num_classes, 3)

        for track_i, obj in enumerate(objects[:self.N_max]):
            cls = int(obj["class_idx"])
            az  = float(obj["az_rad"])
            el  = float(obj["el_rad"])
            cos_el = math.cos(el)
            xyz = torch.tensor([
                cos_el * math.sin(az),
                -math.sin(el),
                cos_el * math.cos(az),
            ])
            target[track_i, cls] = xyz  # activity-coupled direction

        return {"audio": waveform, "target_accdoa": target}
