"""
AudioPrefixEncoder: SELD 출력을 언어 prefix 토큰으로 변환.

각 sound source를 D_l 차원의 토큰으로 인코딩하여 SmolVLA 언어 시퀀스 앞에 prepend.
- in-frame source: 정규화된 2D 픽셀 좌표 (u/W, v/H) ∈ [0,1] 사용
- out-of-frame source: 정규화된 az/el 각도 사용 + frame_type_emb으로 구분 신호 제공

LLM이 "두 소리 사이"와 같은 공간 관계 추론을 할 수 있도록 명시적 위치 정보 제공.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AudioPrefixEncoder(nn.Module):
    """
    SELD 출력 → 언어 prefix 토큰 (B, N, D_l).

    SmolVLAWrapper.encode_language()에서 텍스트 토큰 앞에 prepend하여
    언어 디코더가 공간 관계 추론 시 소리 위치 정보를 직접 활용할 수 있게 함.

    Args:
        D_l:         SmolVLA 언어 토큰 차원 (default: 576)
        num_classes: SELD 클래스 수
        D_s:         중간 hidden 차원
    """

    def __init__(self, D_l: int = 576, num_classes: int = 13, D_s: int = 256):
        super().__init__()
        self.D_l = D_l

        # 위치 인코딩: 정규화 좌표 (2,) → D_l
        # in-frame: (u/W, v/H) ∈ [0, 1]
        # out-of-frame: ((az+π)/(2π), (el+π/2)/π) ∈ [0, 1]
        self.pos_proj = nn.Sequential(
            nn.Linear(2, D_s),
            nn.GELU(),
            nn.Linear(D_s, D_l),
        )

        # 클래스 임베딩: soft logits → D_l
        self.class_proj = nn.Sequential(
            nn.Linear(num_classes, D_s),
            nn.GELU(),
            nn.Linear(D_s, D_l),
        )

        # 에너지 임베딩: scalar → D_l
        self.energy_proj = nn.Sequential(
            nn.Linear(1, D_s),
            nn.GELU(),
            nn.Linear(D_s, D_l),
        )

        # frame type embedding: 0 = out-of-frame, 1 = in-frame
        # LLM에게 "이 좌표가 이미지 내부인지 외부 방향인지" 알려주는 신호
        self.frame_type_emb = nn.Embedding(2, D_l)

        # 4가지 임베딩 융합 → D_l
        self.fuse = nn.Sequential(
            nn.Linear(D_l * 4, D_l),
            nn.LayerNorm(D_l),
            nn.GELU(),
            nn.Linear(D_l, D_l),
        )

    def forward(
        self,
        pixel_coords: torch.Tensor,
        az_el_coords: torch.Tensor,
        class_logits: torch.Tensor,
        energy: torch.Tensor,
        in_frame: torch.Tensor,
        valid_mask: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """
        Args:
            pixel_coords:  (B, N, 2) — AzElToPixel 출력 픽셀 좌표 (raw)
            az_el_coords:  (B, N, 2) — SELD peak_coords (azimuth, elevation in radians)
            class_logits:  (B, N, C) — SELD class logits
            energy:        (B, N, 1) — SELD energy
            in_frame:      (B, N)    — bool, camera FOV 내 여부
            valid_mask:    (B, N)    — bool, 실제 감지된 sound source 여부
            img_h, img_w:  이미지 해상도 (픽셀 정규화용)

        Returns:
            prefix_tokens: (B, N, D_l) — 언어 시퀀스 앞에 prepend할 토큰
                           invalid source (valid_mask=False) 위치는 0으로 마스킹됨
        """
        B, N, _ = az_el_coords.shape
        device = az_el_coords.device

        # in-frame 좌표: 픽셀 → [0, 1] 정규화
        px_norm = torch.stack([
            pixel_coords[..., 0] / (img_w - 1),  # u / (W-1)
            pixel_coords[..., 1] / (img_h - 1),  # v / (H-1)
        ], dim=-1).clamp(0.0, 1.0)  # (B, N, 2)

        # out-of-frame 좌표: az/el → [0, 1] 정규화
        az_norm = (az_el_coords[..., 0] + math.pi) / (2 * math.pi)       # [-π, π] → [0,1]
        el_norm = (az_el_coords[..., 1] + math.pi / 2) / math.pi          # [-π/2, π/2] → [0,1]
        azel_norm = torch.stack([az_norm, el_norm], dim=-1)  # (B, N, 2)

        # in_frame에 따라 좌표 선택
        in_frame_f = in_frame.float().unsqueeze(-1)  # (B, N, 1)
        coords = in_frame_f * px_norm + (1.0 - in_frame_f) * azel_norm   # (B, N, 2)

        # 각 임베딩 계산
        p = self.pos_proj(coords)                        # (B, N, D_l)
        c = self.class_proj(class_logits)                # (B, N, D_l)
        e = self.energy_proj(energy)                     # (B, N, D_l)
        t = self.frame_type_emb(in_frame.long())         # (B, N, D_l)

        tokens = self.fuse(torch.cat([p, c, e, t], dim=-1))  # (B, N, D_l)

        # invalid source 마스킹 (패딩 소스는 0 벡터)
        tokens = tokens * valid_mask.float().unsqueeze(-1)  # (B, N, D_l)

        return tokens
