"""
Binaural audio renderer for Audio-VLA episode generation.

두 가지 렌더러를 제공합니다:
  - SOFABinauralRenderer: data/hrir/*.sofa 파일 기반 실측 HRTF 컨볼루션 (권장)
  - SimpleBinauralRenderer: ITD/ILD 근사 (SOFA 파일 없을 때 fallback)

SOFAPool은 여러 SOFA 파일을 관리하고 에피소드마다 다른 피험자의 HRTF를
랜덤 선택하여 데이터 다양성을 높입니다.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── SOFA 기반 렌더러 ───────────────────────────────────────────────────────────

class SOFABinauralRenderer:
    """
    실측 HRTF를 이용한 바이노럴 렌더러.

    data/hrir/ 내 IRCAM Listen 데이터베이스 (IRC_XXXX_R_44100.sofa) 사용.
    각 소스 위치에 대해 가장 가까운 측정점의 HRTF를 찾아 컨볼루션합니다.

    좌표 규약:
      입력 (본 프로젝트 규약): az=0 정면, az > 0 오른쪽, 라디안
      IRCAM 규약: az=0 정면, az=90 왼쪽 (반시계), 도 단위
      → 변환: ircam_az_deg = -degrees(our_az_rad)
    """

    def __init__(self, sofa_path: str, target_sr: int = 24000):
        self.target_sr = target_sr
        self._load_sofa(sofa_path)

    def _load_sofa(self, path: str) -> None:
        """SOFA 파일 로드 및 HRTF 준비."""
        try:
            import netCDF4 as nc
        except ImportError:
            raise ImportError("netCDF4 is required. Install with: pip install netCDF4")

        try:
            from scipy.signal import resample_poly
            from math import gcd
        except ImportError:
            raise ImportError("scipy is required. Install with: pip install scipy")

        with nc.Dataset(path, "r") as f:
            # SourcePosition: (M, 3) — [azimuth_deg, elevation_deg, radius_m]
            positions = np.array(f.variables["SourcePosition"][:], dtype=np.float32)
            # Data.IR: (M, R, N) — R=2 (left, right), N=IR length
            hrir = np.array(f.variables["Data.IR"][:], dtype=np.float32)
            sr_var = f.variables["Data.SamplingRate"][:]
            hrir_sr = int(np.asarray(sr_var).flat[0])

        # HRTF 샘플레이트 변환 (44100 → target_sr)
        if hrir_sr != self.target_sr:
            g = gcd(self.target_sr, hrir_sr)
            up, down = self.target_sr // g, hrir_sr // g
            resampled = np.stack([
                np.stack([
                    resample_poly(hrir[m, 0], up, down).astype(np.float32),
                    resample_poly(hrir[m, 1], up, down).astype(np.float32),
                ], axis=0)
                for m in range(hrir.shape[0])
            ], axis=0)  # (M, 2, N_new)
            self._hrir = resampled
        else:
            self._hrir = hrir  # (M, 2, N)

        # 위치를 프로젝트 규약(라디안, 오른쪽 양수)으로 변환하여 캐싱
        # IRCAM: az=0 front, 반시계 양수(왼쪽 양수) → 우리: 시계 양수(오른쪽 양수)
        az_deg = positions[:, 0]
        el_deg = positions[:, 1]
        self._pos_az = np.radians(-az_deg)  # (M,) 부호 반전
        self._pos_el = np.radians(el_deg)   # (M,)

    def _find_nearest_hrtf(self, az_rad: float, el_rad: float) -> int:
        """구면 각도 거리 기반 최근접 HRTF 측정점 인덱스 반환."""
        # cos(el) 가중치를 준 방위각 차이 + 앙각 차이로 근사 거리 계산
        daz = self._pos_az - az_rad
        del_ = self._pos_el - el_rad
        dist = np.cos(el_rad) ** 2 * daz ** 2 + del_ ** 2
        return int(np.argmin(dist))

    def render(
        self,
        mono_waveform: torch.Tensor,
        az_rad: float,
        el_rad: float = 0.0,
    ) -> torch.Tensor:
        """
        모노 파형 → 바이노럴 (2채널) HRTF 컨볼루션.

        Args:
            mono_waveform: (T,) 모노 파형 텐서
            az_rad:        방위각 라디안 (0=정면, 양수=오른쪽)
            el_rad:        앙각 라디안 (0=수평, 양수=위)
        Returns:
            binaural: (2, T) 좌/우 채널
        """
        from scipy.signal import fftconvolve

        mono_np = mono_waveform.cpu().numpy().astype(np.float32)
        T = len(mono_np)

        idx = self._find_nearest_hrtf(az_rad, el_rad)
        hrir_l = self._hrir[idx, 0]
        hrir_r = self._hrir[idx, 1]

        left  = fftconvolve(mono_np, hrir_l)[:T].astype(np.float32)
        right = fftconvolve(mono_np, hrir_r)[:T].astype(np.float32)

        return torch.from_numpy(np.stack([left, right], axis=0))


class SOFAPool:
    """
    여러 SOFA 파일을 관리하는 풀.

    에피소드마다 다른 SOFA 파일(피험자)을 랜덤 선택하여
    HRTF 다양성을 확보합니다. 로드된 렌더러는 메모리에 캐싱합니다.
    """

    def __init__(self, hrir_dir: str, target_sr: int = 24000):
        self.target_sr = target_sr
        hrir_path = Path(hrir_dir)
        self._sofa_paths = sorted(hrir_path.glob("*.sofa"))
        if not self._sofa_paths:
            raise FileNotFoundError(
                f"SOFA 파일을 찾을 수 없습니다: {hrir_dir}\n"
                f"data/hrir/ 디렉터리에 .sofa 파일이 있는지 확인하세요."
            )
        log.info(f"SOFAPool: {len(self._sofa_paths)}개 SOFA 파일 발견 ({hrir_dir})")
        self._cache: dict[str, SOFABinauralRenderer] = {}

    def get_random(self) -> SOFABinauralRenderer:
        """랜덤 SOFA 파일의 렌더러 반환 (캐시 사용)."""
        path = random.choice(self._sofa_paths)
        key = str(path)
        if key not in self._cache:
            log.debug(f"SOFA 로드: {path.name}")
            self._cache[key] = SOFABinauralRenderer(key, self.target_sr)
        return self._cache[key]

    @property
    def n_subjects(self) -> int:
        return len(self._sofa_paths)


# ── ITD/ILD 근사 렌더러 (fallback) ────────────────────────────────────────────

class SimpleBinauralRenderer:
    """
    ITD/ILD 근사를 이용한 경량 바이노럴 렌더러 (fallback).

    SOFA 파일이 없거나 빠른 테스트가 필요할 때 사용합니다.
    물리적 정확도는 SOFABinauralRenderer보다 낮습니다.
    """

    HEAD_RADIUS_M  = 0.0875
    SOUND_SPEED_MS = 343.0

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def _compute_itd_samples(self, az_rad: float) -> int:
        itd_sec = (self.HEAD_RADIUS_M / self.SOUND_SPEED_MS) * (az_rad + math.sin(az_rad))
        return int(itd_sec * self.sample_rate)

    def _compute_ild_db(self, az_rad: float, el_rad: float) -> float:
        return 6.0 * math.sin(az_rad) * math.cos(el_rad)

    def render(
        self,
        mono_waveform: torch.Tensor,
        az_rad: float,
        el_rad: float = 0.0,
        room_rir: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = mono_waveform.shape[-1]

        if room_rir is not None:
            rir_len = room_rir.shape[-1]
            src       = mono_waveform.unsqueeze(0).unsqueeze(0)
            left_rir  = room_rir[0].unsqueeze(0).unsqueeze(0)
            right_rir = room_rir[1].unsqueeze(0).unsqueeze(0)
            left  = F.conv1d(src, left_rir.flip(-1),  padding=rir_len - 1)[0, 0, :T]
            right = F.conv1d(src, right_rir.flip(-1), padding=rir_len - 1)[0, 0, :T]
            return torch.stack([left, right], dim=0)

        itd = self._compute_itd_samples(az_rad)
        ild_linear = 10 ** (self._compute_ild_db(az_rad, el_rad) / 20.0)

        left  = mono_waveform.clone()
        right = mono_waveform.clone()

        if itd > 0:
            left  = F.pad(left,  (itd, 0))[..., :T]
            right = right * ild_linear
        elif itd < 0:
            itd = abs(itd)
            right = F.pad(right, (itd, 0))[..., :T]
            left  = left * ild_linear

        return torch.stack([left, right], dim=0)


# ── 멀티소스 믹서 ─────────────────────────────────────────────────────────────

class MultisourceBinauralMixer:
    """
    여러 모노 소스를 서로 다른 공간 위치에 배치하여 바이노럴 믹스 생성.

    renderer로 SOFAPool(권장) 또는 SimpleBinauralRenderer(fallback)를 전달합니다.
    SOFAPool이 주어지면 에피소드마다 새 SOFA 파일이 랜덤 선택됩니다.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        renderer: Optional[SOFAPool | SimpleBinauralRenderer] = None,
    ):
        self.sample_rate = sample_rate
        self._renderer_src = renderer  # SOFAPool 또는 SimpleBinauralRenderer

        # renderer가 없으면 SimpleBinauralRenderer fallback
        if renderer is None:
            log.warning(
                "SOFAPool이 제공되지 않아 SimpleBinauralRenderer(ITD/ILD)를 사용합니다. "
                "더 정확한 시뮬레이션을 위해 --hrir_dir를 지정하세요."
            )
            self._fallback = SimpleBinauralRenderer(sample_rate=sample_rate)
        else:
            self._fallback = None

    def _get_renderer(self):
        """에피소드 단위로 렌더러 반환 (SOFAPool이면 랜덤 선택)."""
        if self._renderer_src is None:
            return self._fallback
        if isinstance(self._renderer_src, SOFAPool):
            return self._renderer_src.get_random()
        return self._renderer_src  # SimpleBinauralRenderer 직접 전달된 경우

    def mix(
        self,
        sources: list[dict],
        target_samples: int,
    ) -> torch.Tensor:
        """
        Args:
            sources: list of dicts:
                waveform: (T,) 모노 텐서
                az:       float 방위각 라디안 (0=정면, 양수=오른쪽)
                el:       float 앙각 라디안 (default 0)
                gain:     float 선형 게인 (default 1.0)
            target_samples: 출력 샘플 수
        Returns:
            mix: (2, target_samples) 바이노럴 믹스
        """
        renderer = self._get_renderer()
        mix = torch.zeros(2, target_samples)

        for src in sources:
            wav  = src["waveform"]
            az   = float(src.get("az",   0.0))
            el   = float(src.get("el",   0.0))
            gain = float(src.get("gain", 1.0))

            # 길이 조정
            if wav.shape[-1] < target_samples:
                wav = F.pad(wav, (0, target_samples - wav.shape[-1]))
            else:
                wav = wav[:target_samples]

            binaural = renderer.render(wav, az, el) * gain  # (2, T)
            mix += binaural

        # 클리핑 방지 피크 정규화
        peak = mix.abs().max()
        if peak > 1.0:
            mix = mix / peak

        return mix
