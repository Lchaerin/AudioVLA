#!/usr/bin/env python3
"""
FSD50K (Freesound Dataset 50K) 다운로드 및 SELD 클래스별 정리 스크립트.

Zenodo record 4060432에서 FSD50K dev 세트를 다운로드하고,
200개 SELD 클래스에 해당하는 오디오 파일을
data/sound_effects/{class_name}/ 형식으로 정리합니다.

【파일 구조】
  FSD50K.ground_truth.zip         ~334 KB  (레이블 CSV)
  FSD50K.dev_audio.zip            ~2.1 GB  (분할 zip 헤더)
  FSD50K.dev_audio.z01 ~ .z05    ~3 GB × 5 (분할 zip 본체)
  합계: ~17 GB 다운로드 → 최종 ~3-5 GB 보존

【필수 외부 도구】
  7z (p7zip): 분할 zip 추출에 사용
    설치: conda install -c conda-forge p7zip
          또는: sudo apt-get install p7zip-full

Usage:
    python scripts/download_sound_effects.py
    python scripts/download_sound_effects.py --max_per_class 300
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────────

ZENODO_RECORD_ID = "4060432"
ZENODO_API_URL   = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# 다운로드할 파일 목록 (Zenodo record 4060432 실제 파일명)
GROUND_TRUTH_KEY = "FSD50K.ground_truth.zip"
DEV_AUDIO_KEYS   = [
    "FSD50K.dev_audio.z01",
    "FSD50K.dev_audio.z02",
    "FSD50K.dev_audio.z03",
    "FSD50K.dev_audio.z04",
    "FSD50K.dev_audio.z05",
    "FSD50K.dev_audio.zip",   # 분할 zip에서 마지막(헤더) 파트
]

# SELD_CLASSES를 scene_builder에서 가져옴 (200개 FSD50K AudioSet 클래스)
# FSD50K 레이블명 = SELD 클래스명 → identity 매핑
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from data.sim_generator.scene_builder import SELD_CLASSES

# 빠른 매핑용: 소문자 클래스명 → 원본 클래스명
_SELD_LOWER: dict[str, str] = {c.lower(): c for c in SELD_CLASSES}


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def _class_dir_name(cls: str) -> str:
    return cls.replace(" ", "_")


def _labels_to_seld(labels_str: str, mids_str: str) -> Optional[str]:
    """
    FSD50K 레이블 문자열에서 SELD 클래스 결정.

    FSD50K 레이블 = AudioSet 클래스명 = SELD 클래스명이므로 identity 매핑.
    labels_str은 쉼표 구분이지만 "Walk, footsteps" 같이 쉼표를 포함한 클래스명도 있으므로
    분할하지 않고 각 SELD 클래스명이 전체 문자열에 포함되는지 substring 검사.
    """
    labels_lower = labels_str.lower()
    for cls_lower, cls_orig in _SELD_LOWER.items():
        if cls_lower in labels_lower:
            return cls_orig
    return None


def _download_file(url: str, dst: Path, desc: str = "") -> None:
    """스트리밍 다운로드 with tqdm 진행바. 이미 완성된 파일은 건너뜀."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    # 이어받기: Content-Length로 완료 여부 확인
    if dst.exists():
        r_head = requests.head(url, timeout=30, allow_redirects=True)
        expected = int(r_head.headers.get("Content-Length", 0))
        if expected and dst.stat().st_size == expected:
            log.info(f"  이미 완료: {dst.name} ({expected / 1e9:.2f} GB) — 건너뜀")
            return

    r = requests.get(url, stream=True, timeout=60, allow_redirects=True)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))

    tmp = dst.with_suffix(dst.suffix + ".part")
    with open(tmp, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        unit_divisor=1024, desc=desc or dst.name,
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dst)


def _get_zenodo_urls() -> dict[str, str]:
    """Zenodo API로 {파일명: 다운로드 URL} 반환."""
    log.info(f"Zenodo record {ZENODO_RECORD_ID} 파일 목록 조회 중...")
    r = requests.get(ZENODO_API_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Zenodo 신 API: links.self 가 이미 /content (실제 다운로드) 엔드포인트
    return {f["key"]: f["links"]["self"] for f in data["files"]}


def _check_7z() -> Optional[str]:
    """7z 또는 7za 바이너리 경로 반환. 없으면 None."""
    for cmd in ["7z", "7za", "7zr"]:
        if shutil.which(cmd):
            return cmd
    return None


def _extract_split_zip(main_zip: Path, out_dir: Path, cmd_7z: str) -> None:
    """
    7z로 분할 zip 전체 추출.
    main_zip (.zip) 과 같은 디렉터리에 .z01~.z05 파트들이 있어야 합니다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"7z 추출 중: {main_zip.name} → {out_dir}")
    result = subprocess.run(
        [cmd_7z, "e", str(main_zip), f"-o{out_dir}", "-y"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"7z 추출 실패 (code {result.returncode}):\n{result.stderr}"
        )


def _extract_split_zip_fallback(zip_parts: list[Path], out_dir: Path) -> None:
    """
    7z 없을 때 fallback: 분할 zip 파트들을 순서대로 이어붙여 추출.
    순서: z01, z02, ..., z05, .zip  (마지막 .zip에 central directory 존재)
    임시 파일 combined.zip을 tmp_dir에 생성 후 삭제합니다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = zip_parts[0].parent

    # 파트 정렬: z01, z02, ..., z05, .zip
    def sort_key(p: Path):
        s = p.suffix  # .z01, .z02, ..., .z05, .zip
        if s == ".zip":
            return 99
        return int(s[2:])  # .z01 → 1

    sorted_parts = sorted(zip_parts, key=sort_key)
    combined = tmp_dir / "FSD50K_combined.zip"

    log.info(f"분할 zip 병합 중 ({len(sorted_parts)}개 파트)...")
    with open(combined, "wb") as out:
        for part in tqdm(sorted_parts, desc="병합"):
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out)

    log.info(f"zipfile 추출 중: {combined.name} → {out_dir}")
    with zipfile.ZipFile(combined) as zf:
        for name in tqdm(zf.namelist(), desc="추출"):
            zf.extract(name, out_dir)

    combined.unlink()


# ── 메인 로직 ─────────────────────────────────────────────────────────────────

def parse_ground_truth(gt_zip_path: Path) -> dict[str, str]:
    """
    FSD50K.ground_truth.zip 파싱 → {fname: seld_class} 매핑.
    ground_truth 디렉터리 내 dev.csv를 주로 사용합니다.
    """
    fname_to_seld: dict[str, str] = {}

    with zipfile.ZipFile(gt_zip_path) as zf:
        # dev.csv 또는 ground_truth_dev.csv 우선 탐색
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        dev_csvs = [n for n in csv_names if "dev" in n.lower()]
        targets  = dev_csvs if dev_csvs else csv_names

        for csv_name in targets:
            log.info(f"  파싱: {csv_name}")
            with zf.open(csv_name) as f:
                text   = io.TextIOWrapper(f, encoding="utf-8")
                reader = csv.DictReader(text)
                for row in reader:
                    fname  = str(row.get("fname", "")).strip()
                    labels = row.get("labels", "")
                    mids   = row.get("mids",   "")
                    if not fname:
                        continue
                    seld_cls = _labels_to_seld(labels, mids)
                    if seld_cls:
                        fname_to_seld[fname] = seld_cls

    # 클래스별 통계 출력
    log.info(f"레이블 매핑: {len(fname_to_seld)}개 파일 → SELD 클래스")
    for cls in SELD_CLASSES:
        cnt = sum(1 for v in fname_to_seld.values() if v == cls)
        log.info(f"  {cls:35s}: {cnt:5d}개")
    return fname_to_seld


def copy_matching_audio(
    extracted_dir: Path,
    fname_to_seld: dict[str, str],
    out_dir: Path,
    max_per_class: int,
) -> dict[str, int]:
    """
    추출된 오디오 파일에서 SELD 클래스 매핑된 파일만 out_dir로 복사.
    Returns: {class_name: count} 복사 통계
    """
    counters: dict[str, int] = {}
    audio_files = (
        list(extracted_dir.rglob("*.wav"))
        + list(extracted_dir.rglob("*.flac"))
    )
    log.info(f"추출된 오디오 파일: {len(audio_files)}개 → 클래스 필터링 중...")

    for audio_path in tqdm(audio_files, desc="분류"):
        stem = audio_path.stem  # e.g. "83347"
        seld_cls = fname_to_seld.get(stem)
        if seld_cls is None:
            continue
        if counters.get(seld_cls, 0) >= max_per_class:
            continue

        dst_dir  = out_dir / _class_dir_name(seld_cls)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / audio_path.name

        if not dst_file.exists():
            shutil.copy2(audio_path, dst_file)
        counters[seld_cls] = counters.get(seld_cls, 0) + 1

    return counters


def download_and_organize(out_dir: Path, tmp_dir: Path, max_per_class: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── 기존 파일 삭제 ────────────────────────────────────────────────────────
    existing = (
        list(out_dir.rglob("*.wav"))
        + list(out_dir.rglob("*.mp3"))
        + list(out_dir.rglob("*.flac"))
    )
    if existing:
        log.info(f"기존 sound_effects 파일 {len(existing)}개 삭제 중...")
        for f in existing:
            f.unlink()
        for d in sorted(out_dir.iterdir()):
            if d.is_dir():
                try:
                    d.rmdir()
                except OSError:
                    pass

    # ── 7z 확인 ───────────────────────────────────────────────────────────────
    cmd_7z = _check_7z()
    if cmd_7z:
        log.info(f"7z 발견: {cmd_7z}")
    else:
        log.warning(
            "7z를 찾을 수 없습니다. 분할 zip 병합 fallback을 사용합니다.\n"
            "  더 빠른 추출을 위해: conda install -c conda-forge p7zip"
        )

    # ── Zenodo 파일 목록 조회 ──────────────────────────────────────────────────
    try:
        file_urls = _get_zenodo_urls()
    except Exception as e:
        log.error(f"Zenodo API 조회 실패: {e}")
        return

    def get_url(key: str) -> str:
        """API 결과 또는 fallback URL 반환."""
        return file_urls.get(
            key,
            f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/{key}/content"
        )

    # ── Ground truth 다운로드 및 파싱 ─────────────────────────────────────────
    gt_zip = tmp_dir / GROUND_TRUTH_KEY
    log.info(f"\nGround truth 다운로드: {GROUND_TRUTH_KEY} (~334 KB)")
    try:
        _download_file(get_url(GROUND_TRUTH_KEY), gt_zip, GROUND_TRUTH_KEY)
    except Exception as e:
        log.error(f"Ground truth 다운로드 실패: {e}")
        return

    fname_to_seld = parse_ground_truth(gt_zip)
    if not fname_to_seld:
        log.error("레이블 파싱 실패 — CSV 구조 확인 필요")
        return

    # ── Dev 오디오 분할 zip 다운로드 ──────────────────────────────────────────
    log.info(f"\nDev 오디오 분할 zip 다운로드 ({len(DEV_AUDIO_KEYS)}개 파트, 총 ~17 GB)")
    zip_parts: list[Path] = []

    for key in DEV_AUDIO_KEYS:
        dst = tmp_dir / key
        size_gb = {
            "FSD50K.dev_audio.zip":  2.1,
            "FSD50K.dev_audio.z01":  3.0,
            "FSD50K.dev_audio.z02":  3.0,
            "FSD50K.dev_audio.z03":  3.0,
            "FSD50K.dev_audio.z04":  3.0,
            "FSD50K.dev_audio.z05":  3.0,
        }.get(key, 3.0)
        log.info(f"\n  다운로드: {key} (~{size_gb:.1f} GB)")
        try:
            _download_file(get_url(key), dst, key)
            zip_parts.append(dst)
        except Exception as e:
            log.error(f"  다운로드 실패 ({key}): {e}")
            return

    # ── 분할 zip 추출 ─────────────────────────────────────────────────────────
    audio_tmp = tmp_dir / "extracted_audio"
    main_zip  = tmp_dir / "FSD50K.dev_audio.zip"

    log.info(f"\n추출 시작 → {audio_tmp}")
    try:
        if cmd_7z:
            _extract_split_zip(main_zip, audio_tmp, cmd_7z)
        else:
            _extract_split_zip_fallback(zip_parts, audio_tmp)
    except Exception as e:
        log.error(f"추출 실패: {e}")
        return

    # ── 클래스별 복사 ─────────────────────────────────────────────────────────
    log.info(f"\n클래스별 파일 분류 및 복사 → {out_dir}")
    counters = copy_matching_audio(audio_tmp, fname_to_seld, out_dir, max_per_class)

    # ── 임시 파일 정리 ────────────────────────────────────────────────────────
    log.info(f"\n임시 파일 정리: {tmp_dir}")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── 결과 보고 ──────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("완료. 클래스별 파일 수:")
    total = 0
    for cls in SELD_CLASSES:
        cnt = counters.get(cls, 0)
        total += cnt
        status = "✓" if cnt > 0 else "✗ 파일 없음"
        log.info(f"  {cls:35s}: {cnt:5d}개  {status}")
    log.info(f"\n총 {total}개 파일 → {out_dir}")

    # 최종 디스크 사용량
    total_size = sum(
        f.stat().st_size for f in out_dir.rglob("*") if f.is_file()
    )
    log.info(f"디스크 사용량: {total_size / 1e9:.2f} GB")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FSD50K dev 세트 다운로드 및 SELD 클래스별 정리"
    )
    parser.add_argument(
        "--out_dir", default="data/sound_effects",
        help="출력 디렉터리 (default: data/sound_effects)",
    )
    parser.add_argument(
        "--tmp_dir", default="data/_fsd50k_tmp",
        help="임시 다운로드/추출 디렉터리 (완료 후 자동 삭제)",
    )
    parser.add_argument(
        "--max_per_class", type=int, default=500,
        help="클래스당 최대 파일 수 (default: 500)",
    )
    args = parser.parse_args()

    base    = Path(__file__).parent.parent
    out_dir = base / args.out_dir
    tmp_dir = base / args.tmp_dir

    log.info("=" * 60)
    log.info("FSD50K 다운로드 스크립트")
    log.info(f"  출력 디렉터리  : {out_dir}")
    log.info(f"  임시 디렉터리  : {tmp_dir}")
    log.info(f"  클래스당 최대  : {args.max_per_class}개")
    log.info(f"  다운로드 용량  : ~17 GB (dev_audio 분할 zip 6개)")
    log.info(f"  최종 보존 용량 : ~{args.max_per_class * 13 * 0.5 / 1024:.1f} GB (예상)")
    log.info("=" * 60)

    download_and_organize(out_dir, tmp_dir, args.max_per_class)


if __name__ == "__main__":
    main()
