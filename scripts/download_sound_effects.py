#!/usr/bin/env python3
"""
FSD50K (Freesound Dataset 50K) 다운로드 및 SELD 클래스별 정리 스크립트.

Zenodo에서 FSD50K dev 세트를 다운로드하고, 13개 SELD 클래스에 해당하는
오디오 파일을 data/sound_effects/{class_name}/ 형식으로 정리합니다.

총 다운로드 용량: 약 19 GB (dev audio 5개 zip)
최종 저장 용량: 약 3~5 GB (클래스 매핑된 파일만 보존, zip 삭제)

Usage:
    python scripts/download_sound_effects.py
    python scripts/download_sound_effects.py --out_dir data/sound_effects
                                              --tmp_dir /tmp/fsd50k
                                              --max_per_class 500
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import shutil
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

# AudioSet 레이블(및 머신 ID) → SELD 클래스 매핑
# FSD50K는 AudioSet ontology를 사용하므로 레이블 문자열 + MID 양쪽으로 매핑
AUDIOSET_LABEL_TO_SELD: dict[str, str] = {
    # Female speech
    "Female speech, woman speaking": "Female speech",
    "female speech":                 "Female speech",
    # Male speech
    "Male speech, man speaking": "Male speech",
    "male speech":               "Male speech",
    # Clapping
    "Clapping":   "Clapping",
    "Hand claps": "Clapping",
    # Telephone
    "Telephone":               "Telephone",
    "Telephone bell ringing":  "Telephone",
    "Ringtone":                "Telephone",
    "Mobile phone":            "Telephone",
    # Laughter
    "Laughter": "Laughter",
    "Giggle":   "Laughter",
    "Chuckle":  "Laughter",
    # Domestic sounds
    "Domestic sounds, home sounds": "Domestic sounds",
    "Vacuum cleaner":               "Domestic sounds",
    "Microwave oven":               "Domestic sounds",
    "Dishwasher":                   "Domestic sounds",
    "Blender":                      "Domestic sounds",
    "Mechanical fan":               "Domestic sounds",
    # Walk / footsteps
    "Walk, footsteps": "Walk, footsteps",
    "Footsteps":       "Walk, footsteps",
    "Run":             "Walk, footsteps",
    # Door
    "Door":             "Door, open or close",
    "Door, open or close": "Door, open or close",
    "Doorbell":         "Door, open or close",
    "Squeak":           "Door, open or close",
    "Creak":            "Door, open or close",
    # Music
    "Music":          "Music",
    "Song":           "Music",
    "Pop music":      "Music",
    "Rock music":     "Music",
    "Electronic music": "Music",
    "Jazz":           "Music",
    "Classical music": "Music",
    # Musical instrument
    "Musical instrument": "Musical instrument",
    "Guitar":             "Musical instrument",
    "Piano":              "Musical instrument",
    "Violin, fiddle":     "Musical instrument",
    "Drum":               "Musical instrument",
    "Trumpet":            "Musical instrument",
    "Saxophone":          "Musical instrument",
    "Flute":              "Musical instrument",
    "Synthesizer":        "Musical instrument",
    # Water tap
    "Water tap, faucet":          "Water tap, faucet",
    "Sink (filling or washing)":  "Water tap, faucet",
    "Running water":              "Water tap, faucet",
    "Water":                      "Water tap, faucet",
    # Bell
    "Bell":         "Bell",
    "Church bell":  "Bell",
    "Bicycle bell": "Bell",
    "Cowbell":      "Bell",
    "Jingle bell":  "Bell",
    "Wind chime":   "Bell",
    # Knock
    "Knock":             "Knock",
    "Knocking on door":  "Knock",
    "Tap":               "Knock",
    "Thump, thud":       "Knock",
}

# AudioSet MID → SELD 클래스 (레이블 파싱 실패 시 fallback)
AUDIOSET_MID_TO_SELD: dict[str, str] = {
    "/m/02zsn":   "Female speech",
    "/m/05zppz":  "Male speech",
    "/m/0l15bq":  "Clapping",
    "/m/07cx4":   "Telephone",
    "/m/07pp_mv": "Telephone",
    "/m/01j3sz":  "Laughter",
    "/m/0dv3j":   "Domestic sounds",
    "/m/01d3sl":  "Domestic sounds",
    "/m/07pbtc8": "Walk, footsteps",
    "/m/0l7xg":   "Walk, footsteps",
    "/m/02y_763": "Door, open or close",
    "/m/04rlf":   "Music",
    "/m/04szw":   "Musical instrument",
    "/m/085jw":   "Musical instrument",
    "/m/0dxrf":   "Water tap, faucet",
    "/m/07n_g":   "Bell",
    "/m/07rv9rh": "Bell",
    "/m/0642b4":  "Knock",
}

SELD_CLASSES = list(dict.fromkeys(AUDIOSET_LABEL_TO_SELD.values()))  # 순서 유지 unique


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def _class_dir_name(cls: str) -> str:
    """SELD 클래스명 → 디렉터리명 (episode_collector와 동일한 규칙)."""
    return cls.replace(" ", "_")


def _labels_to_seld(labels_str: str, mids_str: str) -> Optional[str]:
    """
    FSD50K 레이블 문자열과 MID 문자열에서 SELD 클래스를 매핑.
    labels_str: 쉼표 구분, 단일 레이블 내 쉼표 포함 가능 (e.g. "Walk, footsteps")
    mids_str:   쉼표 구분 AudioSet MID
    """
    # MID 기반 매핑 시도 (더 신뢰성 높음)
    for mid in mids_str.split(","):
        mid = mid.strip()
        if mid in AUDIOSET_MID_TO_SELD:
            return AUDIOSET_MID_TO_SELD[mid]

    # 레이블 문자열 기반 매핑 — 순서대로 substring 매칭
    labels_lower = labels_str.lower()
    for audioset_label, seld_cls in AUDIOSET_LABEL_TO_SELD.items():
        if audioset_label.lower() in labels_lower:
            return seld_cls

    return None


def _download_file(url: str, dst: Path, desc: str = "") -> None:
    """스트리밍 다운로드 with tqdm 진행바."""
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        unit_divisor=1024, desc=desc or dst.name, leave=False
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            f.write(chunk)
            bar.update(len(chunk))


def _get_zenodo_files() -> dict[str, str]:
    """Zenodo API로 파일명 → 다운로드 URL 딕셔너리 반환."""
    log.info(f"Zenodo 레코드 {ZENODO_RECORD_ID} 파일 목록 조회 중...")
    r = requests.get(ZENODO_API_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {f["key"]: f["links"]["self"] for f in data["files"]}


# ── 메인 로직 ─────────────────────────────────────────────────────────────────

def parse_metadata(meta_zip_path: Path) -> dict[str, str]:
    """
    FSD50K dev 메타데이터 zip을 파싱하여 {fname: seld_class} 매핑을 반환.
    fname은 파일 ID (숫자 문자열), .wav 확장자 없음.
    """
    fname_to_seld: dict[str, str] = {}

    with zipfile.ZipFile(meta_zip_path) as zf:
        # ground_truth 또는 collection CSV 탐색
        csv_candidates = [
            n for n in zf.namelist()
            if n.endswith(".csv") and ("ground_truth" in n or "collection" in n)
        ]
        if not csv_candidates:
            csv_candidates = [n for n in zf.namelist() if n.endswith(".csv")]

        for csv_name in csv_candidates:
            log.info(f"  메타데이터 파일 파싱: {csv_name}")
            with zf.open(csv_name) as f:
                text = io.TextIOWrapper(f, encoding="utf-8")
                reader = csv.DictReader(text)
                for row in reader:
                    fname   = str(row.get("fname", "")).strip()
                    labels  = row.get("labels", "")
                    mids    = row.get("mids", "")
                    if not fname:
                        continue
                    seld_cls = _labels_to_seld(labels, mids)
                    if seld_cls:
                        fname_to_seld[fname] = seld_cls

    log.info(f"메타데이터 매핑 완료: {len(fname_to_seld)}개 파일 → SELD 클래스")
    for cls in SELD_CLASSES:
        cnt = sum(1 for v in fname_to_seld.values() if v == cls)
        log.info(f"  {cls:35s}: {cnt:5d}개")
    return fname_to_seld


def extract_audio_zip(
    zip_path: Path,
    fname_to_seld: dict[str, str],
    out_dir: Path,
    counters: dict[str, int],
    max_per_class: int,
) -> int:
    """
    오디오 zip에서 SELD 클래스에 해당하는 파일만 추출하여 out_dir에 저장.
    Returns: 추출된 파일 수
    """
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in tqdm(names, desc=f"  {zip_path.name}", leave=False):
            stem = Path(name).stem  # e.g. "83347"
            if stem not in fname_to_seld:
                continue
            seld_cls = fname_to_seld[stem]
            if counters.get(seld_cls, 0) >= max_per_class:
                continue

            dst_dir = out_dir / _class_dir_name(seld_cls)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_file = dst_dir / Path(name).name

            if dst_file.exists():
                counters[seld_cls] = counters.get(seld_cls, 0) + 1
                continue

            with zf.open(name) as src, open(dst_file, "wb") as dst:
                shutil.copyfileobj(src, dst)

            counters[seld_cls] = counters.get(seld_cls, 0) + 1
            extracted += 1

    return extracted


def download_and_organize(
    out_dir: Path,
    tmp_dir: Path,
    max_per_class: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── 기존 파일 삭제 ────────────────────────────────────────────────────────
    existing = list(out_dir.rglob("*.wav")) + list(out_dir.rglob("*.mp3")) + list(out_dir.rglob("*.flac"))
    if existing:
        log.info(f"기존 sound_effects 파일 {len(existing)}개 삭제 중...")
        for f in existing:
            f.unlink()
        # 빈 디렉터리 정리
        for d in sorted(out_dir.iterdir()):
            if d.is_dir():
                try:
                    d.rmdir()
                except OSError:
                    pass

    # ── Zenodo 파일 목록 조회 ──────────────────────────────────────────────────
    try:
        file_urls = _get_zenodo_files()
    except Exception as e:
        log.error(f"Zenodo API 조회 실패: {e}")
        log.error("네트워크 연결 확인 후 재시도하세요.")
        return

    # ── 메타데이터 다운로드 ────────────────────────────────────────────────────
    meta_key  = "FSD50K.dev.meta.zip"
    meta_url  = file_urls.get(meta_key)
    if not meta_url:
        # fallback URL
        meta_url = f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files/{meta_key}"

    meta_zip = tmp_dir / meta_key
    if not meta_zip.exists():
        log.info(f"메타데이터 다운로드: {meta_key}")
        _download_file(meta_url, meta_zip, "메타데이터")
    else:
        log.info(f"메타데이터 캐시 사용: {meta_zip}")

    fname_to_seld = parse_metadata(meta_zip)
    if not fname_to_seld:
        log.error("메타데이터 파싱 실패 — CSV 형식이 예상과 다를 수 있습니다.")
        return

    # ── 오디오 zip 다운로드 및 추출 ───────────────────────────────────────────
    audio_keys = sorted(k for k in file_urls if k.startswith("FSD50K.dev.audio"))
    if not audio_keys:
        # fallback: 알려진 파일명
        audio_keys = [f"FSD50K.dev.audio.{i}.zip" for i in range(1, 6)]

    log.info(f"\n오디오 zip {len(audio_keys)}개 처리 예정:")
    for k in audio_keys:
        size_str = ""
        # Zenodo 메타데이터에서 크기 가져오기 (가능하면)
        log.info(f"  {k}")

    counters: dict[str, int] = {}
    total_extracted = 0

    for key in audio_keys:
        # 클래스별 한도 도달 여부 확인
        all_full = all(counters.get(c, 0) >= max_per_class for c in SELD_CLASSES)
        if all_full:
            log.info("모든 클래스 한도 도달 — 나머지 zip 건너뜀")
            break

        url = file_urls.get(key, f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files/{key}")
        zip_path = tmp_dir / key

        if not zip_path.exists():
            log.info(f"\n다운로드: {key}")
            try:
                _download_file(url, zip_path, key)
            except Exception as e:
                log.error(f"다운로드 실패 ({key}): {e}. 건너뜁니다.")
                continue
        else:
            log.info(f"캐시 사용: {zip_path}")

        log.info(f"추출 중: {key}")
        n = extract_audio_zip(zip_path, fname_to_seld, out_dir, counters, max_per_class)
        total_extracted += n
        log.info(f"  {n}개 파일 추출 (누적 {total_extracted}개)")

        # zip 삭제하여 디스크 절약
        zip_path.unlink()
        log.info(f"  {key} 삭제 완료")

    # ── 결과 보고 ──────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("다운로드 완료. 클래스별 파일 수:")
    total = 0
    for cls in SELD_CLASSES:
        cnt = counters.get(cls, 0)
        total += cnt
        status = "✓" if cnt > 0 else "✗ (파일 없음)"
        log.info(f"  {cls:35s}: {cnt:5d}개  {status}")
    log.info(f"\n총 {total}개 파일 → {out_dir}")

    # tmp 디렉터리 정리 (메타 zip 포함)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        log.info(f"임시 디렉터리 삭제: {tmp_dir}")


# ── 진입점 ─────────────────────────────────────────────────────────────────────

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
        help="임시 다운로드 디렉터리 (완료 후 자동 삭제)",
    )
    parser.add_argument(
        "--max_per_class", type=int, default=500,
        help="클래스당 최대 파일 수 (default: 500). 더 줄이면 용량 감소.",
    )
    args = parser.parse_args()

    base = Path(__file__).parent.parent  # 프로젝트 루트
    out_dir = base / args.out_dir
    tmp_dir = base / args.tmp_dir

    log.info("=" * 60)
    log.info("FSD50K 다운로드 스크립트")
    log.info(f"  출력 디렉터리  : {out_dir}")
    log.info(f"  임시 디렉터리  : {tmp_dir}")
    log.info(f"  클래스당 최대  : {args.max_per_class}개")
    log.info(f"  예상 다운로드  : ~19 GB (dev audio 1~5.zip)")
    log.info(f"  예상 최종 용량 : ~{args.max_per_class * 13 * 0.5 / 1024:.1f} GB")
    log.info("=" * 60)

    download_and_organize(out_dir, tmp_dir, args.max_per_class)


if __name__ == "__main__":
    main()
