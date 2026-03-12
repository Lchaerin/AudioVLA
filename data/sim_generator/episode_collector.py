"""
Episode collector for Audio-VLA simulation data generation.

SceneBuilder + SOFABinauralRenderer(또는 fallback)를 조합하여
AudioVLADataset이 기대하는 형식으로 학습 에피소드를 생성합니다.

Usage:
    python data/sim_generator/episode_collector.py \
        --output_dir data/sim_episodes \
        --num_episodes 5000 \
        --sound_effects_dir data/sound_effects \
        --hrir_dir data/hrir
"""

import argparse
import json
import logging
import random
from pathlib import Path

import torch
import torchaudio

try:
    from .scene_builder import SceneBuilder, SELD_CLASSES
    from .binaural_renderer import MultisourceBinauralMixer, SOFAPool
except ImportError:
    # python data/sim_generator/episode_collector.py 직접 실행 시
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from data.sim_generator.scene_builder import SceneBuilder, SELD_CLASSES
    from data.sim_generator.binaural_renderer import MultisourceBinauralMixer, SOFAPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# 기본 경로 (프로젝트 루트 기준)
_PROJECT_ROOT       = Path(__file__).parent.parent.parent
DEFAULT_SOUND_DIR   = _PROJECT_ROOT / "data" / "sound_effects"
DEFAULT_HRIR_DIR    = _PROJECT_ROOT / "data" / "hrir"


def find_audio_clips(sound_dir: Path, num_classes: int) -> dict[int, list[Path]]:
    """
    sound_dir 하위 클래스 디렉터리에서 오디오 파일을 인덱싱합니다.

    디렉터리명은 SELD_CLASSES[i].replace(" ", "_") 규칙을 따릅니다.
    예) data/sound_effects/Telephone/, data/sound_effects/Walk,_footsteps/

    클래스 디렉터리가 없으면 해당 클래스는 화이트 노이즈 fallback을 사용합니다.
    """
    clips: dict[int, list[Path]] = {i: [] for i in range(num_classes)}
    if not sound_dir.exists():
        log.warning(
            f"sound_effects 디렉터리가 없습니다: {sound_dir}\n"
            f"  python scripts/download_sound_effects.py 를 먼저 실행하세요."
        )
        return clips

    for cls_idx, cls_name in enumerate(SELD_CLASSES[:num_classes]):
        cls_dir = sound_dir / cls_name.replace(" ", "_")
        if cls_dir.exists():
            found = (
                list(cls_dir.glob("*.wav"))
                + list(cls_dir.glob("*.flac"))
                + list(cls_dir.glob("*.mp3"))
            )
            clips[cls_idx] = found
            log.debug(f"  {cls_name}: {len(found)}개 파일")
        else:
            log.warning(f"  클래스 디렉터리 없음: {cls_dir} (화이트 노이즈 사용)")

    total = sum(len(v) for v in clips.values())
    log.info(f"오디오 클립 인덱싱 완료: 총 {total}개 ({sound_dir})")
    return clips


def load_or_generate_audio(
    clips: dict[int, list[Path]],
    cls_idx: int,
    target_samples: int,
    sample_rate: int,
) -> torch.Tensor:
    """
    해당 클래스의 오디오 클립을 랜덤 선택하여 로드합니다.
    클립이 없으면 화이트 노이즈를 반환합니다.

    target_samples보다 짧으면 랜덤 offset으로 반복 타일링합니다.
    """
    paths = clips.get(cls_idx, [])
    if paths:
        path = random.choice(paths)
        try:
            wav, sr = torchaudio.load(str(path))
        except Exception as e:
            log.warning(f"오디오 로드 실패 ({path}): {e}. 화이트 노이즈 사용.")
            return torch.randn(target_samples) * 0.05

        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        wav = wav.mean(0)  # → 모노

        # target_samples보다 짧으면 타일링
        if wav.shape[0] < target_samples:
            repeats = (target_samples // wav.shape[0]) + 2
            wav = wav.repeat(repeats)
        # 랜덤 offset으로 크롭
        max_start = max(0, wav.shape[0] - target_samples)
        start = random.randint(0, max_start)
        wav = wav[start : start + target_samples]
    else:
        wav = torch.randn(target_samples) * 0.05  # fallback

    return wav


def generate_episode(
    scene_builder: SceneBuilder,
    mixer: MultisourceBinauralMixer,
    clips: dict[int, list[Path]],
    audio_sr: int,
    audio_duration: float,
    action_dim: int = 7,
    action_chunk: int = 50,
) -> dict:
    """단일 학습 에피소드 생성."""
    target_samples = int(audio_sr * audio_duration)
    scene = scene_builder.build()

    # 멀티소스 바이노럴 오디오 생성 (SOFA HRTF 컨볼루션)
    sources = []
    for obj in scene.objects:
        mono = load_or_generate_audio(clips, obj.class_idx, target_samples, audio_sr)
        sources.append({
            "waveform": mono,
            "az":   obj.az_rad,
            "el":   obj.el_rad,
            "gain": random.uniform(0.5, 1.0),
        })

    binaural = mixer.mix(sources, target_samples)  # (2, T)

    # target 인덱스 (scene.objects 리스트 내 위치)
    target_idx = next(
        i for i, o in enumerate(scene.objects)
        if o.object_id == scene.target_object_id
    )

    # Dummy action (실제 환경에서는 scripted policy 또는 teleoperation 데이터로 교체)
    target_action = torch.zeros(action_chunk, action_dim)

    meta = {
        "command":          scene.language_command,
        "target_sound_idx": target_idx,
        "target_actions":   target_action.tolist(),
        "camera_intrinsic": scene.camera_intrinsic.tolist(),
        "objects": [
            {
                "class_idx":  o.class_idx,
                "class_name": o.class_name,
                "az_rad":     o.az_rad,
                "el_rad":     o.el_rad,
                "position_m": list(o.position_m),
            }
            for o in scene.objects
        ],
    }

    return {"binaural": binaural, "meta": meta}


def collect_episodes(args):
    out_dir    = Path(args.output_dir)
    sound_dir  = Path(args.sound_effects_dir)
    hrir_dir   = Path(args.hrir_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 오디오 클립 인덱싱 ──────────────────────────────────────────────────────
    clips = find_audio_clips(sound_dir, num_classes=len(SELD_CLASSES))
    n_available = sum(len(v) for v in clips.values())
    if n_available == 0:
        log.warning(
            "사용 가능한 오디오 클립이 없습니다. "
            "python scripts/download_sound_effects.py 를 먼저 실행하세요. "
            "화이트 노이즈로 계속 진행합니다."
        )

    # ── SOFAPool 초기화 ────────────────────────────────────────────────────────
    try:
        sofa_pool = SOFAPool(str(hrir_dir), target_sr=args.audio_sr)
        log.info(f"SOFAPool 초기화: {sofa_pool.n_subjects}개 SOFA 파일")
        renderer = sofa_pool
    except FileNotFoundError as e:
        log.warning(f"SOFA 초기화 실패: {e}\nITD/ILD 근사 렌더러를 사용합니다.")
        renderer = None  # MultisourceBinauralMixer에서 SimpleBinauralRenderer로 fallback

    builder = SceneBuilder(img_size=args.img_size)
    mixer   = MultisourceBinauralMixer(sample_rate=args.audio_sr, renderer=renderer)

    log.info(f"에피소드 생성 시작: {args.num_episodes}개 → {out_dir}")

    for ep_idx in range(args.num_episodes):
        ep_dir = out_dir / f"{ep_idx:05d}"
        ep_dir.mkdir(exist_ok=True)

        try:
            episode = generate_episode(
                builder, mixer, clips,
                audio_sr=args.audio_sr,
                audio_duration=args.audio_duration,
            )
        except Exception as e:
            log.error(f"에피소드 {ep_idx} 생성 실패: {e}")
            continue

        # 오디오 저장
        torchaudio.save(
            str(ep_dir / "audio.wav"),
            episode["binaural"],
            args.audio_sr,
        )

        # 메타데이터 저장
        with open(ep_dir / "meta.json", "w") as f:
            json.dump(episode["meta"], f, indent=2)

        # placeholder 이미지 (SELD 학습에는 불필요 — --skip_image 로 생략 가능)
        if not args.skip_image:
            try:
                from PIL import Image
                import numpy as np
                dummy_img = np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8)
                Image.fromarray(dummy_img).save(ep_dir / "image.png")
            except ImportError:
                pass

        if (ep_idx + 1) % 500 == 0 or (ep_idx + 1) == args.num_episodes:
            log.info(f"  {ep_idx + 1} / {args.num_episodes} 에피소드 완료")

    log.info(f"완료. 에피소드 저장 위치: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Audio-VLA 시뮬레이션 에피소드 생성")
    parser.add_argument("--output_dir",         default="data/sim_episodes",
                        help="에피소드 저장 디렉터리")
    parser.add_argument("--num_episodes",        type=int, default=5000,
                        help="생성할 에피소드 수")
    parser.add_argument("--sound_effects_dir",   default=str(DEFAULT_SOUND_DIR),
                        help="SELD 클래스별 오디오 파일 디렉터리 (FSD50K 정리된 위치)")
    parser.add_argument("--hrir_dir",            default=str(DEFAULT_HRIR_DIR),
                        help="SOFA HRTF 파일 디렉터리 (data/hrir/)")
    parser.add_argument("--audio_sr",            type=int,   default=24000,
                        help="출력 오디오 샘플레이트")
    parser.add_argument("--audio_duration",      type=float, default=5.0,
                        help="에피소드당 오디오 길이 (초)")
    parser.add_argument("--img_size",            type=int,   default=512,
                        help="이미지 해상도")
    parser.add_argument("--skip_image",          action="store_true",
                        help="image.png 생성 생략 (SELD 전용 학습 시 사용)")
    args = parser.parse_args()
    collect_episodes(args)


if __name__ == "__main__":
    main()
