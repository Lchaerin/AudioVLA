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

import numpy as np
import soundfile as sf
import torch

try:
    from .scene_builder import SceneBuilder, SELD_CLASSES
    from .binaural_renderer import MultisourceBinauralMixer, SOFAPool
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from data.sim_generator.scene_builder import SceneBuilder, SELD_CLASSES
    from data.sim_generator.binaural_renderer import MultisourceBinauralMixer, SOFAPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

_PROJECT_ROOT     = Path(__file__).parent.parent.parent
DEFAULT_SOUND_DIR = _PROJECT_ROOT / "data" / "sound_effects"
DEFAULT_HRIR_DIR  = _PROJECT_ROOT / "data" / "hrir"


def find_audio_clips(sound_dir: Path, num_classes: int) -> dict[int, list[Path]]:
    """
    sound_dir 하위 클래스 디렉터리에서 오디오 파일을 인덱싱합니다.
    파일이 없는 클래스는 빈 리스트로 유지 (씬 생성 시 해당 클래스 제외).
    """
    clips: dict[int, list[Path]] = {i: [] for i in range(num_classes)}
    if not sound_dir.exists():
        log.warning(f"sound_effects 디렉터리가 없습니다: {sound_dir}")
        return clips

    missing = []
    for cls_idx, cls_name in enumerate(SELD_CLASSES[:num_classes]):
        cls_dir = sound_dir / cls_name.replace(" ", "_")
        if cls_dir.exists():
            found = (
                list(cls_dir.glob("*.wav"))
                + list(cls_dir.glob("*.flac"))
                + list(cls_dir.glob("*.mp3"))
            )
            clips[cls_idx] = found
        else:
            missing.append(cls_name)

    available = sum(1 for v in clips.values() if len(v) > 0)
    total     = sum(len(v) for v in clips.values())
    log.info(f"오디오 클립 인덱싱 완료: {available}/{num_classes}개 클래스, 총 {total}개 파일")
    if missing:
        log.info(f"  오디오 없는 클래스 {len(missing)}개는 씬 생성에서 제외됩니다.")
    return clips


def load_audio_soundfile(path: Path, target_samples: int, sample_rate: int) -> torch.Tensor:
    """
    soundfile로 오디오 로드 (torchaudio/torchcodec 의존성 없음).
    Returns: (target_samples,) 모노 float32 텐서
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.mean(axis=1))  # → 모노 (T,)

    if sr != sample_rate:
        # torchaudio.functional.resample은 torchcodec 없이도 동작
        import torchaudio
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, sample_rate).squeeze(0)

    if wav.shape[0] < target_samples:
        repeats = (target_samples // wav.shape[0]) + 2
        wav = wav.repeat(repeats)

    max_start = max(0, wav.shape[0] - target_samples)
    start = random.randint(0, max_start)
    return wav[start: start + target_samples]


def load_or_generate_audio(
    clips: dict[int, list[Path]],
    cls_idx: int,
    target_samples: int,
    sample_rate: int,
) -> torch.Tensor:
    paths = clips.get(cls_idx, [])
    if not paths:
        return torch.randn(target_samples) * 0.05

    path = random.choice(paths)
    try:
        return load_audio_soundfile(path, target_samples, sample_rate)
    except Exception as e:
        log.debug(f"오디오 로드 실패 ({path.name}): {e}")
        return torch.randn(target_samples) * 0.05


def generate_episode(
    scene_builder: SceneBuilder,
    mixer: MultisourceBinauralMixer,
    clips: dict[int, list[Path]],
    available_class_indices: list[int],
    audio_sr: int,
    audio_duration: float,
    action_dim: int = 7,
    action_chunk: int = 50,
) -> dict:
    """단일 학습 에피소드 생성."""
    target_samples = int(audio_sr * audio_duration)
    scene = scene_builder.build(available_class_indices=available_class_indices)

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

    target_idx = next(
        i for i, o in enumerate(scene.objects)
        if o.object_id == scene.target_object_id
    )
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


def save_audio_soundfile(path: Path, tensor: torch.Tensor, sample_rate: int) -> None:
    """soundfile로 wav 저장 (torchaudio/torchcodec 의존성 없음)."""
    data = tensor.numpy().T  # (2, T) → (T, 2)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")


def collect_episodes(args):
    out_dir   = Path(args.output_dir)
    sound_dir = Path(args.sound_effects_dir)
    hrir_dir  = Path(args.hrir_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = find_audio_clips(sound_dir, num_classes=len(SELD_CLASSES))

    # 실제 오디오가 있는 클래스 인덱스만 추출
    available_class_indices = [i for i, v in clips.items() if len(v) > 0]
    if not available_class_indices:
        log.warning("오디오 파일이 없어 모든 클래스를 허용합니다 (화이트 노이즈 사용).")
        available_class_indices = list(range(len(SELD_CLASSES)))

    try:
        sofa_pool = SOFAPool(str(hrir_dir), target_sr=args.audio_sr)
        log.info(f"SOFAPool 초기화: {sofa_pool.n_subjects}개 SOFA 파일")
        renderer = sofa_pool
    except FileNotFoundError as e:
        log.warning(f"SOFA 초기화 실패: {e}\nITD/ILD 근사 렌더러를 사용합니다.")
        renderer = None

    builder = SceneBuilder(img_size=args.img_size)
    mixer   = MultisourceBinauralMixer(sample_rate=args.audio_sr, renderer=renderer)

    log.info(f"에피소드 생성 시작: {args.num_episodes}개 → {out_dir}")

    for ep_idx in range(args.num_episodes):
        ep_dir = out_dir / f"{ep_idx:05d}"
        ep_dir.mkdir(exist_ok=True)

        try:
            episode = generate_episode(
                builder, mixer, clips, available_class_indices,
                audio_sr=args.audio_sr,
                audio_duration=args.audio_duration,
            )
        except Exception as e:
            log.error(f"에피소드 {ep_idx} 생성 실패: {e}")
            continue

        save_audio_soundfile(ep_dir / "audio.wav", episode["binaural"], args.audio_sr)

        with open(ep_dir / "meta.json", "w") as f:
            json.dump(episode["meta"], f, indent=2)

        if not args.skip_image:
            try:
                from PIL import Image
                dummy_img = np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8)
                Image.fromarray(dummy_img).save(ep_dir / "image.png")
            except ImportError:
                pass

        if (ep_idx + 1) % 500 == 0 or (ep_idx + 1) == args.num_episodes:
            log.info(f"  {ep_idx + 1} / {args.num_episodes} 에피소드 완료")

    log.info(f"완료. 에피소드 저장 위치: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Audio-VLA 시뮬레이션 에피소드 생성")
    parser.add_argument("--output_dir",        default="data/sim_episodes")
    parser.add_argument("--num_episodes",       type=int,   default=5000)
    parser.add_argument("--sound_effects_dir",  default=str(DEFAULT_SOUND_DIR))
    parser.add_argument("--hrir_dir",           default=str(DEFAULT_HRIR_DIR))
    parser.add_argument("--audio_sr",           type=int,   default=24000)
    parser.add_argument("--audio_duration",     type=float, default=5.0)
    parser.add_argument("--img_size",           type=int,   default=512)
    parser.add_argument("--skip_image",         action="store_true",
                        help="image.png 생성 생략 (SELD 전용 학습 시 사용)")
    args = parser.parse_args()
    collect_episodes(args)


if __name__ == "__main__":
    main()
