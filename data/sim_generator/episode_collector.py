"""
Episode collector for Audio-VLA simulation data generation.

Combines SceneBuilder + BinauralRenderer to generate complete training episodes
and saves them to disk in the format expected by AudioVLADataset.

Usage:
    python data/sim_generator/episode_collector.py \
        --output_dir data/sim_episodes \
        --num_episodes 5000 \
        --audio_dir /path/to/audioset_clips
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path

import torch
import torchaudio

from .scene_builder import SceneBuilder, SELD_CLASSES
from .binaural_renderer import MultisourceBinauralMixer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def find_audio_clips(audio_dir: Path, num_classes: int) -> dict[int, list[Path]]:
    """Index audio files by class index. Assumes files are in subdirs named by class."""
    clips: dict[int, list[Path]] = {i: [] for i in range(num_classes)}
    if not audio_dir.exists():
        log.warning(f"Audio dir {audio_dir} not found. Using synthetic noise clips.")
        return clips
    for cls_idx, cls_name in enumerate(SELD_CLASSES[:num_classes]):
        cls_dir = audio_dir / cls_name.replace(" ", "_")
        if cls_dir.exists():
            clips[cls_idx] = list(cls_dir.glob("*.wav"))
    return clips


def load_or_generate_audio(
    clips: dict[int, list[Path]],
    cls_idx: int,
    target_samples: int,
    sample_rate: int,
) -> torch.Tensor:
    """Return a mono audio clip for the given class, or white noise if unavailable."""
    paths = clips.get(cls_idx, [])
    if paths:
        path = random.choice(paths)
        wav, sr = torchaudio.load(str(path))
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        wav = wav.mean(0)  # mono
    else:
        wav = torch.randn(target_samples) * 0.1  # fallback: white noise
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
    """Generate a single training episode."""
    target_samples = int(audio_sr * audio_duration)
    scene = scene_builder.build()

    # Build multi-source binaural audio
    sources = []
    for obj in scene.objects:
        mono = load_or_generate_audio(clips, obj.class_idx, target_samples, audio_sr)
        sources.append({
            "waveform": mono,
            "az": obj.az_rad,
            "el": obj.el_rad,
            "gain": random.uniform(0.5, 1.0),
        })

    binaural = mixer.mix(sources, target_samples)  # (2, T)

    # Dummy action (replace with scripted policy / teleop)
    target_action = torch.zeros(action_chunk, action_dim)

    # Metadata
    target_idx = next(
        i for i, o in enumerate(scene.objects)
        if o.object_id == scene.target_object_id
    )

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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = Path(args.audio_dir) if args.audio_dir else Path("nonexistent")
    clips = find_audio_clips(audio_dir, num_classes=len(SELD_CLASSES))

    builder = SceneBuilder(img_size=args.img_size)
    mixer   = MultisourceBinauralMixer(sample_rate=args.audio_sr)

    for ep_idx in range(args.num_episodes):
        ep_dir = out_dir / f"{ep_idx:05d}"
        ep_dir.mkdir(exist_ok=True)

        episode = generate_episode(
            builder, mixer, clips,
            audio_sr=args.audio_sr,
            audio_duration=args.audio_duration,
        )

        # Save audio
        torchaudio.save(str(ep_dir / "audio.wav"), episode["binaural"], args.audio_sr)

        # Save metadata
        with open(ep_dir / "meta.json", "w") as f:
            json.dump(episode["meta"], f, indent=2)

        # Placeholder image (replace with simulator-rendered image)
        try:
            from PIL import Image
            import numpy as np
            dummy_img = np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8)
            Image.fromarray(dummy_img).save(ep_dir / "image.png")
        except ImportError:
            pass

        if (ep_idx + 1) % 100 == 0:
            log.info(f"Generated {ep_idx + 1} / {args.num_episodes} episodes")

    log.info(f"Done. Episodes saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate Audio-VLA simulation episodes")
    parser.add_argument("--output_dir",    default="data/sim_episodes")
    parser.add_argument("--num_episodes",  type=int, default=5000)
    parser.add_argument("--audio_dir",     default=None, help="Root dir of AudioSet clips")
    parser.add_argument("--audio_sr",      type=int,   default=24000)
    parser.add_argument("--audio_duration",type=float, default=5.0)
    parser.add_argument("--img_size",      type=int,   default=512)
    args = parser.parse_args()
    collect_episodes(args)


if __name__ == "__main__":
    main()
