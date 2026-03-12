"""
Audio-VLA inference demo.

Usage:
    python scripts/run_inference.py \
        --config configs/default.yaml \
        --audio path/to/binaural.wav \
        --image path/to/image.png \
        --command "Pick up the object making the siren sound." \
        --fusion_ckpt outputs/sim_training/fusion/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
from omegaconf import OmegaConf

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.audio_vla_pipeline import AudioVLAPipeline
from evaluation.visualize import visualize_attention_map


def load_image(path: str, img_size: int = 512) -> torch.Tensor:
    from PIL import Image
    from torchvision import transforms as T
    img = Image.open(path).convert("RGB")
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    return transform(img)


def main():
    parser = argparse.ArgumentParser(description="Audio-VLA inference demo")
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--audio",       required=True,  help="Path to binaural .wav file")
    parser.add_argument("--image",       required=True,  help="Path to RGB image")
    parser.add_argument("--command",     required=True,  help="Language command string")
    parser.add_argument("--fusion_ckpt", default=None,   help="Path to fusion checkpoint")
    parser.add_argument("--visualize",   action="store_true", help="Show attention map")
    parser.add_argument("--save_vis",    default=None,   help="Save visualisation to file")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

    print(f"Loading Audio-VLA pipeline on {device} ...")
    pipeline = AudioVLAPipeline(config)
    pipeline.freeze_pretrained()
    if args.fusion_ckpt:
        pipeline.load_fusion(args.fusion_ckpt)
        print(f"Loaded fusion checkpoint: {args.fusion_ckpt}")
    pipeline.eval()
    pipeline = pipeline.to(device)

    # Load inputs
    waveform, sr = torchaudio.load(args.audio)
    if sr != config.data.audio_sr:
        waveform = torchaudio.functional.resample(waveform, sr, config.data.audio_sr)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)  # mono → pseudo-binaural

    image = load_image(args.image, img_size=config.data.img_size)

    # Default camera intrinsic
    f = 512.0
    c = config.data.img_size / 2
    K = torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1]], dtype=torch.float32)

    print(f"Command: {args.command}")
    print("Running inference ...")

    actions, debug = pipeline.predict(waveform, image, args.command, K)

    print(f"\nPredicted action chunk shape: {actions.shape}")
    print(f"First action step: {actions[0].tolist()}")
    print(f"\nDetected {debug['valid_mask' if 'in_frame_mask' not in debug else 'in_frame_mask'].sum().item()} in-frame sound sources")

    if args.visualize or args.save_vis:
        from data.sim_generator.scene_builder import SELD_CLASSES
        pred_classes = debug["class_logits"].argmax(dim=-1)
        class_labels = [SELD_CLASSES[c] for c in pred_classes.tolist()]
        visualize_attention_map(
            image=image,
            audio_attn_map=debug["audio_attn_map"],
            pixel_coords=debug["pixel_coords"],
            attn_weights=debug["attn_weights"],
            class_names=class_labels,
            save_path=args.save_vis,
            title=f'"{args.command}"',
        )


if __name__ == "__main__":
    main()
