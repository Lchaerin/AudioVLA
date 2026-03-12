"""
Action prediction evaluation.

Compares Audio-VLA action outputs against ground-truth actions or
against a baseline SmolVLA (no audio) to measure the benefit of audio fusion.

Metrics:
  - L2 error on predicted action chunk
  - Success rate (from simulator or human evaluation)
"""

import argparse
import logging

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import AudioVLADataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def evaluate_action(config, fusion_ckpt: str, data_root: str, split: str = "val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.audio_vla_pipeline import AudioVLAPipeline
    pipeline = AudioVLAPipeline(config).to(device)
    pipeline.freeze_pretrained()

    if fusion_ckpt:
        pipeline.load_fusion(fusion_ckpt)
    pipeline.eval()

    dataset = AudioVLADataset(
        data_root=data_root,
        audio_sr=config.data.audio_sr,
        img_size=config.data.img_size,
        split=split,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_l2 = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            audio         = batch["audio"][0]           # (2, T)
            image         = batch["image"][0]           # (3, H, W)
            command       = batch["command"][0]
            camera_K      = batch["camera_intrinsic"][0]
            target_actions= batch["target_actions"][0]  # (50, A)

            try:
                actions, _ = pipeline.predict(audio, image, command, camera_K)
                l2 = (actions.cpu() - target_actions).pow(2).mean().item()
                total_l2 += l2
                n += 1
            except NotImplementedError:
                log.warning("predict_action not implemented — skipping action eval.")
                break

    if n > 0:
        log.info(f"Mean action L2 error ({split}): {total_l2 / n:.6f}")
    return total_l2 / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--fusion_ckpt", default=None)
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--split",       default="val")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    evaluate_action(config, args.fusion_ckpt, args.data_root, args.split)


if __name__ == "__main__":
    main()
