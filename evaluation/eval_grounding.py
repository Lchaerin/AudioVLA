"""
Audio grounding evaluation.

Measures how accurately the Audio-Language Cross-Attention selects
the correct sound source (the one referred to by the language command).

Metric: Top-1 grounding accuracy (attn_weights.argmax() == target_sound_idx)
"""

import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import AudioVLADataset
from models.seld import ResNetConformerSELD, SoundTokenEncoder
from models.fusion import AudioLanguageCrossAttention

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def evaluate_grounding(config, fusion_ckpt: str, data_root: str, split: str = "val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models
    seld = ResNetConformerSELD(num_classes=config.model.num_seld_classes).to(device)
    if config.checkpoints.seld_checkpoint:
        seld.load_state_dict(torch.load(config.checkpoints.seld_checkpoint, map_location=device))
    seld.eval()

    sound_token_enc = SoundTokenEncoder(
        D_s=config.model.D_s, num_classes=config.model.num_seld_classes
    ).to(device)
    cross_attn = AudioLanguageCrossAttention(
        D_s=config.model.D_s, D_l=config.model.D_l, D_hidden=config.model.D_hidden
    ).to(device)

    if fusion_ckpt:
        state = torch.load(fusion_ckpt, map_location=device)
        sound_token_enc.load_state_dict(state["sound_token_enc"])
        cross_attn.load_state_dict(state["audio_lang_cross_attn"])

    sound_token_enc.eval()
    cross_attn.eval()

    dataset = AudioVLADataset(
        data_root=data_root,
        audio_sr=config.data.audio_sr,
        img_size=config.data.img_size,
        D_l=config.model.D_l,
        D_v=config.model.D_v,
        split=split,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in loader:
            audio       = batch["audio"].to(device)
            lang_tokens = batch["lang_tokens"].to(device)
            target_idx  = batch["target_sound_idx"].to(device)

            seld_out = seld(audio)
            sound_tokens = sound_token_enc(
                seld_out.peak_coords, seld_out.class_logits, seld_out.energy
            )
            attn_weights, _ = cross_attn(sound_tokens, lang_tokens, seld_out.valid_mask)

            pred_idx = attn_weights.argmax(dim=-1)  # (B,)
            correct += (pred_idx == target_idx).sum().item()
            total   += target_idx.size(0)

    accuracy = correct / max(1, total)
    log.info(f"Grounding accuracy ({split}): {accuracy:.4f} ({correct}/{total})")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--fusion_ckpt",  default=None)
    parser.add_argument("--data_root",    required=True)
    parser.add_argument("--split",        default="val")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    evaluate_grounding(config, args.fusion_ckpt, args.data_root, args.split)


if __name__ == "__main__":
    main()
