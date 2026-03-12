"""
Phase 1 – Train the SELD model on STARSS23 / DCASE2025 Stereo SELD data.

Usage:
    python training/train_seld.py --config configs/default.yaml
"""

import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

from models.seld import ResNetConformerSELD
from data.dataset import SELDDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ACCDOA loss for SELD
# ---------------------------------------------------------------------------

class MultiACCDOALoss(nn.Module):
    """
    Activity-Coupled Cartesian Direction of Arrival loss.
    Target: (B, T, N_tracks, num_classes, 3) — activity-weighted xyz vectors
    """

    def forward(self, pred_xyz, target_xyz):
        return F.mse_loss(pred_xyz, target_xyz)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_seld(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config.training.output_dir) / "seld"
    out_dir.mkdir(parents=True, exist_ok=True)

    if _WANDB:
        wandb.init(project="audio-vla", name="seld-training", config=OmegaConf.to_container(config))

    # Model
    model = ResNetConformerSELD(
        num_classes=config.model.num_seld_classes,
        n_max_sources=config.model.N_max,
        sample_rate=config.data.audio_sr,
    ).to(device)

    # Dataset
    data_root = getattr(config.data, "seld_data_root", "data/sim_episodes")
    log.info(f"SELDDataset 로드: {data_root}")
    train_dataset = SELDDataset(
        data_root=data_root,
        audio_sr=config.data.audio_sr,
        num_classes=config.model.num_seld_classes,
        N_max=config.model.N_max,
        split="train",
        train_fraction=config.data.train_split,
    )
    val_dataset = SELDDataset(
        data_root=data_root,
        audio_sr=config.data.audio_sr,
        num_classes=config.model.num_seld_classes,
        N_max=config.model.N_max,
        split="val",
        train_fraction=config.data.train_split,
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            f"에피소드를 찾을 수 없습니다: {data_root}\n"
            "먼저 episode_collector.py를 실행해 에피소드를 생성하세요."
        )
    log.info(f"  train: {len(train_dataset)}개  val: {len(val_dataset)}개")
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    criterion  = MultiACCDOALoss()
    optimizer  = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = get_cosine_schedule(
        optimizer, config.training.warmup_steps, config.training.total_steps
    )

    global_step = 0
    model.train()

    while global_step < config.training.total_steps:
        for batch in loader:
            audio      = batch["audio"]
            target_xyz = batch["target_accdoa"]
            audio      = audio.to(device)
            target_xyz = target_xyz.to(device)  # (B, N, C, 3)

            seld_out = model(audio)

            # Build per-track prediction tensor
            # class_logits: (B, N, C) — use as activity proxy
            # peak_coords: (B, N, 2) — az, el
            # Reconstruct xyz from direction + activity
            az = seld_out.peak_coords[..., 0]
            el = seld_out.peak_coords[..., 1]
            cos_el = torch.cos(el)
            pred_dir = torch.stack([
                cos_el * torch.sin(az),
                -torch.sin(el),
                cos_el * torch.cos(az),
            ], dim=-1)  # (B, N, 3)

            activity = torch.sigmoid(seld_out.class_logits)  # (B, N, C)
            pred_xyz = pred_dir.unsqueeze(-2) * activity.unsqueeze(-1)  # (B, N, C, 3)

            loss = criterion(pred_xyz, target_xyz)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % config.training.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                log.info(f"[SELD] step={global_step} loss={loss.item():.4f} lr={lr:.2e}")
                if _WANDB:
                    wandb.log({"seld/loss": loss.item(), "seld/lr": lr}, step=global_step)

            if global_step % config.training.eval_every == 0 and len(val_dataset) > 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vbatch in val_loader:
                        vaudio = vbatch["audio"].to(device)
                        vtgt   = vbatch["target_accdoa"].to(device)
                        vout   = model(vaudio)
                        vaz, vel = vout.peak_coords[..., 0], vout.peak_coords[..., 1]
                        vcos_el  = torch.cos(vel)
                        vpred_dir = torch.stack([
                            vcos_el * torch.sin(vaz),
                            -torch.sin(vel),
                            vcos_el * torch.cos(vaz),
                        ], dim=-1)
                        vact     = torch.sigmoid(vout.class_logits)
                        vpred    = vpred_dir.unsqueeze(-2) * vact.unsqueeze(-1)
                        val_losses.append(criterion(vpred, vtgt).item())
                val_loss = sum(val_losses) / len(val_losses)
                log.info(f"[SELD] step={global_step} val_loss={val_loss:.4f}")
                if _WANDB:
                    wandb.log({"seld/val_loss": val_loss}, step=global_step)
                model.train()

            if global_step % config.training.save_every == 0:
                ckpt = out_dir / f"seld_step{global_step:06d}.pt"
                torch.save(model.state_dict(), ckpt)
                log.info(f"Saved SELD checkpoint: {ckpt}")

            if global_step >= config.training.total_steps:
                break

    final = out_dir / "seld_final.pt"
    torch.save(model.state_dict(), final)
    log.info(f"SELD training complete. Final checkpoint: {final}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    train_seld(config)


if __name__ == "__main__":
    main()
