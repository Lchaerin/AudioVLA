"""
Phase 1 – Train CLAPProjection MLP.

Maps SELD class logits (one-hot-like) to LAION CLAP text embedding space
using precomputed CLAP text embeddings as supervision targets.

Usage:
    python training/train_clap_proj.py --config configs/default.yaml
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

from models.fusion import CLAPProjection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# STARSS23 / DCASE 2025 class names
SELD_CLASSES = [
    "Female speech",
    "Male speech",
    "Clapping",
    "Telephone",
    "Laughter",
    "Domestic sounds",
    "Walk, footsteps",
    "Door, open or close",
    "Music",
    "Musical instrument",
    "Water tap, faucet",
    "Bell",
    "Knock",
]


def precompute_clap_embeddings(class_names: list[str], clap_ckpt: str, device: str):
    """Precompute CLAP text embeddings for all class names."""
    try:
        import laion_clap
    except ImportError:
        raise ImportError("pip install laion-clap")

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(clap_ckpt)
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        embeds = model.get_text_embeddings(class_names)  # (C, clap_dim)

    return embeds  # (C, 512)


def train_clap_proj(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config.training.output_dir) / "clap_proj"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = config.model.num_seld_classes
    clap_dim    = config.model.clap_dim

    if _WANDB:
        wandb.init(project="audio-vla", name="clap-proj-training")

    # Precompute target CLAP embeddings
    log.info("Precomputing CLAP text embeddings …")
    clap_ckpt = config.checkpoints.clap_checkpoint
    if clap_ckpt and Path(clap_ckpt).exists():
        clap_embeds = precompute_clap_embeddings(
            SELD_CLASSES[:num_classes], clap_ckpt, str(device)
        )  # (C, 512)
    else:
        log.warning("CLAP checkpoint not found. Using random targets for demonstration.")
        clap_embeds = torch.randn(num_classes, clap_dim, device=device)
        clap_embeds = F.normalize(clap_embeds, dim=-1)

    # Dataset: one-hot class vectors → corresponding CLAP embedding
    one_hots = torch.eye(num_classes, device=device)  # (C, C)
    # Repeat to create a larger dataset
    n_repeat = 1000
    x = one_hots.repeat(n_repeat, 1)  # (C*n, C)
    y = clap_embeds.repeat(n_repeat, 1)  # (C*n, 512)

    # Add small noise to one-hots to improve generalisation
    x = x + 0.01 * torch.randn_like(x)

    dataset = TensorDataset(x.cpu(), y.cpu())
    loader  = DataLoader(dataset, batch_size=256, shuffle=True)

    model = CLAPProjection(num_classes=num_classes, clap_dim=clap_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    num_epochs = 50
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)  # (B, clap_dim)
            loss = F.mse_loss(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        log.info(f"Epoch {epoch+1:3d}/{num_epochs}  loss={avg:.6f}")
        if _WANDB:
            wandb.log({"clap_proj/loss": avg, "clap_proj/epoch": epoch})

    ckpt = out_dir / "clap_proj.pt"
    torch.save(model.state_dict(), ckpt)
    log.info(f"CLAPProjection saved to {ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    train_clap_proj(config)


if __name__ == "__main__":
    main()
