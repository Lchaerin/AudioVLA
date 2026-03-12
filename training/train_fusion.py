"""
Phase 2 – Train the Audio-VLA Fusion Module.

SELD, SmolVLA, and CLAP are all frozen.
Only the fusion modules (~5-10M params) are trained.

Usage:
    python training/train_fusion.py --config configs/sim_training.yaml
"""

import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

from models.seld import ResNetConformerSELD, SoundTokenEncoder
from models.fusion import (
    AudioLanguageCrossAttention,
    AzElToPixel,
    AudioAttentionMapGenerator,
    AudioVisualFusion,
    CLAPProjection,
)
from training.losses import AudioVLALoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_fusion_modules(config, device):
    D_s   = config.model.D_s
    D_l   = config.model.D_l
    D_v   = config.model.D_v
    D_h   = config.model.D_hidden
    n_cls = config.model.num_seld_classes

    modules = nn.ModuleDict({
        "sound_token_enc":       SoundTokenEncoder(D_s=D_s, num_classes=n_cls),
        "audio_lang_cross_attn": AudioLanguageCrossAttention(D_s=D_s, D_l=D_l, D_hidden=D_h,
                                                              num_heads=config.model.num_heads),
        "azel_to_pixel":         AzElToPixel(),
        "attn_map_gen":          AudioAttentionMapGenerator(sigma=config.model.sigma_init),
        "av_fusion":             AudioVisualFusion(D_v=D_v, audio_context_dim=D_h),
        "clap_proj":             CLAPProjection(num_classes=n_cls, clap_dim=config.model.clap_dim),
    })
    return modules.to(device)


def train_fusion(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config.training.output_dir) / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    if _WANDB:
        wandb.init(project="audio-vla", name="fusion-training",
                   config=OmegaConf.to_container(config))

    # ── Frozen models ────────────────────────────────────────────────
    seld = ResNetConformerSELD(
        num_classes=config.model.num_seld_classes,
        n_max_sources=config.model.N_max,
        sample_rate=config.data.audio_sr,
    ).to(device)

    if config.checkpoints.seld_checkpoint:
        seld.load_state_dict(torch.load(config.checkpoints.seld_checkpoint, map_location=device))
    seld.eval()
    for p in seld.parameters():
        p.requires_grad = False

    # SmolVLA wrapper (lazy-loaded; language + vision encoding)
    try:
        from models.vla import SmolVLAWrapper
        smolvla = SmolVLAWrapper(
            pretrained=config.checkpoints.smolvla_checkpoint, device=str(device)
        )
        _smolvla_available = True
    except Exception as e:
        log.warning(f"SmolVLA not available: {e}. Using dummy lang/vision features.")
        _smolvla_available = False

    # ── Trainable fusion modules ─────────────────────────────────────
    fusion = build_fusion_modules(config, device)

    if config.checkpoints.fusion_checkpoint:
        state = torch.load(config.checkpoints.fusion_checkpoint, map_location=device)
        fusion.load_state_dict(state)
        log.info(f"Loaded fusion checkpoint: {config.checkpoints.fusion_checkpoint}")

    trainable_params = list(fusion.parameters())
    log.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = get_cosine_schedule(
        optimizer, config.training.warmup_steps, config.training.total_steps
    )
    criterion = AudioVLALoss(
        lambda_loc=config.loss.lambda_loc,
        lambda_action=config.loss.lambda_action,
    )

    # ── Dataset ──────────────────────────────────────────────────────
    # Replace with actual AudioVLADataset
    log.info("NOTE: Using dummy data. Replace with AudioVLADataset from data/dataset.py.")
    from data.dataset import DummyAudioVLADataset
    dataset = DummyAudioVLADataset(
        num_samples=config.data.sim_episodes,
        audio_sr=config.data.audio_sr,
        audio_duration=config.data.audio_duration,
        img_size=config.data.img_size,
        N_max=config.model.N_max,
        num_classes=config.model.num_seld_classes,
        D_l=config.model.D_l,
        D_v=config.model.D_v,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    # ── Training loop ────────────────────────────────────────────────
    global_step = 0
    fusion.train()
    best_loss = float("inf")

    while global_step < config.training.total_steps:
        for batch in loader:
            audio         = batch["audio"].to(device)           # (B, 2, T)
            image         = batch["image"].to(device)           # (B, 3, H, W)
            camera_K      = batch["camera_intrinsic"].to(device)# (B, 3, 3)
            lang_tokens   = batch["lang_tokens"].to(device)     # (B, T, D_l)
            visual_feats  = batch["visual_features"].to(device) # (B, 64, D_v)
            target_idx    = batch["target_sound_idx"].to(device)# (B,)
            target_actions= batch["target_actions"].to(device)  # (B, 50, A)

            # --- SELD forward (frozen) ---
            with torch.no_grad():
                seld_out = seld(audio)

            # --- Fusion forward ---
            sound_tokens = fusion["sound_token_enc"](
                seld_out.peak_coords, seld_out.class_logits, seld_out.energy
            )  # (B, N, D_s)

            attn_weights, audio_context = fusion["audio_lang_cross_attn"](
                sound_tokens, lang_tokens, seld_out.valid_mask
            )  # (B, N), (B, D_h)

            pixel_coords, in_frame = fusion["azel_to_pixel"](
                seld_out.peak_coords, camera_K,
                img_h=image.shape[-2], img_w=image.shape[-1]
            )

            H_feat = W_feat = 8
            audio_attn_map = fusion["attn_map_gen"](
                pixel_coords, attn_weights, in_frame, H_feat, W_feat
            )  # (B, H, W)

            fused_features = fusion["av_fusion"](
                visual_feats, audio_attn_map, audio_context
            )  # (B, 64, D_v)

            # Placeholder action prediction (replace with actual SmolVLA call)
            # For now use fused_features mean as a dummy action proxy
            pred_actions = fused_features.mean(dim=1).unsqueeze(1).expand(
                -1, 50, target_actions.shape[-1]
            )

            pred = {"attn_weights": attn_weights, "actions": pred_actions}
            tgt  = {"target_sound_idx": target_idx, "target_actions": target_actions}

            loss, metrics = criterion(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.training.gradient_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % config.training.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                log.info(
                    f"[Fusion] step={global_step:5d} "
                    f"total={metrics['total_loss']:.4f} "
                    f"grounding={metrics['grounding_loss']:.4f} "
                    f"action={metrics['action_loss']:.4f} "
                    f"lr={lr:.2e}"
                )
                if _WANDB:
                    wandb.log({f"fusion/{k}": v for k, v in metrics.items()}, step=global_step)
                    wandb.log({"fusion/lr": lr}, step=global_step)

            if global_step % config.training.save_every == 0:
                ckpt = out_dir / f"fusion_step{global_step:06d}.pt"
                torch.save(fusion.state_dict(), ckpt)
                if metrics["total_loss"] < best_loss:
                    best_loss = metrics["total_loss"]
                    torch.save(fusion.state_dict(), out_dir / "best_model.pt")
                log.info(f"Saved checkpoint: {ckpt}")

            if global_step >= config.training.total_steps:
                break

    final = out_dir / "fusion_final.pt"
    torch.save(fusion.state_dict(), final)
    log.info(f"Fusion training complete. Final checkpoint: {final}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sim_training.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    train_fusion(config)


if __name__ == "__main__":
    main()
