"""
Attention map and grounding visualisation utilities.

Creates overlay images showing:
  - Audio source locations (peak_coords projected to pixels)
  - Attention weight per source (circle size / color)
  - Audio attention map overlaid on the original image
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    _MPL = True
except ImportError:
    _MPL = False


def visualize_attention_map(
    image: torch.Tensor,
    audio_attn_map: torch.Tensor,
    pixel_coords: torch.Tensor,
    attn_weights: torch.Tensor,
    class_names: list[str] | None = None,
    save_path: str | None = None,
    title: str = "",
):
    """
    Overlay the audio attention map on the image and draw source markers.

    Args:
        image:          (3, H, W) or (H, W, 3) — normalised RGB
        audio_attn_map: (H_grid, W_grid) — attention map at visual token resolution
        pixel_coords:   (N, 2) — [u, v] pixel coordinates of sound sources
        attn_weights:   (N,) — per-source attention weights
        class_names:    optional list of N class label strings
        save_path:      if given, save figure to this path
        title:          figure title
    """
    if not _MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")

    # Prepare image as numpy HWC
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy()
        if img_np.shape[0] == 3:
            img_np = img_np.transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = np.array(image)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

    H, W = img_np.shape[:2]

    # Upscale attention map to image size
    attn_np = audio_attn_map.cpu().float().numpy()
    from PIL import Image as PILImage
    attn_pil = PILImage.fromarray((attn_np * 255).astype(np.uint8))
    attn_resized = np.array(attn_pil.resize((W, H), PILImage.BILINEAR)) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=12)

    # Left: image + source markers
    ax = axes[0]
    ax.imshow(img_np)
    ax.set_title("Sound source locations")
    ax.axis("off")

    if isinstance(pixel_coords, torch.Tensor):
        coords_np = pixel_coords.cpu().numpy()
    else:
        coords_np = np.array(pixel_coords)

    if isinstance(attn_weights, torch.Tensor):
        weights_np = attn_weights.cpu().numpy()
    else:
        weights_np = np.array(attn_weights)

    cmap = plt.cm.get_cmap("RdYlGn")
    for i, (uv, w) in enumerate(zip(coords_np, weights_np)):
        u, v = uv
        radius = 10 + 30 * w
        color  = cmap(w)
        circle = patches.Circle((u, v), radius, linewidth=2, edgecolor=color,
                                 facecolor=(*color[:3], 0.3))
        ax.add_patch(circle)
        label = f"{class_names[i] if class_names else i} ({w:.2f})"
        ax.text(u, v - radius - 5, label, color="white", fontsize=8,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    # Right: attention map overlay
    ax = axes[1]
    ax.imshow(img_np)
    ax.imshow(attn_resized, cmap="hot", alpha=0.5)
    ax.set_title("Audio attention map overlay")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_seld_output(
    peak_coords: torch.Tensor,
    class_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    class_names: list[str] | None = None,
    save_path: str | None = None,
):
    """
    Polar plot of detected sound sources.

    Args:
        peak_coords:  (N, 2) — [azimuth, elevation] in radians
        class_logits: (N, C) — class logits
        valid_mask:   (N,)   — bool
        class_names:  optional list of C strings
    """
    if not _MPL:
        raise ImportError("matplotlib required")

    peak_coords  = peak_coords.cpu()
    class_logits = class_logits.cpu()
    valid_mask   = valid_mask.cpu()

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.set_title("SELD output (azimuth-elevation polar plot)", pad=15)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    for i in range(peak_coords.shape[0]):
        if not valid_mask[i]:
            continue
        az = peak_coords[i, 0].item()
        el = peak_coords[i, 1].item()
        # Map elevation to radial distance: centre = 90° (straight up), edge = 0° (horizon)
        r = math.pi / 2 - el
        dom_cls = class_logits[i].argmax().item()
        label   = class_names[dom_cls] if class_names else str(dom_cls)
        ax.plot(az, r, "o", markersize=12)
        ax.text(az, r, f" {label}", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
