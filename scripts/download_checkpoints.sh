#!/usr/bin/env bash
# Download pretrained model checkpoints for Audio-VLA
set -e

CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

# ── LAION CLAP ────────────────────────────────────────────────────────────────
echo "Downloading LAION CLAP checkpoint..."
pip install laion-clap --quiet
python - <<'EOF'
import laion_clap, os
model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
# This will download to ~/.cache by default; copy to checkpoints/ if desired
model.load_ckpt()  # uses default online download
print("CLAP checkpoint ready.")
EOF

# ── SmolVLA via lerobot ───────────────────────────────────────────────────────
echo "Downloading SmolVLA checkpoint via lerobot..."
pip install lerobot --quiet
python - <<'EOF'
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.save_pretrained("checkpoints/smolvla_base")
print("SmolVLA checkpoint saved to checkpoints/smolvla_base")
EOF

echo ""
echo "All checkpoints downloaded."
echo "  CLAP:    ~/.cache/... (or specify --clap_checkpoint in config)"
echo "  SmolVLA: checkpoints/smolvla_base"
