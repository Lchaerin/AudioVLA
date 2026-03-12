# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**Audio-VLA** fuses a **Binaural SELD model** (Sound Event Localization & Detection) with **SmolVLA** (a 450M VLA robot policy) via a lightweight **Fusion Module** (~5–10M params). Given a binaural audio stream, an RGB image, and a language command (e.g. "pick up the phone making the siren sound"), the system localises the relevant sound source and routes that information into SmolVLA's visual-token stream so the robot selects the correct object.

---

## Common Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Generate simulation training data
```bash
bash scripts/generate_sim_data.sh data/sim_episodes 5000
# With real AudioSet clips:
bash scripts/generate_sim_data.sh data/sim_episodes 5000 /path/to/audioset
```

### Download pretrained checkpoints (CLAP + SmolVLA)
```bash
bash scripts/download_checkpoints.sh
```

### Phase 1 – Train SELD model
```bash
python training/train_seld.py --config configs/default.yaml
```

### Phase 1 – Train CLAP Projection MLP
```bash
python training/train_clap_proj.py --config configs/default.yaml
```

### Phase 2 – Train Fusion Module (main training)
```bash
python training/train_fusion.py --config configs/sim_training.yaml
# Fine-tune on real data:
python training/train_fusion.py --config configs/real_finetune.yaml
```

### Evaluate audio grounding accuracy
```bash
python evaluation/eval_grounding.py \
    --config configs/default.yaml \
    --fusion_ckpt outputs/sim_training/fusion/best_model.pt \
    --data_root data/sim_episodes
```

### Run inference demo
```bash
python scripts/run_inference.py \
    --config configs/default.yaml \
    --audio path/to/binaural.wav \
    --image path/to/image.png \
    --command "Pick up the object making the siren sound." \
    --fusion_ckpt outputs/sim_training/fusion/best_model.pt \
    --visualize
```

---

## Architecture

### Three-Phase Training Strategy
1. **Phase 1a** – Train `ResNetConformerSELD` on STARSS23 / DCASE2025 stereo data (frozen during Phase 2).
2. **Phase 1b** – Train `CLAPProjection` MLP to map SELD class logits → LAION CLAP embedding space.
3. **Phase 2** – Train only the Fusion Module (~5–10M params) with SELD, SmolVLA, and CLAP frozen.

### Inference Data Flow
```
binaural audio → ResNetConformerSELD → SELDOutput {peak_coords, class_logits, energy, valid_mask}
                                          ↓
                               SoundTokenEncoder → sound_tokens (B, N, D_s)
                                          ↓
language command → SmolVLA lang encoder → lang_tokens (B, T, D_l)
                                          ↓
                        AudioLanguageCrossAttention → attn_weights (B, N), audio_context (B, D_h)
                                          ↓
                      AzElToPixel (camera intrinsics) → pixel_coords (B, N, 2)
                                          ↓
               AudioAttentionMapGenerator (Gaussian splatting) → attn_map (B, H_grid, W_grid)
                                          ↓
image → SmolVLA vision encoder → visual_tokens (B, 64, D_v)
                                          ↓
                  AudioVisualFusion (gated modulation) → fused_tokens (B, 64, D_v)
                                          ↓
                       SmolVLA action expert → action_chunk (B, 50, action_dim)
```

### Key Modules and Files

| File | Role |
|------|------|
| `models/seld/resnet_conformer.py` | SELD model — ResNet-Conformer + EINv2 head |
| `models/seld/sound_token_encoder.py` | Converts SELD output → sound tokens via sinusoidal + class + energy embeddings |
| `models/fusion/audio_language_cross_attention.py` | Multi-head cross-attention: lang queries over sound keys → per-source importance |
| `models/fusion/azel_to_pixel.py` | Spherical → image pixel projection (pinhole camera model) |
| `models/fusion/audio_attention_map.py` | Gaussian splatting of attention weights onto visual token grid |
| `models/fusion/audio_visual_fusion.py` | Gated modulation: fuse audio attention into visual tokens with learnable residual scale |
| `models/fusion/clap_projection.py` | MLP projecting SELD class logits into CLAP embedding space |
| `models/vla/smolvla_wrapper.py` | Wraps SmolVLA to expose `encode_language`, `encode_vision`, `predict_action` |
| `models/audio_vla_pipeline.py` | Full inference pipeline |
| `training/losses.py` | `AudioVLALoss` (grounding CE + action MSE) + `CLAPProjectionLoss` |
| `data/dataset.py` | `AudioVLADataset` (disk) + `DummyAudioVLADataset` (testing) |
| `data/sim_generator/` | Scene builder, binaural renderer (ITD/ILD), episode collector |

### Fusion Module Insertion Point
SmolVLA's action expert reads from the **middle layer** (layer L/2) of the language decoder. The Fusion Module intercepts the 64 visual tokens at that layer via `SmolVLAWrapper` forward hooks, applies gated modulation, and passes the modified tokens to the action expert. If direct injection is not possible with a given lerobot version, a concat-injection fallback is documented in `smolvla_wrapper.py`.

### Hyperparameters (default.yaml)
- `D_s=256` (sound token dim), `D_l=576` (SmolVLA lang dim), `D_v=576` (visual dim)
- `N_max=8` (max simultaneous sound sources), `sigma_init=20.0` (Gaussian splatting sigma)
- Optimizer: AdamW lr=1e-4, weight_decay=0.01, cosine schedule with 500-step warmup

### Known Integration Requirements
- **lerobot** must be installed for SmolVLA; adapt `SmolVLAWrapper.predict_action` to the specific lerobot API version (it raises `NotImplementedError` by default).
- **laion-clap** must be installed and the CLAP checkpoint path set in `configs/default.yaml` under `checkpoints.clap_checkpoint`.
- Episode data must follow the layout described in `data/dataset.py` (`audio.wav`, `image.png`, `meta.json` per episode directory).
