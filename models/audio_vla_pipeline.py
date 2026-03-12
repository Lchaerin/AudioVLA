"""
Full Audio-VLA inference pipeline.

Binaural audio + RGB image + language command → robot action chunk
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .seld import ResNetConformerSELD, SoundTokenEncoder
from .fusion import (
    AudioLanguageCrossAttention,
    AzElToPixel,
    AudioAttentionMapGenerator,
    AudioVisualFusion,
    CLAPProjection,
    AudioPrefixEncoder,
)
from .vla import SmolVLAWrapper


def _load_clap(checkpoint_path: str, device: str):
    """Load pretrained LAION CLAP model (frozen)."""
    try:
        import laion_clap
    except ImportError:
        raise ImportError("laion-clap required. Install with: pip install laion-clap")

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(checkpoint_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


class AudioVLAPipeline(nn.Module):
    """
    Full Audio-VLA inference pipeline integrating:
      SELD → SoundTokenEncoder → AudioLanguageCrossAttention
      → AzElToPixel → AudioAttentionMapGenerator
      → SmolVLA (vision + language) → AudioVisualFusion
      → SmolVLA action expert → robot actions
    """

    def __init__(self, config):
        super().__init__()
        device = getattr(config, "device", "cuda")
        D_s    = config.model.D_s
        D_l    = config.model.D_l
        D_v    = config.model.D_v
        D_h    = config.model.D_hidden
        n_cls  = config.model.num_seld_classes

        # ── Frozen models ────────────────────────────────────────────
        self.seld = ResNetConformerSELD(num_classes=n_cls)
        self.smolvla = SmolVLAWrapper(
            pretrained=config.checkpoints.smolvla_checkpoint, device=device
        )
        self.clap = _load_clap(config.checkpoints.clap_checkpoint, device)

        # ── Trainable fusion modules (~5-10 M params) ─────────────────
        self.sound_token_enc = SoundTokenEncoder(D_s=D_s, num_classes=n_cls)
        # AudioPrefixEncoder: SELD 위치 정보 → 언어 prefix 토큰 (B, N, D_l)
        # LLM이 "두 소리 사이의 물체" 같은 공간 관계 추론을 직접 처리할 수 있게 함
        self.audio_prefix_enc = AudioPrefixEncoder(
            D_l=D_l, num_classes=n_cls, D_s=D_s
        )
        self.audio_lang_cross_attn = AudioLanguageCrossAttention(
            D_s=D_s, D_l=D_l, D_hidden=D_h, num_heads=config.model.num_heads
        )
        self.azel_to_pixel = AzElToPixel()
        self.attn_map_gen = AudioAttentionMapGenerator(sigma=config.model.sigma_init)
        self.av_fusion = AudioVisualFusion(D_v=D_v, audio_context_dim=D_h)
        self.clap_proj = CLAPProjection(
            num_classes=n_cls, clap_dim=config.model.clap_dim
        )

        self._device = device

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def freeze_pretrained(self):
        """Freeze SELD, SmolVLA, and CLAP; leave fusion modules trainable."""
        for m in [self.seld, self.clap]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def trainable_parameters(self):
        """Return parameters of fusion modules only."""
        modules = [
            self.sound_token_enc,
            self.audio_prefix_enc,
            self.audio_lang_cross_attn,
            self.attn_map_gen,
            self.av_fusion,
            self.clap_proj,
        ]
        return [p for m in modules for p in m.parameters() if p.requires_grad]

    def save_fusion(self, path: str | Path):
        """Save only fusion module weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "sound_token_enc":       self.sound_token_enc.state_dict(),
            "audio_prefix_enc":      self.audio_prefix_enc.state_dict(),
            "audio_lang_cross_attn": self.audio_lang_cross_attn.state_dict(),
            "attn_map_gen":          self.attn_map_gen.state_dict(),
            "av_fusion":             self.av_fusion.state_dict(),
            "clap_proj":             self.clap_proj.state_dict(),
        }
        torch.save(state, path)

    def load_fusion(self, path: str | Path):
        """Load fusion module weights."""
        state = torch.load(path, map_location=self._device)
        self.sound_token_enc.load_state_dict(state["sound_token_enc"])
        self.audio_prefix_enc.load_state_dict(state["audio_prefix_enc"])
        self.audio_lang_cross_attn.load_state_dict(state["audio_lang_cross_attn"])
        self.attn_map_gen.load_state_dict(state["attn_map_gen"])
        self.av_fusion.load_state_dict(state["av_fusion"])
        self.clap_proj.load_state_dict(state["clap_proj"])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        binaural_audio: torch.Tensor,
        image: torch.Tensor,
        language_command: str,
        camera_intrinsic: torch.Tensor,
        camera_extrinsic: Optional[torch.Tensor] = None,
        robot_state: Optional[torch.Tensor] = None,
    ):
        """
        Full pipeline inference.

        Args:
            binaural_audio:   (2, T_samples) — stereo waveform
            image:            (3, H, W)      — normalised RGB
            language_command: str
            camera_intrinsic: (3, 3)
            camera_extrinsic: (4, 4) or None
            robot_state:      (state_dim,) or None
        Returns:
            actions:    (50, action_dim) — action chunk
            debug_info: dict             — intermediate outputs for visualisation
        """
        # Add batch dimension
        audio = binaural_audio.unsqueeze(0).to(self._device)  # (1, 2, T)
        img   = image.unsqueeze(0).to(self._device)            # (1, 3, H, W)
        K     = camera_intrinsic.to(self._device)
        if camera_extrinsic is not None:
            camera_extrinsic = camera_extrinsic.to(self._device)

        img_h, img_w = img.shape[-2], img.shape[-1]

        # Step 1: SELD
        seld_out = self.seld(audio)

        # Step 2: Sound tokens (for cross-attention)
        sound_tokens = self.sound_token_enc(
            seld_out.peak_coords, seld_out.class_logits, seld_out.energy
        )  # (1, N, D_s)

        # Step 3: az/el → pixel (언어 인코딩 전에 먼저 수행)
        # prefix token 생성에 픽셀 좌표가 필요하므로 순서 변경
        pixel_coords, in_frame = self.azel_to_pixel(
            seld_out.peak_coords, K, camera_extrinsic,
            img_h=img_h, img_w=img_w,
        )  # (1, N, 2), (1, N)

        # Step 4: Audio prefix tokens 생성
        # 정규화 2D 좌표(in-frame) 또는 정규화 az/el(out-of-frame)을 D_l 토큰으로 인코딩
        # 이 토큰들이 언어 시퀀스 앞에 prepend되어 LLM이 공간 관계를 직접 추론할 수 있음
        audio_prefix_tokens = self.audio_prefix_enc(
            pixel_coords=pixel_coords,
            az_el_coords=seld_out.peak_coords,
            class_logits=seld_out.class_logits,
            energy=seld_out.energy,
            in_frame=in_frame,
            valid_mask=seld_out.valid_mask,
            img_h=img_h,
            img_w=img_w,
        )  # (1, N, D_l)

        # Step 5: Language encoding with audio prefix prepended
        # lang_tokens shape: (1, N+T, D_l) — N개 audio prefix + T개 text tokens
        lang_tokens = self.smolvla.encode_language(
            language_command, audio_prefix_tokens=audio_prefix_tokens
        )

        # Step 6: Audio-Language Cross-Attention
        # 확장된 lang_tokens(N+T)를 query로 사용 → 더 풍부한 문맥으로 소스 중요도 결정
        attn_weights, audio_context = self.audio_lang_cross_attn(
            sound_tokens, lang_tokens, seld_out.valid_mask
        )  # (1, N), (1, D_hidden)

        # Step 7: Audio attention map (on visual token grid)
        H_feat = W_feat = 8  # SmolVLA has 64 = 8×8 visual tokens
        audio_attn_map = self.attn_map_gen(
            pixel_coords, attn_weights, in_frame, H_feat, W_feat
        )  # (1, 8, 8)

        # Step 8: Visual encoding
        visual_features = self.smolvla.encode_vision(img)  # (1, 64, D_v)

        # Step 9: AudioVisual Fusion
        fused_features = self.av_fusion(
            visual_features, audio_attn_map, audio_context
        )  # (1, 64, D_v)

        # Step 10: Action prediction
        actions = self.smolvla.predict_action(fused_features, lang_tokens, robot_state)

        debug_info = {
            "peak_coords":        seld_out.peak_coords[0].cpu(),
            "class_logits":       seld_out.class_logits[0].cpu(),
            "attn_weights":       attn_weights[0].cpu(),
            "pixel_coords":       pixel_coords[0].cpu(),
            "audio_attn_map":     audio_attn_map[0].cpu(),
            "in_frame_mask":      in_frame[0].cpu(),
            "audio_prefix_tokens": audio_prefix_tokens[0].cpu(),
        }

        return actions[0], debug_info
