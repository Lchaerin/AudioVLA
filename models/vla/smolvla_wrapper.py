"""
SmolVLA wrapper for Audio-VLA.

Wraps lerobot SmolVLAPolicy to expose:
  - encode_language(text) → intermediate language token features
  - encode_vision(image)  → visual token features
  - predict_action(visual_feats, lang_tokens, robot_state) → action chunk

SmolVLA is kept frozen throughout training; only fusion modules are updated.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SmolVLAWrapper(nn.Module):
    """
    Thin wrapper around lerobot SmolVLAPolicy exposing the intermediate features
    required by the Audio-VLA fusion module.

    The fusion point is the VLM's intermediate layer (layer N = L/2), where the
    action expert also reads from. We intercept the visual tokens at that layer.
    """

    def __init__(self, pretrained: str = "lerobot/smolvla_base", device: str = "cuda"):
        super().__init__()
        self._loaded = False
        self.pretrained = pretrained
        self.device_str = device

        # Lazy-load to avoid import errors when lerobot is not installed
        self._policy = None
        self._hooks = []

        # Buffers filled by forward hooks
        self._visual_feats: Optional[torch.Tensor] = None
        self._lang_tokens: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self):
        if self._loaded:
            return
        try:
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as e:
            raise ImportError(
                "lerobot is required. Install with: pip install lerobot"
            ) from e

        policy = SmolVLAPolicy.from_pretrained(self.pretrained)
        policy = policy.to(self.device_str)
        policy.eval()
        for p in policy.parameters():
            p.requires_grad = False

        self._policy = policy
        self._register_hooks()
        self._loaded = True

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        policy = self._policy

        # Hook into vision encoder output
        def vision_hook(module, input, output):
            # output shape depends on SmolVLA internals; store as-is
            self._visual_feats = output

        # Hook into language decoder at intermediate layer
        def lang_hook(module, input, output):
            if isinstance(output, tuple):
                self._lang_tokens = output[0]
            else:
                self._lang_tokens = output

        # Attach hooks — adjust layer names to match actual SmolVLA internals
        try:
            vision_module = policy.model.vision_encoder
            h1 = vision_module.register_forward_hook(vision_hook)
            self._hooks.append(h1)
        except AttributeError:
            pass  # Will fall back to direct access

        try:
            num_layers = len(policy.model.language_model.model.layers)
            mid_layer = policy.model.language_model.model.layers[num_layers // 2]
            h2 = mid_layer.register_forward_hook(lang_hook)
            self._hooks.append(h2)
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_language(
        self,
        text: str | list[str],
        audio_prefix_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the language encoder and return intermediate features,
        optionally prepending audio spatial prefix tokens.

        audio_prefix_tokens를 prepend하면 언어 디코더가 소리 위치 정보를
        직접 참조하여 "두 소리 사이의 물체" 같은 공간 관계 추론이 가능해짐.

        Args:
            text:                str or list[str]
            audio_prefix_tokens: (B, N, D_l) or None — AudioPrefixEncoder 출력.
                                 None이면 기존 동작과 동일.
        Returns:
            lang_tokens: (B, N+T, D_l) if prefix provided, else (B, T, D_l)
        """
        self._load()
        if isinstance(text, str):
            text = [text]

        # Tokenise
        tokenizer = self._policy.model.language_model.tokenizer
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device_str)

        # Forward through language model up to intermediate layer — hooks capture output
        self._lang_tokens = None
        _ = self._policy.model.language_model(**inputs)

        if self._lang_tokens is not None:
            lang_tokens = self._lang_tokens
        else:
            # Fallback: return last hidden state
            out = self._policy.model.language_model(**inputs, output_hidden_states=True)
            lang_tokens = out.hidden_states[len(out.hidden_states) // 2]

        # Prepend audio spatial prefix tokens
        if audio_prefix_tokens is not None:
            lang_tokens = torch.cat([audio_prefix_tokens, lang_tokens], dim=1)

        return lang_tokens

    @torch.no_grad()
    def encode_vision(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run the vision encoder and return token features.

        Args:
            image: (B, 3, H, W) — normalised RGB
        Returns:
            visual_features: (B, num_visual_tokens, D_v)
        """
        self._load()
        self._visual_feats = None
        _ = self._policy.model.vision_encoder(image)

        if self._visual_feats is not None:
            feat = self._visual_feats
            if feat.dim() == 4:
                # (B, C, H, W) → (B, H*W, C)
                B, C, H, W = feat.shape
                feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            return feat

        # Fallback: use pixel_values path
        out = self._policy.model.vision_encoder(image, output_hidden_states=True)
        return out.last_hidden_state

    def predict_action(
        self,
        visual_features: torch.Tensor,
        lang_tokens: torch.Tensor,
        robot_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the action expert given (optionally fused) visual features.

        In this wrapper we cannot easily bypass SmolVLA's internal token pipeline
        and inject modified visual tokens. Two strategies are supported:
          1. Hook-based injection (preferred, model-version dependent)
          2. Concat injection: append a single fused-audio token

        This implementation uses strategy 2 for robustness.

        Args:
            visual_features: (B, num_tokens, D_v) — fused visual tokens
            lang_tokens:     (B, T, D_l)
            robot_state:     (B, state_dim) or None
        Returns:
            actions: (B, 50, action_dim)
        """
        self._load()
        policy = self._policy

        # SmolVLAPolicy.predict_action signature expects a batch dict.
        # We pass the raw image path for compatibility and patch vision output.
        # This is a simplified stub — adapt to actual lerobot API.
        raise NotImplementedError(
            "predict_action requires adaptation to your specific lerobot version. "
            "See models/vla/smolvla_wrapper.py for details on injection strategies."
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
