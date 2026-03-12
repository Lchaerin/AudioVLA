import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioVLALoss(nn.Module):
    """
    Combined loss for Fusion Module training.

    Components:
      1. grounding_loss  — cross-entropy encouraging attention on the correct sound source
      2. action_loss     — MSE on the action chunk (flow-matching compatible)
    """

    def __init__(
        self,
        lambda_loc: float = 1.0,
        lambda_cls: float = 0.5,
        lambda_action: float = 1.0,
    ):
        super().__init__()
        self.lambda_loc    = lambda_loc
        self.lambda_cls    = lambda_cls
        self.lambda_action = lambda_action

    def forward(self, pred: dict, target: dict):
        """
        Args:
            pred:
                attn_weights:    (B, N)           — predicted per-source importance
                actions:         (B, 50, action_dim) — predicted action chunk
            target:
                target_sound_idx: (B,)            — GT sound source index
                target_actions:   (B, 50, action_dim) — GT action chunk
        Returns:
            total_loss: scalar tensor
            metrics:    dict of float scalars for logging
        """
        # 1. Audio grounding: attention should peak at the correct source
        grounding_loss = F.cross_entropy(
            pred["attn_weights"],       # (B, N) — treated as logits
            target["target_sound_idx"], # (B,)
        )

        # 2. Action regression
        action_loss = F.mse_loss(pred["actions"], target["target_actions"])

        total = self.lambda_loc * grounding_loss + self.lambda_action * action_loss

        return total, {
            "grounding_loss": grounding_loss.item(),
            "action_loss":    action_loss.item(),
            "total_loss":     total.item(),
        }


class CLAPProjectionLoss(nn.Module):
    """
    Loss for training CLAPProjection MLP.
    Minimises MSE between projected class logits and CLAP text embeddings.
    """

    def forward(self, pred_embed: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_embed:   (B, clap_dim) or (B, N, clap_dim)
            target_embed: same shape
        Returns:
            loss: scalar
        """
        return F.mse_loss(pred_embed, target_embed)
