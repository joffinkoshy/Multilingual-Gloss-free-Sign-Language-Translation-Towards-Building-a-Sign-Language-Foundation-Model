"""Improvement 3 — multilingual pretraining, cost-reduced form (Section 8).

Frozen SlowFastSign backbone (external, Section 2.2) + a small SignCLIP-style
contrastive projection head trained on a modest multilingual subset of
JWSign/Fleurs-ASL: few layers get gradients, small batch, few epochs. This is
deliberately NOT full pretraining from scratch (Section 11) — do not revert
to that without discussing it first.

Produced checkpoint is loaded into the main model's feature pipeline before
Stage 8 training (config: data.pretraining.checkpoint) — `ProjectedFeatureWrapper`
is the composition point: it sits between the frozen backbone's raw features
and Sign2LIDText's InputProjection, config-gated, never edited into the
reproduction files directly (Section 7's separation rule).
"""
import torch
from torch import nn


class ContrastiveProjectionHead(nn.Module):
    """Projects frozen per-frame features into a joint sign/text space.
    Mean-pools over time to get one clip-level embedding for the contrastive
    loss during pretraining; at inference time (composed into the main
    model) it instead returns per-frame projections so InputProjection still
    sees a full [T, proj_dim] sequence."""

    def __init__(self, d_feat: int, proj_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_feat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def project_frames(self, features: torch.Tensor) -> torch.Tensor:
        """[B, T, d_feat] -> [B, T, proj_dim], used for composition into the
        main model (Improvement 3's actual downstream effect)."""
        return self.net(features)

    def clip_embedding(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> torch.Tensor:
        """[B, T, d_feat] + lengths -> [B, proj_dim] mean-pooled embedding,
        used only during the contrastive pretraining stage itself."""
        projected = self.project_frames(features)
        mask = (
            torch.arange(features.size(1), device=features.device)[None, :] < feature_lengths[:, None]
        ).float().unsqueeze(-1)
        summed = (projected * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts


class TextEncoderForPretraining(nn.Module):
    """Minimal text-side encoder for the contrastive objective only — not the
    main translation decoder's embeddings, kept separate so the pretraining
    stage doesn't entangle with the reproduction's own vocab/decoder state."""

    def __init__(self, vocab_size: int, proj_dim: int = 256, hidden_dim: int = 512, pad_index: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_index)
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.pad_index = pad_index

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        mask = (text_ids != self.pad_index).float().unsqueeze(-1)
        emb = self.embed(text_ids)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.proj(pooled)


def info_nce_loss(sign_emb: torch.Tensor, text_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE over a batch of (sign, text) pairs, SignCLIP-style."""
    sign_emb = nn.functional.normalize(sign_emb, dim=-1)
    text_emb = nn.functional.normalize(text_emb, dim=-1)
    logits = sign_emb @ text_emb.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_s2t = nn.functional.cross_entropy(logits, labels)
    loss_t2s = nn.functional.cross_entropy(logits.t(), labels)
    return (loss_s2t + loss_t2s) / 2


class ProjectedFeatureWrapper(nn.Module):
    """Composition point used by the main model when
    config.data.pretraining.enabled=true: wraps the frozen backbone's raw
    features through a pretrained, then-frozen ContrastiveProjectionHead
    before InputProjection sees them. d_model config's `d_feat` must then be
    set to `proj_dim`, not the backbone's raw d_feat."""

    def __init__(self, head: ContrastiveProjectionHead):
        super().__init__()
        self.head = head
        for p in self.head.parameters():
            p.requires_grad_(False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.head.project_frames(features)
