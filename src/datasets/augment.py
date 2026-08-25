"""Improvement 2 — cost-reduced data augmentation (Section 8).

Deliberate cost decision carried from reproduction_plan.md (Section 11 of the
handoff): video-level transforms (crop/flip/time-warp/time-mask) and
feature-space transforms (SpecAugment-style masking/mixup on cached
SlowFastSign features) — NOT a trained SignBT sign-generation model. Do not
"fix" this by reverting to a generative approach without discussing it first.

Video-level transforms need raw frames, which requires the real SlowFastSign
video pipeline (Section 2.2, external/frozen) — `VideoAugmenter` documents
the intended transform set as a config-driven hook to be wired in once real
video decoding exists; it is not runnable against synthetic features.
`FeatureAugmenter` operates on already-extracted (or stub) feature tensors
and is fully runnable today.
"""
from typing import List, Optional

import torch


class FeatureAugmenter:
    """SpecAugment-style masking + mixup, applied to feature tensors
    [T, d_feat] before batching."""

    def __init__(
        self,
        transforms: List[str],
        time_mask_max: int = 5,
        feature_mask_max: int = 8,
        mixup_alpha: float = 0.2,
        seed: int = 0,
    ):
        self.transforms = set(transforms)
        self.time_mask_max = time_mask_max
        self.feature_mask_max = feature_mask_max
        self.mixup_alpha = mixup_alpha
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        out = features
        if "spec_augment" in self.transforms:
            out = self._spec_augment(out)
        return out

    def _spec_augment(self, features: torch.Tensor) -> torch.Tensor:
        out = features.clone()
        T, D = out.shape
        if T > 1:
            t_len = int(torch.randint(0, min(self.time_mask_max, T), (1,), generator=self.generator))
            if t_len > 0:
                t0 = int(torch.randint(0, T - t_len + 1, (1,), generator=self.generator))
                out[t0 : t0 + t_len, :] = 0.0
        f_len = int(torch.randint(0, min(self.feature_mask_max, D), (1,), generator=self.generator))
        if f_len > 0:
            f0 = int(torch.randint(0, D - f_len + 1, (1,), generator=self.generator))
            out[:, f0 : f0 + f_len] = 0.0
        return out

    def mixup(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """Feature-space mixup between two (already length-aligned) examples.
        Caller is responsible for truncating/padding to a common length —
        this is a weaker intervention than true video-level mixup, an
        accepted tradeoff for compute (Section 8's stated risk)."""
        lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample())
        n = min(features_a.size(0), features_b.size(0))
        return lam * features_a[:n] + (1 - lam) * features_b[:n]


class VideoAugmenter:
    """Placeholder for the video-level half of Improvement 2
    (crop/flip/time-warp/time-mask). Requires raw video frames, which are
    out of scope for this sandboxed session (Section 12 item 10) — wire this
    in against the real SlowFastSign video-loading path before real training."""

    def __init__(self, transforms: List[str]):
        self.transforms = transforms
        raise NotImplementedError(
            "VideoAugmenter needs raw video frame access (SlowFastSign's own "
            "loading pipeline, Section 2.2) which isn't available in this "
            "environment. Use FeatureAugmenter for pipeline development; "
            "implement this against real frames on the target machine."
        )
