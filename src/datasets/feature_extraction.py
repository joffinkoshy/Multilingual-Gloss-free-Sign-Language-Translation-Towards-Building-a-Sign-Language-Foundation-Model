"""Frozen SlowFastSign feature interface.

SlowFastSign (Ahn et al., ICASSP 2024, https://github.com/kaistmm/SlowFastSign)
is treated as an external, frozen dependency (Section 2.2) — its own
video-side preprocessing is used as-is, not reimplemented here.

Two important facts recovered by directly inspecting kaistmm/SlowFastSign and
Claire874/Gloss-free-MLSLT's readme (Section 12 item 10, item 1):
  1. The publicly released SlowFastSign checkpoints are trained with GLOSS
     supervision on PHOENIX2014 / PHOENIX2014-T / CSL-Daily only (its README's
     own reported metric is WER, a recognition metric). The MLSLT paper
     explicitly uses a variant retrained with spoken-text supervision instead
     (Section 2.2) — that retrained checkpoint is not public. There is also
     no public SP-10 checkpoint or precomputed-feature release at all.
  2. Therefore: no public precomputed features exist for any of the three
     datasets in the form this paper actually needs. Real feature extraction
     requires (a) obtaining/training a spoken-text-supervised SlowFastSign
     checkpoint, and (b) running it over raw video for all three datasets —
     both out of scope for this sandboxed session (no GPU, no dataset access
     here) and left as the explicit next step for the user's own RTX 3050
     machine. See experiments/results/reference_inspection_notes.md.

`SlowFastSignExtractor` is the real interface: point it at a checkpoint and
it returns frame-level features. `StubFeatureGenerator` produces
deterministic pseudo-features of the right shape so the rest of the pipeline
(Stages 2-7) can be built and tested before that checkpoint exists.
"""
from typing import Optional

import torch


class SlowFastSignExtractor:
    """Wraps a frozen, pretrained SlowFastSign checkpoint.

    d_feat is read from the checkpoint, never hardcoded (Section 12 item 1).
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        state = torch.load(checkpoint_path, map_location=device)
        self.d_feat = self._infer_feature_dim(state)
        self._state = state
        raise NotImplementedError(
            "SlowFastSignExtractor requires the actual SlowFastSign model "
            "definition (kaistmm/SlowFastSign) plus a spoken-text-supervised "
            "checkpoint, neither of which is available in this environment. "
            "Wire the real forward pass in here once both exist; until then "
            "use StubFeatureGenerator for pipeline development."
        )

    @staticmethod
    def _infer_feature_dim(state: dict) -> int:
        for key in ("d_feat", "feature_dim", "hidden_size"):
            if key in state:
                return int(state[key])
        raise ValueError(
            "Could not infer d_feat from checkpoint; the value must be "
            "known before use (Section 12 item 1)."
        )

    def extract(self, video: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StubFeatureGenerator:
    """Deterministic random-feature generator matching expected shapes
    (Stage 2 of the handoff explicitly allows this before real SlowFastSign
    features exist). Same video_id + seed always yields the same tensor.
    """

    def __init__(self, d_feat: int, seed: int = 0, frames_per_token: float = 4.0):
        self.d_feat = d_feat
        self.seed = seed
        self.frames_per_token = frames_per_token

    def __call__(self, video_id: str, n_text_tokens: int) -> torch.Tensor:
        # |F| > |T| is expected (sign video is longer than its translation) —
        # scale frame count off the text length so that holds by construction.
        n_frames = max(int(n_text_tokens * self.frames_per_token), n_text_tokens + 4)
        g = torch.Generator().manual_seed(self._hash(video_id))
        return torch.randn(n_frames, self.d_feat, generator=g)

    def _hash(self, video_id: str) -> int:
        h = self.seed
        for ch in video_id:
            h = (h * 131 + ord(ch)) % (2**31 - 1)
        return h
