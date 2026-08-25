"""Improvement 1 — language-family-aware routing (Section 8).

h' = h + A_family(h) + A_language(h), lightweight bottleneck-MLP residual
adapters, not a full parallel encoder per family. Kept in its own module and
only ever composed into Sign2LIDText via the `family_router=` constructor arg
(Section 7's separation rule) — the reproduction files never import this
module or branch on it internally.

language_to_family mapping is a config-driven, explicitly-flagged
implementation choice (Section 12 item 9), not a carried-over research
result — see configs/v1_family_routing.yaml for the mapping actually used
and its rationale.
"""
from typing import Dict, List

import torch
from torch import nn


class BottleneckAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class FamilyRouter(nn.Module):
    def __init__(self, d_model: int, language_to_family: Dict[str, str], bottleneck_dim: int = 64):
        super().__init__()
        self.language_to_family = language_to_family
        self.families = sorted(set(language_to_family.values()))
        self.languages = sorted(language_to_family.keys())

        self.family_adapters = nn.ModuleDict({
            fam: BottleneckAdapter(d_model, bottleneck_dim) for fam in self.families
        })
        self.language_adapters = nn.ModuleDict({
            lang: BottleneckAdapter(d_model, bottleneck_dim) for lang in self.languages
        })
        self._current_src_langs: List[str] = []

    def set_batch_languages(self, src_langs: List[str]) -> None:
        """Trainer calls this before the forward pass that needs routing,
        since nn.Module.forward can't take an extra non-tensor batch arg
        cleanly through nn.Sequential-style composition points."""
        self._current_src_langs = src_langs

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        assert len(self._current_src_langs) == h.size(0), (
            "FamilyRouter.set_batch_languages(...) must be called with one "
            "language per batch element before forward()."
        )
        out = h.clone()
        for i, lang in enumerate(self._current_src_langs):
            family = self.language_to_family[lang]
            out[i] = h[i] + self.family_adapters[family](h[i]) + self.language_adapters[lang](h[i])
        return out
