"""Shared encoder/decoder building blocks (Section 2.3).

Uses torch.nn.TransformerEncoderLayer/TransformerDecoder directly: these
already are "standard Transformer encoder/decoder layers" (MHSA + FFN +
residual + LayerNorm, masked self-attn + cross-attn for the decoder) per
Vaswani et al. 2017, which is exactly the paper's cited backbone — no need to
reimplement attention from scratch.
"""
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (ASSUMPTION, Section 12: the
    paper does not specify — this matches its cited Vaswani et al. backbone)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class InputProjection(nn.Module):
    """Linear(d_feat->d_model) + PositionalEncoding + Dropout (Section 2.2/2.3).

    d_feat is read from config, not hardcoded — SlowFastSign's exact output
    dim is unconfirmed (Section 12 item 1); this layer is correct for
    whatever value it turns out to be.
    """

    def __init__(self, d_feat: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(d_feat, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.linear(features)
        x = self.pos_enc(x)
        return self.dropout(x)


def build_encoder_layer(d_model: int, num_heads: int, ff_size: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=ff_size,
        dropout=dropout,
        batch_first=True,
    )


def build_decoder(d_model: int, num_heads: int, ff_size: int, dropout: float, num_layers: int) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=ff_size,
        dropout=dropout,
        batch_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=num_layers)


def xavier_init(module: nn.Module) -> None:
    """Xavier init for all linear/attention projection weights (paper
    Appendix B, Section 2.3)."""
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def generate_causal_mask(size: int, device=None) -> torch.Tensor:
    return nn.Transformer.generate_square_subsequent_mask(size, device=device)


def key_padding_mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """True where padded (for `src_key_padding_mask`/`tgt_key_padding_mask`)."""
    device = lengths.device
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)
