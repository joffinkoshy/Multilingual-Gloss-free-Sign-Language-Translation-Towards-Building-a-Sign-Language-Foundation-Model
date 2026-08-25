"""Architecture (A): vanilla Transformer baseline (Section 2.3).

No CTC branch at all — this is "Individual (10)"/"Universal (1)" in
Table 4/8 and "Baseline" in Table 3.
"""
import torch
from torch import nn

from src.models.layers import InputProjection, PositionalEncoding, build_decoder, xavier_init


class BaselineTransformer(nn.Module):
    def __init__(
        self,
        d_feat: int,
        d_model: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        ff_size: int,
        dropout: float,
        txt_vocab_size: int,
        txt_pad_index: int,
    ):
        super().__init__()
        self.txt_pad_index = txt_pad_index
        self.input_proj = InputProjection(d_feat, d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.tgt_embed = nn.Embedding(txt_vocab_size, d_model, padding_idx=txt_pad_index)
        self.tgt_pos_enc = PositionalEncoding(d_model)
        self.decoder = build_decoder(d_model, num_heads, ff_size, dropout, num_decoder_layers)
        self.output_proj = nn.Linear(d_model, txt_vocab_size)
        xavier_init(self)

    def encode(self, features: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt_input_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt = self.tgt_pos_enc(self.tgt_embed(tgt_input_ids))
        hidden = self.decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return torch.log_softmax(self.output_proj(hidden), dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(features, src_key_padding_mask)
        return self.decode(tgt_input_ids, memory, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)
