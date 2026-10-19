"""Architecture (B): Sign2(LID+Text), the paper's hierarchical model
(Section 2.3):

  F' -> EncoderLayer_1 -> h_int (Eq. 3) -> [LID_CTC_head]
                       -> EncoderLayers_2..N -> h_fin -> [Txt_CTC_head]
                                                       -> Decoder -> L_Attn

`use_lid_branch=False` (one_to_one, lambda1=0) skips *instantiating* the LID
head and its encoder split entirely, not just its loss weight (Section 2.3:
"Implement this as a config flag that disables the LID branch's forward pass
entirely ... to save compute") — in that mode all num_encoder_layers layers
run as one plain stack and h_int is never computed.

family_router (Improvement 1, optional): applied to h_fin as
h' = h + A_family(h) + A_language(h) before the Txt-CTC head / decoder,
composed via config flag per Section 7's separation rule — this file never
special-cases routing logic itself, it only calls the passed-in module.

lid_conditioner (proposed novelty, optional): applied to h_int after the
LID-CTC head as h_int' = h_int + g * W_c softmax(LID logits) before
EncoderLayers_2..N (src/models/lid_self_conditioning.py). Same composition
rule as family_router; requires use_lid_branch=True.
"""
from typing import Optional

import torch
from torch import nn

from src.losses.ctc_losses import CTCHead
from src.models.layers import InputProjection, PositionalEncoding, build_decoder, xavier_init


class Sign2LIDText(nn.Module):
    def __init__(
        self,
        d_feat: int,
        d_model: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        ff_size: int,
        dropout: float,
        lid_split_layer: int,
        use_lid_branch: bool,
        txt_vocab_size: int,
        txt_pad_index: int,
        lid_vocab_size: Optional[int] = None,
        family_router: Optional[nn.Module] = None,
        lid_conditioner: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.use_lid_branch = use_lid_branch
        self.lid_split_layer = lid_split_layer
        self.txt_pad_index = txt_pad_index
        self.family_router = family_router
        if lid_conditioner is not None and not use_lid_branch:
            raise ValueError("lid_conditioner requires use_lid_branch=True (it conditions on the LID-CTC posterior)")

        self.input_proj = InputProjection(d_feat, d_model, dropout)

        def make_layer():
            return nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=ff_size,
                dropout=dropout, batch_first=True,
            )

        if use_lid_branch:
            assert lid_vocab_size is not None, "use_lid_branch=True requires lid_vocab_size"
            assert 0 < lid_split_layer < num_encoder_layers, (
                "lid_split_layer must leave at least one layer on each side "
                "of h_int (Section 2.3: EncoderLayer_1 then EncoderLayers_2..N)."
            )
            self.enc_layers_1 = nn.ModuleList([make_layer() for _ in range(lid_split_layer)])
            self.enc_layers_2 = nn.ModuleList([make_layer() for _ in range(num_encoder_layers - lid_split_layer)])
            self.lid_head = CTCHead(d_model, lid_vocab_size)
        else:
            # Single plain stack — LID branch not instantiated at all.
            self.enc_layers_1 = nn.ModuleList([make_layer() for _ in range(num_encoder_layers)])
            self.enc_layers_2 = nn.ModuleList()
            self.lid_head = None

        self.txt_ctc_head = CTCHead(d_model, txt_vocab_size)

        self.tgt_embed = nn.Embedding(txt_vocab_size, d_model, padding_idx=txt_pad_index)
        self.tgt_pos_enc = PositionalEncoding(d_model)
        self.decoder = build_decoder(d_model, num_heads, ff_size, dropout, num_decoder_layers)
        self.output_proj = nn.Linear(d_model, txt_vocab_size)

        xavier_init(self)
        # Assigned after xavier_init so the conditioner keeps its own init
        # (the gate logit is a 0-dim parameter xavier_init skips anyway).
        self.lid_conditioner = lid_conditioner

    def encode(self, features: torch.Tensor, src_key_padding_mask: torch.Tensor):
        x = self.input_proj(features)
        for layer in self.enc_layers_1:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        h_int = x

        lid_log_probs = None
        if self.use_lid_branch:
            lid_log_probs = self.lid_head(h_int)
            if self.lid_conditioner is not None:
                x = self.lid_conditioner(h_int, lid_log_probs)
            for layer in self.enc_layers_2:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
        h_fin = x

        if self.family_router is not None:
            h_fin = self.family_router(h_fin)

        return h_fin, lid_log_probs

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
    ) -> dict:
        h_fin, lid_log_probs = self.encode(features, src_key_padding_mask)
        txt_ctc_log_probs = self.txt_ctc_head(h_fin)
        attn_log_probs = self.decode(tgt_input_ids, h_fin, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)
        return {
            "lid_log_probs": lid_log_probs,
            "txt_ctc_log_probs": txt_ctc_log_probs,
            "attn_log_probs": attn_log_probs,
            "h_fin": h_fin,
        }
