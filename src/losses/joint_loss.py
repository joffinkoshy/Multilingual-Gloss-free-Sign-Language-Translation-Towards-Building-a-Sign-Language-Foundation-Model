"""Joint weighted loss, Eq. 5: L = lambda1*L_LID + lambda2*L_Txt + lambda3*L_Attn.

lambda=(1,5,3) for many-to-one/many-to-many; lambda1=0 for one-to-one, where
the LID branch's forward pass is skipped entirely (Section 2.3) — L_LID is
simply absent from the dict in that case, not zeroed.
"""
from typing import Dict, Optional

import torch
from torch import nn

from src.losses.attention_loss import AttentionCELoss
from src.losses.ctc_losses import CTCLossWrapper


class JointLoss(nn.Module):
    def __init__(
        self,
        lambda_lid: float,
        lambda_txt: float,
        lambda_attn: float,
        txt_pad_index: int,
        label_smoothing: float = 0.0,
        ctc_blank: int = 0,
    ):
        super().__init__()
        self.lambda_lid = lambda_lid
        self.lambda_txt = lambda_txt
        self.lambda_attn = lambda_attn
        self.lid_ctc = CTCLossWrapper(blank=ctc_blank)
        self.txt_ctc = CTCLossWrapper(blank=ctc_blank)
        self.attn_ce = AttentionCELoss(pad_index=txt_pad_index, label_smoothing=label_smoothing)

    def forward(
        self,
        lid_log_probs: Optional[torch.Tensor],   # [B, |F|, |LID_vocab|] or None
        txt_ctc_log_probs: torch.Tensor,            # [B, |F|, |txt_vocab|]
        attn_log_probs: torch.Tensor,                 # [B, |T|, |txt_vocab|]
        feature_lengths: torch.Tensor,
        text_ids: torch.Tensor,                          # [B, |T|] (includes <eos>, no <bos>)
        text_lengths: torch.Tensor,
        lid_ids: Optional[torch.Tensor] = None,           # [B, |T|] token-level LID targets
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        if self.lambda_lid > 0:
            assert lid_log_probs is not None and lid_ids is not None, (
                "lambda_lid > 0 requires the LID branch to be active — "
                "one_to_one configs must set use_lid_branch=false (Section 2.3)."
            )
            losses["lid"] = self.lid_ctc(lid_log_probs, lid_ids, feature_lengths, text_lengths)

        losses["txt_ctc"] = self.txt_ctc(txt_ctc_log_probs, text_ids, feature_lengths, text_lengths)
        losses["attn"] = self.attn_ce(attn_log_probs, text_ids) / text_ids.size(0)

        total = self.lambda_txt * losses["txt_ctc"] + self.lambda_attn * losses["attn"]
        if "lid" in losses:
            total = total + self.lambda_lid * losses["lid"]
        losses["total"] = total
        return losses
