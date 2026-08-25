"""CTC heads + loss wrapper for L_LID (Eq. 2) and L_Txt (Eq. 4)."""
import torch
from torch import nn


class CTCHead(nn.Module):
    """Linear(d_model->|vocab|) -> log_softmax, shared shape for both the
    LID-CTC head (Eq. 2) and the Txt-CTC head (Eq. 4)."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.proj(hidden), dim=-1)


class CTCLossWrapper(nn.Module):
    """blank=0 by convention (matches reference repo's silence-token-at-index-0
    convention, signjoey/vocabulary.py); every CTC vocab used here must
    reserve index 0 for the blank/silence token."""

    def __init__(self, blank: int = 0):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")

    def forward(
        self,
        log_probs: torch.Tensor,       # [B, T, V]
        targets: torch.Tensor,           # [B, T_tgt]
        input_lengths: torch.Tensor,      # [B]
        target_lengths: torch.Tensor,      # [B]
    ) -> torch.Tensor:
        # nn.CTCLoss expects [T, B, V]
        log_probs_tbv = log_probs.transpose(0, 1)
        return self.ctc(log_probs_tbv, targets, input_lengths, target_lengths)
