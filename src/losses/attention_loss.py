"""Teacher-forced cross-entropy for L_Attn, with optional label smoothing
(off by default — Section 12 item 6, confirmed against reference repo)."""
import torch
from torch import nn


class AttentionCELoss(nn.Module):
    def __init__(self, pad_index: int, label_smoothing: float = 0.0):
        super().__init__()
        self.pad_index = pad_index
        self.label_smoothing = label_smoothing
        if label_smoothing <= 0.0:
            self.criterion = nn.NLLLoss(ignore_index=pad_index, reduction="sum")
        else:
            self.criterion = nn.KLDivLoss(reduction="sum")

    def _smooth_targets(self, targets: torch.Tensor, vocab_size: int) -> torch.Tensor:
        smooth = targets.new_zeros((targets.size(0), vocab_size), dtype=torch.float)
        smooth.fill_(self.label_smoothing / (vocab_size - 2))
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        smooth[:, self.pad_index] = 0.0
        pad_positions = (targets == self.pad_index).nonzero(as_tuple=False)
        if len(pad_positions) > 0:
            smooth.index_fill_(0, pad_positions.squeeze(-1), 0.0)
        return smooth

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """log_probs: [B, T, V] (log-softmax already applied), targets: [B, T]."""
        flat_log_probs = log_probs.contiguous().view(-1, log_probs.size(-1))
        flat_targets = targets.contiguous().view(-1)
        if self.label_smoothing > 0.0:
            smoothed = self._smooth_targets(flat_targets, log_probs.size(-1))
            return self.criterion(flat_log_probs, smoothed)
        return self.criterion(flat_log_probs, flat_targets)
