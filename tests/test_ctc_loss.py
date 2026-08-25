"""Stage 6 verification: CTC loss finite/positive on a hand-constructed toy
alignment with known expected loss; attention CE matches
torch.nn.functional.cross_entropy on an equivalent manual case."""
import torch
import torch.nn.functional as F

from src.losses.attention_loss import AttentionCELoss
from src.losses.ctc_losses import CTCLossWrapper


def test_ctc_loss_finite_and_matches_reference():
    torch.manual_seed(0)
    T, B, V = 10, 2, 5
    logits = torch.randn(T, B, V)
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = torch.tensor([1, 2, 3, 4])
    input_lengths = torch.tensor([T, T])
    target_lengths = torch.tensor([2, 2])

    reference = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean")

    wrapper = CTCLossWrapper(blank=0)
    got = wrapper(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)

    assert torch.isfinite(got)
    assert got.item() > 0
    assert torch.allclose(got, reference, atol=1e-5)


def test_attention_ce_matches_cross_entropy_no_smoothing():
    torch.manual_seed(0)
    B, T, V = 3, 4, 6
    pad_index = 0
    logits = torch.randn(B, T, V)
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = torch.randint(1, V, (B, T))
    targets[0, -1] = pad_index  # exercise pad-ignoring path

    loss_fn = AttentionCELoss(pad_index=pad_index, label_smoothing=0.0)
    got = loss_fn(log_probs, targets)

    reference = F.nll_loss(
        log_probs.reshape(-1, V), targets.reshape(-1), ignore_index=pad_index, reduction="sum"
    )
    assert torch.allclose(got, reference, atol=1e-5)
