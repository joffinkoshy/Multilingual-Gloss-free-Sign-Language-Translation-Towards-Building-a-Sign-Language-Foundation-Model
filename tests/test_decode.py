"""Stage 6 verification: decode a tiny dummy batch and confirm well-formed
output (valid vocab ids); BLEU=100 on a hand-constructed identical
hypothesis/reference pair (Section 6)."""
import torch

from src.evaluation.decode import beam_search_decode, greedy_decode
from src.evaluation.metrics import compute_bleu, compute_rouge
from src.models.sign2_lid_text import Sign2LIDText


def _tiny_model(vocab=30):
    return Sign2LIDText(
        d_feat=16, d_model=16, num_encoder_layers=3, num_decoder_layers=2,
        num_heads=4, ff_size=24, dropout=0.0, lid_split_layer=1,
        use_lid_branch=True, txt_vocab_size=vocab, txt_pad_index=1, lid_vocab_size=6,
    )


def test_greedy_decode_produces_valid_ids():
    model = _tiny_model(vocab=30)
    features = torch.randn(2, 14, 16)
    feature_lengths = torch.tensor([14, 10])
    hyp_ids = greedy_decode(model, features, feature_lengths, bos_id=2, eos_id=3, max_len=8)
    assert len(hyp_ids) == 2
    for seq in hyp_ids:
        assert all(0 <= i < 30 for i in seq)
        assert len(seq) <= 8


def test_beam_search_decode_with_ctc_rescoring_runs():
    model = _tiny_model(vocab=30)
    features = torch.randn(2, 14, 16)
    feature_lengths = torch.tensor([14, 12])
    hyp_ids = beam_search_decode(
        model, features, feature_lengths, bos_id=2, eos_id=3, max_len=6,
        beam_size=3, ctc_weight=0.3,
    )
    assert len(hyp_ids) == 2
    for seq in hyp_ids:
        assert all(0 <= i < 30 for i in seq)


def test_bleu_100_on_identical_hyp_ref():
    hyps = ["the weather is cold today", "hello there friend"]
    refs = ["the weather is cold today", "hello there friend"]
    assert abs(compute_bleu(hyps, refs) - 100.0) < 1e-6


def test_rouge_computes_without_error():
    hyps = ["the weather is cold today"]
    refs = ["the weather is very cold today"]
    score = compute_rouge(hyps, refs)
    assert 0.0 <= score <= 100.0
