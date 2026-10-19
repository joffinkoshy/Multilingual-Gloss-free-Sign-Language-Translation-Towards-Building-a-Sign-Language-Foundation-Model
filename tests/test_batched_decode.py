"""Batched beam search must reproduce the per-example reference decoder."""
import pytest
import torch

from src.evaluation.decode import beam_search_decode, beam_search_decode_reference, greedy_decode
from src.models.baseline_transformer import BaselineTransformer
from src.models.sign2_lid_text import Sign2LIDText

BOS, EOS, V = 2, 3, 40


def make(kind, seed):
    torch.manual_seed(seed)
    if kind == "baseline":
        return BaselineTransformer(d_feat=24, d_model=16, num_encoder_layers=2, num_decoder_layers=2,
                                   num_heads=4, ff_size=32, dropout=0.0, txt_vocab_size=V, txt_pad_index=1)
    return Sign2LIDText(d_feat=24, d_model=16, num_encoder_layers=3, num_decoder_layers=2, num_heads=4,
                        ff_size=32, dropout=0.0, lid_split_layer=1, use_lid_branch=True,
                        txt_vocab_size=V, txt_pad_index=1, lid_vocab_size=14)


@pytest.mark.parametrize("kind,ctc_weight", [("baseline", 0.0), ("sign2", 0.0), ("sign2", 0.3)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batched_matches_reference(kind, ctc_weight, seed):
    model = make(kind, seed).eval()
    torch.manual_seed(100 + seed)
    feats = torch.randn(6, 20, 24) * 3
    lengths = torch.full((6,), 20)  # equal lengths: no padding, reference CTC length is exact
    kw = dict(bos_id=BOS, eos_id=EOS, max_len=8, beam_size=4, ctc_weight=ctc_weight)
    ref = beam_search_decode_reference(model, feats, lengths, **kw)
    new = beam_search_decode(model, feats, lengths, **kw)
    assert new == ref


def test_padding_does_not_change_result():
    model = make("sign2", 0).eval()
    torch.manual_seed(5)
    a = torch.randn(1, 12, 24)
    alone = beam_search_decode(model, a, torch.tensor([12]), BOS, EOS, 8, 4, 0.3)
    padded = torch.zeros(2, 20, 24)
    padded[0, :12] = a[0]
    padded[1] = torch.randn(20, 24)
    batched = beam_search_decode(model, padded, torch.tensor([12, 20]), BOS, EOS, 8, 4, 0.3)
    assert batched[0] == alone[0]


def test_beam_one_equals_greedy():
    model = make("baseline", 1).eval()
    feats = torch.randn(4, 15, 24)
    lengths = torch.full((4,), 15)
    greedy = greedy_decode(model, feats, lengths, BOS, EOS, 8)
    beam = beam_search_decode(model, feats, lengths, BOS, EOS, 8, beam_size=1, ctc_weight=0.0)
    cut = lambda ids: ids[: ids.index(EOS) + 1] if EOS in ids else ids  # noqa: E731
    assert [cut(g) for g in greedy] == [cut(b) for b in beam]
