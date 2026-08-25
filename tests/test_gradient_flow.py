"""Stage 3/4 verification: after one backward pass, every parameter has a
non-None, non-all-zero gradient — except the LID branch specifically when
lambda1=0 (one_to_one), where it must be absent entirely (Section 6)."""
import torch

from src.losses.joint_loss import JointLoss
from src.models.baseline_transformer import BaselineTransformer
from src.models.sign2_lid_text import Sign2LIDText


def _dummy_batch(bsz=3, n_frames=12, n_text=5, d_feat=32, vocab=40):
    features = torch.randn(bsz, n_frames, d_feat)
    src_kpm = torch.zeros(bsz, n_frames, dtype=torch.bool)
    text_ids = torch.randint(2, vocab, (bsz, n_text))
    tgt_input = torch.randint(2, vocab, (bsz, n_text))
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(n_text)
    tgt_kpm = torch.zeros(bsz, n_text, dtype=torch.bool)
    feature_lengths = torch.full((bsz,), n_frames, dtype=torch.long)
    text_lengths = torch.full((bsz,), n_text, dtype=torch.long)
    return features, src_kpm, text_ids, tgt_input, tgt_mask, tgt_kpm, feature_lengths, text_lengths


def test_baseline_gradient_flow():
    features, src_kpm, text_ids, tgt_input, tgt_mask, tgt_kpm, _, _ = _dummy_batch()
    model = BaselineTransformer(
        d_feat=32, d_model=16, num_encoder_layers=2, num_decoder_layers=2,
        num_heads=4, ff_size=32, dropout=0.0, txt_vocab_size=40, txt_pad_index=1,
    )
    log_probs = model(features, src_kpm, tgt_input, tgt_mask, tgt_kpm)
    loss = torch.nn.functional.nll_loss(log_probs.reshape(-1, 40), text_ids.reshape(-1))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.any(p.grad != 0), f"{name} has an all-zero gradient"


def test_sign2_lid_text_gradient_flow_many_to_one():
    features, src_kpm, text_ids, tgt_input, tgt_mask, tgt_kpm, feat_len, text_len = _dummy_batch()
    model = Sign2LIDText(
        d_feat=32, d_model=16, num_encoder_layers=4, num_decoder_layers=2,
        num_heads=4, ff_size=32, dropout=0.0, lid_split_layer=1,
        use_lid_branch=True, txt_vocab_size=40, txt_pad_index=1, lid_vocab_size=8,
    )
    loss_fn = JointLoss(lambda_lid=1.0, lambda_txt=5.0, lambda_attn=3.0, txt_pad_index=1)
    out = model(features, src_kpm, tgt_input, tgt_mask, tgt_kpm)
    lid_ids = torch.randint(1, 8, (3, 5))
    losses = loss_fn(
        lid_log_probs=out["lid_log_probs"], txt_ctc_log_probs=out["txt_ctc_log_probs"],
        attn_log_probs=out["attn_log_probs"], feature_lengths=feat_len,
        text_ids=text_ids, text_lengths=text_len, lid_ids=lid_ids,
    )
    losses["total"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.any(p.grad != 0), f"{name} has an all-zero gradient"


def test_sign2_lid_text_one_to_one_has_no_lid_params_at_all():
    features, src_kpm, text_ids, tgt_input, tgt_mask, tgt_kpm, feat_len, text_len = _dummy_batch()
    model = Sign2LIDText(
        d_feat=32, d_model=16, num_encoder_layers=4, num_decoder_layers=2,
        num_heads=4, ff_size=32, dropout=0.0, lid_split_layer=1,
        use_lid_branch=False, txt_vocab_size=40, txt_pad_index=1,
    )
    assert not any("lid_head" in n for n, _ in model.named_parameters())

    loss_fn = JointLoss(lambda_lid=0.0, lambda_txt=5.0, lambda_attn=3.0, txt_pad_index=1)
    out = model(features, src_kpm, tgt_input, tgt_mask, tgt_kpm)
    losses = loss_fn(
        lid_log_probs=out["lid_log_probs"], txt_ctc_log_probs=out["txt_ctc_log_probs"],
        attn_log_probs=out["attn_log_probs"], feature_lengths=feat_len,
        text_ids=text_ids, text_lengths=text_len, lid_ids=None,
    )
    losses["total"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
