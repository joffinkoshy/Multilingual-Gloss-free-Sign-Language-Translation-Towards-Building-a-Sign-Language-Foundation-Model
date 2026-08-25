"""Section 6: loss at initialization should be finite and roughly
-log(1/|vocab|) scale (untrained model ~= uniform distribution over vocab).
A much smaller initial loss suggests label leakage (e.g. decoder seeing the
target it's predicting)."""
import math

import torch

from src.losses.attention_loss import AttentionCELoss
from src.models.sign2_lid_text import Sign2LIDText


def test_initial_attention_loss_is_near_uniform_scale():
    torch.manual_seed(0)
    vocab = 50
    pad_index = 1
    model = Sign2LIDText(
        d_feat=24, d_model=32, num_encoder_layers=4, num_decoder_layers=2,
        num_heads=4, ff_size=64, dropout=0.0, lid_split_layer=1,
        use_lid_branch=False, txt_vocab_size=vocab, txt_pad_index=pad_index,
    )
    model.eval()

    bsz, n_frames, n_text = 4, 20, 6
    features = torch.randn(bsz, n_frames, 24)
    tgt_input = torch.randint(2, vocab, (bsz, n_text))
    text_ids = torch.randint(2, vocab, (bsz, n_text))

    with torch.no_grad():
        out = model(
            features=features,
            src_key_padding_mask=torch.zeros(bsz, n_frames, dtype=torch.bool),
            tgt_input_ids=tgt_input,
            tgt_mask=torch.nn.Transformer.generate_square_subsequent_mask(n_text),
            tgt_key_padding_mask=torch.zeros(bsz, n_text, dtype=torch.bool),
        )

    loss_fn = AttentionCELoss(pad_index=pad_index, label_smoothing=0.0)
    per_token_loss = loss_fn(out["attn_log_probs"], text_ids) / (bsz * n_text)

    expected = math.log(vocab)
    # generous band: a freshly Xavier-initialized model isn't exactly uniform,
    # but should be within an order of magnitude of -log(1/vocab), not near 0.
    assert expected * 0.3 < per_token_loss.item() < expected * 3.0, (
        f"initial per-token loss {per_token_loss.item():.3f} is far from the "
        f"expected near-uniform scale {expected:.3f} — check for label leakage."
    )
