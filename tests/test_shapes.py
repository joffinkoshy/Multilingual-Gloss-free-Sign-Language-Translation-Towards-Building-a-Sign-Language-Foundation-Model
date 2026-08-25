"""Stage 2/3 verification: tokenizer round-trip, dataset batch shapes,
component-level forward-pass shapes (IMPLEMENTATION_HANDOFF.md Section 6)."""
import torch

from src.datasets.common import SLTDataset, collate_slt_batch
from src.models.layers import InputProjection, build_decoder, build_encoder_layer
from src.models.sign2_lid_text import Sign2LIDText
from tests.conftest import make_index


def test_tokenizer_round_trip(tokenizer):
    text = "the weather tomorrow will be cold and windy"
    ids = tokenizer.encode(text, add_eos=True)
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_dataset_batch_shapes(tmp_path, tokenizer):
    d_feat = 32
    for dataset in ("sp10", "phoenix14t", "csl_daily"):
        index_path = make_index(tmp_path, dataset, "train", n=6)
        ds = SLTDataset(index_path, tokenizer, d_feat=d_feat)
        batch = [ds[i] for i in range(len(ds))]
        collated = collate_slt_batch(batch, pad_id=tokenizer.pad_id)

        assert collated["features"].shape == (6, collated["features"].shape[1], d_feat)
        assert collated["text_ids"].shape[0] == 6
        assert not torch.isnan(collated["features"]).any()

        for i in range(6):
            n_frames = int(collated["feature_lengths"][i])
            n_text = int(collated["text_lengths"][i])
            assert n_frames > n_text, f"{dataset}: expected |F| > |T|, got {n_frames} <= {n_text}"


def test_input_projection_shape():
    proj = InputProjection(d_feat=32, d_model=16, dropout=0.0)
    x = torch.randn(4, 10, 32)
    out = proj(x)
    assert out.shape == (4, 10, 16)


def test_encoder_layer_shape():
    layer = build_encoder_layer(d_model=16, num_heads=4, ff_size=32, dropout=0.0)
    x = torch.randn(4, 10, 16)
    out = layer(x)
    assert out.shape == x.shape


def test_decoder_shape():
    decoder = build_decoder(d_model=16, num_heads=4, ff_size=32, dropout=0.0, num_layers=2)
    tgt = torch.randn(4, 5, 16)
    memory = torch.randn(4, 10, 16)
    out = decoder(tgt=tgt, memory=memory)
    assert out.shape == tgt.shape


def test_sign2_lid_text_output_shapes():
    model = Sign2LIDText(
        d_feat=32, d_model=16, num_encoder_layers=6, num_decoder_layers=2,
        num_heads=4, ff_size=32, dropout=0.0, lid_split_layer=1,
        use_lid_branch=True, txt_vocab_size=50, txt_pad_index=1, lid_vocab_size=14,
    )
    features = torch.randn(3, 12, 32)
    tgt_ids = torch.randint(0, 50, (3, 5))
    out = model(
        features=features,
        src_key_padding_mask=torch.zeros(3, 12, dtype=torch.bool),
        tgt_input_ids=tgt_ids,
        tgt_mask=torch.nn.Transformer.generate_square_subsequent_mask(5),
        tgt_key_padding_mask=torch.zeros(3, 5, dtype=torch.bool),
    )
    assert out["lid_log_probs"].shape == (3, 12, 14)
    assert out["txt_ctc_log_probs"].shape == (3, 12, 50)
    assert out["attn_log_probs"].shape == (3, 5, 50)


def test_one_to_one_skips_lid_branch_entirely():
    model = Sign2LIDText(
        d_feat=32, d_model=16, num_encoder_layers=6, num_decoder_layers=2,
        num_heads=4, ff_size=32, dropout=0.0, lid_split_layer=1,
        use_lid_branch=False, txt_vocab_size=50, txt_pad_index=1,
    )
    assert model.lid_head is None
    assert len(model.enc_layers_2) == 0
    features = torch.randn(2, 8, 32)
    out = model(
        features=features,
        src_key_padding_mask=torch.zeros(2, 8, dtype=torch.bool),
        tgt_input_ids=torch.randint(0, 50, (2, 4)),
        tgt_mask=torch.nn.Transformer.generate_square_subsequent_mask(4),
        tgt_key_padding_mask=torch.zeros(2, 4, dtype=torch.bool),
    )
    assert out["lid_log_probs"] is None
