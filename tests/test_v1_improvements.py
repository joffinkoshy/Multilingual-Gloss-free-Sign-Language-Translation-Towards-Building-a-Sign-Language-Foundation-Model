"""V1 improvements composed via config flags, never edited into the
reproduction files (Section 7's separation rule) — verified independently
here rather than only through the sequential ablation ladder."""
import torch

from src.datasets.augment import FeatureAugmenter
from src.models.family_router import FamilyRouter
from src.models.pretrained_encoder import (
    ContrastiveProjectionHead,
    TextEncoderForPretraining,
    info_nce_loss,
)
from src.models.sign2_lid_text import Sign2LIDText


def test_family_router_composes_into_sign2_lid_text():
    lang_to_family = {"csl": "isolate", "gsg": "germanic", "swl": "germanic"}
    router = FamilyRouter(d_model=16, language_to_family=lang_to_family, bottleneck_dim=8)
    model = Sign2LIDText(
        d_feat=24, d_model=16, num_encoder_layers=4, num_decoder_layers=2,
        num_heads=4, ff_size=32, dropout=0.0, lid_split_layer=1,
        use_lid_branch=True, txt_vocab_size=30, txt_pad_index=1, lid_vocab_size=6,
        family_router=router,
    )
    bsz, n_frames, n_text = 3, 10, 4
    src_langs = ["csl", "gsg", "swl"]
    router.set_batch_languages(src_langs)

    features = torch.randn(bsz, n_frames, 24)
    out = model(
        features=features,
        src_key_padding_mask=torch.zeros(bsz, n_frames, dtype=torch.bool),
        tgt_input_ids=torch.randint(2, 30, (bsz, n_text)),
        tgt_mask=torch.nn.Transformer.generate_square_subsequent_mask(n_text),
        tgt_key_padding_mask=torch.zeros(bsz, n_text, dtype=torch.bool),
    )
    assert out["h_fin"].shape == (bsz, n_frames, 16)
    # router params must receive gradient too
    loss = out["attn_log_probs"].sum()
    loss.backward()
    assert router.family_adapters["germanic"].down.weight.grad is not None
    assert torch.any(router.family_adapters["germanic"].down.weight.grad != 0)


def test_feature_augmenter_spec_augment_changes_features_but_keeps_shape():
    aug = FeatureAugmenter(transforms=["spec_augment"], seed=1)
    features = torch.ones(20, 16)
    out = aug(features)
    assert out.shape == features.shape
    assert not torch.equal(out, features)  # something got masked


def test_contrastive_pretraining_head_and_info_nce_loss():
    torch.manual_seed(0)
    head = ContrastiveProjectionHead(d_feat=24, proj_dim=8, hidden_dim=16)
    text_encoder = TextEncoderForPretraining(vocab_size=30, proj_dim=8, hidden_dim=16, pad_index=1)

    features = torch.randn(5, 12, 24)
    feature_lengths = torch.full((5,), 12, dtype=torch.long)
    text_ids = torch.randint(2, 30, (5, 6))

    sign_emb = head.clip_embedding(features, feature_lengths)
    text_emb = text_encoder(text_ids)
    assert sign_emb.shape == (5, 8)
    assert text_emb.shape == (5, 8)

    loss = info_nce_loss(sign_emb, text_emb)
    assert torch.isfinite(loss)
    loss.backward()
    assert head.net[0].weight.grad is not None
