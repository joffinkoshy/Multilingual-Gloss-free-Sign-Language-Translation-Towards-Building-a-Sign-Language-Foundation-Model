"""Proposed novelty (LID self-conditioning) + the pipeline pieces added with
it: config wiring, many-to-one language filter, cached-feature reader,
DINOv2 extraction plumbing, gradient accumulation."""
import copy
import os

import pytest
import torch

from src.datasets.feature_cache import CachedFeatureReader, feature_key
from src.models.build import build_model
from src.models.lid_self_conditioning import LIDSelfConditioner
from src.models.sign2_lid_text import Sign2LIDText
from src.training.setup import language_filter
from src.utils.config import ConfigError, load_config, validate_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LID_V = 14  # 4 specials + 10 SP-10 languages (LIDTokenizer)


def tiny_model(lid_conditioner=None, use_lid_branch=True):
    return Sign2LIDText(
        d_feat=24, d_model=16, num_encoder_layers=3, num_decoder_layers=2,
        num_heads=4, ff_size=32, dropout=0.0, lid_split_layer=1,
        use_lid_branch=use_lid_branch, txt_vocab_size=30, txt_pad_index=1,
        lid_vocab_size=LID_V if use_lid_branch else None, lid_conditioner=lid_conditioner,
    )


def tiny_inputs(bsz=3, n_frames=10, n_text=4):
    torch.manual_seed(0)
    return dict(
        features=torch.randn(bsz, n_frames, 24),
        src_key_padding_mask=torch.zeros(bsz, n_frames, dtype=torch.bool),
        tgt_input_ids=torch.randint(4, 30, (bsz, n_text)),
        tgt_mask=torch.nn.Transformer.generate_square_subsequent_mask(n_text),
        tgt_key_padding_mask=torch.zeros(bsz, n_text, dtype=torch.bool),
    )


def test_conditioner_adds_gated_projection_of_posterior():
    cond = LIDSelfConditioner(LID_V, 16, gate_init=0.0)
    h = torch.randn(2, 5, 16)
    log_probs = torch.log_softmax(torch.randn(2, 5, LID_V), dim=-1)
    out = cond(h, log_probs)
    expected = h + 0.5 * cond.proj(log_probs.exp())
    assert out.shape == h.shape
    assert torch.allclose(out, expected, atol=1e-6)
    assert sum(p.numel() for p in cond.parameters()) == LID_V * 16 + 1


def test_closed_gate_reproduces_paper_model_exactly():
    cond = LIDSelfConditioner(LID_V, 16, gate_init=-1e4, learn_gate=False)  # sigmoid -> 0
    proposed = tiny_model(lid_conditioner=cond).eval()
    baseline = tiny_model().eval()
    state = {k: v for k, v in proposed.state_dict().items() if not k.startswith("lid_conditioner.")}
    baseline.load_state_dict(state)
    inputs = tiny_inputs()
    out_p, out_b = proposed(**inputs), baseline(**inputs)
    for key in ("attn_log_probs", "txt_ctc_log_probs", "lid_log_probs", "h_fin"):
        assert torch.allclose(out_p[key], out_b[key], atol=1e-6), key


def test_open_gate_changes_output():
    cond = LIDSelfConditioner(LID_V, 16, gate_init=2.0)
    proposed = tiny_model(lid_conditioner=cond).eval()
    baseline = tiny_model().eval()
    baseline.load_state_dict({k: v for k, v in proposed.state_dict().items() if not k.startswith("lid_conditioner.")})
    inputs = tiny_inputs()
    assert not torch.allclose(proposed(**inputs)["h_fin"], baseline(**inputs)["h_fin"])


@pytest.mark.parametrize("with_conditioner", [False, True])
def test_translation_loss_reaches_lid_head_only_with_conditioning(with_conditioner):
    cond = LIDSelfConditioner(LID_V, 16, gate_init=0.0) if with_conditioner else None
    model = tiny_model(lid_conditioner=cond)
    out = model(**tiny_inputs())
    out["attn_log_probs"].sum().backward()  # L_Attn only, no L_LID term
    grad = model.lid_head.proj.weight.grad
    if with_conditioner:
        assert grad is not None and torch.any(grad != 0)
        assert cond.gate_logit.grad is not None and cond.gate_logit.grad != 0
    else:
        assert grad is None or torch.all(grad == 0)


def test_conditioner_requires_lid_branch():
    with pytest.raises(ValueError):
        tiny_model(lid_conditioner=LIDSelfConditioner(LID_V, 16), use_lid_branch=False)


def test_configs_wire_the_conditioner():
    baseline_cfg = load_config(os.path.join(REPO, "configs", "sp10_m2o_dinov2.yaml"))
    proposed_cfg = load_config(os.path.join(REPO, "configs", "proposed_lid_selfcond.yaml"))
    assert not baseline_cfg["lid_self_conditioning"]["enabled"]
    assert proposed_cfg["lid_self_conditioning"]["enabled"]
    # the two configs differ only in the novelty switch
    stripped = copy.deepcopy(proposed_cfg)
    stripped["lid_self_conditioning"]["enabled"] = False
    assert stripped == baseline_cfg

    for cfg, has in ((baseline_cfg, False), (proposed_cfg, True)):
        cfg = copy.deepcopy(cfg)
        cfg["model"].update(d_model=16, ff_size=32, num_heads=4, num_encoder_layers=2, num_decoder_layers=1)
        model, _ = build_model(cfg, txt_vocab_size=30, txt_pad_index=1, lid_vocab_size=LID_V)
        assert (model.lid_conditioner is not None) == has

    bad = copy.deepcopy(proposed_cfg)
    bad["model"].update(variant="baseline", use_lid_branch=False)
    bad["loss"]["lambda_lid"] = 0.0
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_many_to_one_filter_keeps_only_english_targets():
    cfg = load_config(os.path.join(REPO, "configs", "sp10_m2o_dinov2.yaml"))
    pairs = language_filter(cfg)
    assert len(pairs) == 10 and all(tgt == "en" for _, tgt in pairs)

    split_path = os.path.join(REPO, "data", "splits", "sp10", "train.json")
    if not os.path.exists(split_path):
        pytest.skip("real SP-10 index not present")
    from src.datasets.common import load_index
    examples = [e for e in load_index(split_path) if (e.src_lang, e.tgt_lang) in set(pairs)]
    assert len(examples) == 8300  # paper Appendix C: 10 x 830


def test_feature_key_shares_video_across_targets():
    assert feature_key("410_zh_to_en") == feature_key("410_zh") == "410_zh"
    assert feature_key("410_en") == "410_en"


def test_cached_feature_reader(tmp_path):
    torch.save(torch.randn(7, 24).half(), tmp_path / "410_zh.pt")
    reader = CachedFeatureReader(str(tmp_path), d_feat=24)
    feats = reader("410_zh_to_en", n_text_tokens=2)
    assert feats.shape == (7, 24) and feats.dtype == torch.float32
    with pytest.raises(FileNotFoundError):
        reader("999_de", 2)
    with pytest.raises(ValueError):
        CachedFeatureReader(str(tmp_path), d_feat=32)("410_zh", 2)
    with pytest.raises(FileNotFoundError):
        CachedFeatureReader(str(tmp_path / "nope"), d_feat=24)


def test_dinov2_extraction_plumbing(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    import importlib.util

    spec = importlib.util.spec_from_file_location("extract", os.path.join(REPO, "scripts", "extract_features_dinov2.py"))
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)

    video = tmp_path / "videos" / "train" / "410" / "zh.mp4"
    video.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()

    frames = extract.read_frames(str(video), stride=2, size=32)
    assert frames.shape == (5, 3, 32, 32)

    class DummyEncoder(torch.nn.Module):  # stands in for DINOv2: images -> [B, d]
        def forward(self, x):
            return x.mean(dim=(2, 3)).repeat(1, 8)  # [B, 24]

    videos = list(extract.find_videos(str(tmp_path / "videos"), ["train"]))
    assert [(s, k) for s, k, _ in videos] == [("train", "410_zh")]
    out_dir = tmp_path / "features"
    stats = extract.extract_all(videos, DummyEncoder(), str(out_dir), "cpu", stride=2, size=32, batch_size=2)
    assert stats["written"] == 1
    feats = CachedFeatureReader(str(out_dir), d_feat=24)("410_zh_to_en", 1)
    assert feats.shape == (5, 24)
    # second pass skips existing files
    assert extract.extract_all(videos, DummyEncoder(), str(out_dir), "cpu", 2, 32, 2)["skipped_existing"] == 1


def test_gradient_accumulation_steps_optimizer_every_k_batches(tmp_path, tokenizer, lid_tokenizer):
    from src.datasets.common import SLTDataset, collate_slt_batch
    from src.training.trainer import Trainer
    from tests.conftest import make_index

    cfg = load_config(os.path.join(REPO, "configs", "proposed_lid_selfcond.yaml"))
    cfg["model"].update(d_feat=24, d_model=16, ff_size=32, num_heads=4, num_encoder_layers=2, num_decoder_layers=1)
    cfg["training"].update(grad_accum_steps=2, batch_size=2)
    model, _ = build_model(cfg, len(tokenizer), tokenizer.pad_id, len(lid_tokenizer))
    trainer = Trainer(cfg, model, tokenizer, lid_tokenizer, None, str(tmp_path / "run"), "cpu")

    ds = SLTDataset(make_index(tmp_path, "sp10", "train", n=6), tokenizer, d_feat=24)
    batches = [collate_slt_batch([ds[i], ds[i + 1]], pad_id=tokenizer.pad_id) for i in (0, 2, 4)]
    for b in batches:
        trainer.training_step(b)
    assert trainer.step == 1          # 3 micro-batches, accumulation 2 -> 1 step so far
    trainer._flush_accumulated_grads()
    assert trainer.step == 2          # leftover micro-batch applied at epoch end
