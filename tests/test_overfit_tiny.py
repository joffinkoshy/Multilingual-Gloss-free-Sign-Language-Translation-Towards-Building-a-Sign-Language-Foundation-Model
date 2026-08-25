"""Stage 7 gate: train on ~10-20 examples with no regularization until
near-zero training loss / near-100 BLEU on that same tiny set. Must pass
before any real-scale training run (Section 6). Also covers Stage 5's
resume-reproduces-trajectory check.

Runs on CPU with tiny dims — this is a correctness gate, not a speed
benchmark, so it's allowed to take a few seconds.
"""
import copy

import torch
from torch.utils.data import DataLoader

from src.datasets.common import SLTDataset, collate_slt_batch
from src.evaluation.decode import greedy_decode
from src.evaluation.metrics import compute_bleu, decode_ids_to_text
from src.models.sign2_lid_text import Sign2LIDText
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.trainer import Trainer
from src.utils.seed import set_seed
from tests.conftest import make_index

TINY_CONFIG = {
    "task": "many_to_one",
    "model": {
        "variant": "sign2_lid_text", "d_model": 16, "d_feat": 24,
        "num_encoder_layers": 4, "num_decoder_layers": 2, "num_heads": 4,
        "ff_size": 32, "dropout": 0.0, "lid_split_layer": 1, "use_lid_branch": True,
    },
    "loss": {"lambda_lid": 1.0, "lambda_txt": 5.0, "lambda_attn": 3.0, "label_smoothing": 0.0},
    "optimizer": {"name": "adam", "lr": 3.0e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
    "scheduler": {"name": "constant"},
    "training": {
        "batch_size": 8, "seed": 42, "grad_clip_norm": 5.0,
        "early_stopping_patience": 1000, "max_epochs": 60,
        "checkpoint_every_epochs": 1000, "checkpoint_every_minutes": None,
    },
    "decode": {"beam_size": 1, "ctc_weight": 0.0},
    "data": {"dataset": "sp10", "tokenizer": {"type": "word"}, "feature_cache_dir": "", "splits_dir": ""},
}


def _build_tiny_loader(tmp_path, tokenizer, lid_tokenizer, n=12):
    index_path = make_index(tmp_path, "sp10", "overfit", n=n)
    ds = SLTDataset(index_path, tokenizer, d_feat=TINY_CONFIG["model"]["d_feat"])
    loader = DataLoader(
        ds, batch_size=TINY_CONFIG["training"]["batch_size"], shuffle=True,
        collate_fn=lambda b: collate_slt_batch(b, pad_id=tokenizer.pad_id),
    )
    return ds, loader


def test_overfit_tiny_dataset(tmp_path, tokenizer, lid_tokenizer):
    set_seed(TINY_CONFIG["training"]["seed"])
    ds, loader = _build_tiny_loader(tmp_path, tokenizer, lid_tokenizer)

    model = Sign2LIDText(
        d_feat=TINY_CONFIG["model"]["d_feat"], d_model=TINY_CONFIG["model"]["d_model"],
        num_encoder_layers=TINY_CONFIG["model"]["num_encoder_layers"],
        num_decoder_layers=TINY_CONFIG["model"]["num_decoder_layers"],
        num_heads=TINY_CONFIG["model"]["num_heads"], ff_size=TINY_CONFIG["model"]["ff_size"],
        dropout=0.0, lid_split_layer=1, use_lid_branch=True,
        txt_vocab_size=len(tokenizer), txt_pad_index=tokenizer.pad_id,
        lid_vocab_size=len(lid_tokenizer),
    )
    trainer = Trainer(TINY_CONFIG, model, tokenizer, lid_tokenizer, model_dir=str(tmp_path / "run"))

    losses = []
    for epoch in range(TINY_CONFIG["training"]["max_epochs"]):
        epoch_losses = [trainer.training_step(batch)["total"] for batch in loader]
        losses.append(sum(epoch_losses) / len(epoch_losses))

    assert losses[-1] < losses[0] * 0.3, f"loss did not drop enough: {losses[0]:.3f} -> {losses[-1]:.3f}"

    # near-100 BLEU on the same tiny (train) set it just overfit
    all_batches = list(DataLoader(ds, batch_size=len(ds), collate_fn=lambda b: collate_slt_batch(b, pad_id=tokenizer.pad_id)))
    batch = all_batches[0]
    hyp_ids = greedy_decode(
        model, batch["features"], batch["feature_lengths"],
        bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, max_len=batch["text_ids"].shape[1] + 2,
    )
    hyps = decode_ids_to_text(hyp_ids, tokenizer)
    refs = batch["texts"]
    bleu = compute_bleu(hyps, refs)
    assert bleu > 60.0, f"expected near-100 BLEU after overfitting, got {bleu:.1f}\nhyps={hyps}\nrefs={refs}"


def test_resume_reproduces_loss_trajectory(tmp_path, tokenizer, lid_tokenizer):
    """Section 2.6 / Stage 5: resuming from a checkpoint mid-run reproduces
    the same loss trajectory as an uninterrupted run."""
    set_seed(123)
    ds, loader = _build_tiny_loader(tmp_path, tokenizer, lid_tokenizer, n=8)
    batches = list(loader)

    def make_model():
        return Sign2LIDText(
            d_feat=TINY_CONFIG["model"]["d_feat"], d_model=TINY_CONFIG["model"]["d_model"],
            num_encoder_layers=TINY_CONFIG["model"]["num_encoder_layers"],
            num_decoder_layers=TINY_CONFIG["model"]["num_decoder_layers"],
            num_heads=TINY_CONFIG["model"]["num_heads"], ff_size=TINY_CONFIG["model"]["ff_size"],
            dropout=0.0, lid_split_layer=1, use_lid_branch=True,
            txt_vocab_size=len(tokenizer), txt_pad_index=tokenizer.pad_id,
            lid_vocab_size=len(lid_tokenizer),
        )

    set_seed(7)
    model_full = make_model()
    trainer_full = Trainer(TINY_CONFIG, model_full, tokenizer, lid_tokenizer, model_dir=str(tmp_path / "full"))
    full_losses = [trainer_full.training_step(b)["total"] for b in batches * 3]

    set_seed(7)
    model_a = make_model()
    trainer_a = Trainer(TINY_CONFIG, model_a, tokenizer, lid_tokenizer, model_dir=str(tmp_path / "a"))
    first_half = [trainer_a.training_step(b)["total"] for b in batches]
    ckpt_path = trainer_a.save("mid")

    model_b = make_model()
    trainer_b = Trainer(TINY_CONFIG, model_b, tokenizer, lid_tokenizer, model_dir=str(tmp_path / "b"))
    trainer_b.resume(ckpt_path)
    second_half = [trainer_b.training_step(b) for b in batches * 2]
    resumed_losses = first_half + [l["total"] for l in second_half]

    for a, b in zip(full_losses, resumed_losses):
        assert abs(a - b) < 1e-3, f"resumed trajectory diverged: {a} vs {b}"
