#!/usr/bin/env python
"""CLI entry point: train baseline or Sign2(LID+Text) from a config.

Usage:
    python scripts/train.py --config configs/sp10_many_to_one.yaml \
        --splits-dir data/splits --model-dir experiments/baseline/sp10_many_to_one

Real-data prerequisite (see experiments/results/reference_inspection_notes.md):
this needs (a) populated data/splits/<dataset>/{train,dev,test}.json index
files and (b) either real cached SlowFastSign features under
data/features/, or config.data.feature_cache_dir pointed at them. Neither
exists in this repo yet — see Section 12 item 10's findings. Without them,
this script falls back to StubFeatureGenerator, which is only useful for
pipeline smoke-testing (matches what tests/test_overfit_tiny.py already
verifies), not real numbers.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.datasets import csl_daily, phoenix14t, sp10
from src.datasets.common import collate_slt_batch, load_index
from src.datasets.tokenizer import LIDTokenizer, WordTokenizer
from src.evaluation.decode import beam_search_decode
from src.evaluation.metrics import compute_bleu, decode_ids_to_text
from src.models.build import build_model
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed

DATASET_MODULES = {"sp10": sp10, "phoenix14t": phoenix14t, "csl_daily": csl_daily}
ALL_SP10_LANGUAGES = ["csl", "ukl", "rsl", "icl", "gsg", "ise", "bqn", "swl", "lls", "bfi"]


def build_tokenizers(config: dict, splits_dir: str):
    dataset = config["data"]["dataset"]
    train_index_path = os.path.join(splits_dir, dataset, "train.json")
    examples = load_index(train_index_path)
    tok_cfg = config["data"]["tokenizer"]
    tokenizer = WordTokenizer(min_freq=tok_cfg["min_freq"], max_size=tok_cfg["max_size"])
    tokenizer.build([e.text for e in examples])

    lid_tokenizer = None
    if config["model"]["use_lid_branch"]:
        languages = config["data"].get("source_languages", ALL_SP10_LANGUAGES)
        lid_tokenizer = LIDTokenizer(languages)
    return tokenizer, lid_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits-dir", default=None, help="overrides config.data.splits_dir")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    splits_dir = args.splits_dir or config["data"]["splits_dir"]
    set_seed(config["training"]["seed"])

    tokenizer, lid_tokenizer = build_tokenizers(config, splits_dir)
    dataset_module = DATASET_MODULES[config["data"]["dataset"]]

    train_ds = dataset_module.build("train", splits_dir, tokenizer, d_feat=config["model"]["d_feat"])
    dev_ds = dataset_module.build("dev", splits_dir, tokenizer, d_feat=config["model"]["d_feat"])

    train_loader = DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], shuffle=True,
        collate_fn=lambda b: collate_slt_batch(b, pad_id=tokenizer.pad_id),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=config["training"].get("eval_batch_size", config["training"]["batch_size"]),
        shuffle=False, collate_fn=lambda b: collate_slt_batch(b, pad_id=tokenizer.pad_id),
    )

    lid_vocab_size = len(lid_tokenizer) if lid_tokenizer else None
    model, family_router = build_model(config, len(tokenizer), tokenizer.pad_id, lid_vocab_size)

    trainer = Trainer(config, model, tokenizer, lid_tokenizer, family_router, args.model_dir, args.device)
    if args.resume:
        trainer.resume(args.resume)

    def dev_eval_fn(model) -> float:
        hyps, refs = [], []
        for batch in dev_loader:
            if family_router is not None:
                family_router.set_batch_languages(batch["src_langs"])
            hyp_ids = beam_search_decode(
                model, batch["features"].to(args.device), batch["feature_lengths"].to(args.device),
                bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
                max_len=batch["text_ids"].shape[1] + 10,
                beam_size=config["decode"]["beam_size"], ctc_weight=config["decode"]["ctc_weight"],
            )
            hyps.extend(decode_ids_to_text(hyp_ids, tokenizer))
            refs.extend(batch["texts"])
        return compute_bleu(hyps, refs)

    trainer.fit(train_loader, dev_eval_fn=dev_eval_fn)


if __name__ == "__main__":
    main()
