#!/usr/bin/env python
"""CLI entry point: train baseline or Sign2(LID+Text) from a config.

Usage:
    python scripts/train.py --config configs/sp10_many_to_one.yaml \
        --splits-dir data/splits --model-dir experiments/baseline/sp10_many_to_one

Real-data prerequisite (see experiments/results/reference_inspection_notes.md):
this needs (a) populated data/splits/<dataset>/{train,dev,test}.json index
files and (b) cached features under config.data.feature_cache_dir (see
scripts/extract_features_dinov2.py). A missing cache is a hard error; pass
--allow-stub-features to smoke-test the pipeline on random features.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# nn.CTCLoss has no MPS kernel; let it run on CPU when --device mps (must be set before importing torch)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import yaml
from torch.utils.data import DataLoader

from src.datasets.common import collate_slt_batch
from src.evaluation.decode import decode_loader
from src.evaluation.metrics import compute_bleu
from src.models.build import build_model
from src.training.setup import TXT_VOCAB_FILE, build_feature_fn, build_split, build_tokenizers, language_filter
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits-dir", default=None, help="overrides config.data.splits_dir")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=None, help="overrides config.training.seed")
    parser.add_argument("--max-epochs", type=int, default=None, help="overrides config.training.max_epochs")
    parser.add_argument("--allow-stub-features", action="store_true",
                        help="use random stub features (pipeline smoke test only)")
    parser.add_argument(
        "--language-pair", default=None, metavar="SRC:TGT",
        help="restrict a one_to_one run to a single (src_lang, tgt_lang) pair, "
             "e.g. csl:zh — required for SP-10 one_to_one, since its index files "
             "hold all 10 pairs (plus cross-to-English augmentation) together; "
             "each pair is trained as its own independent model (Section 2.1).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    splits_dir = args.splits_dir or config["data"]["splits_dir"]
    set_seed(config["training"]["seed"])

    language_pair = tuple(args.language_pair.split(":")) if args.language_pair else None
    if config["task"] == "one_to_one" and config["data"]["dataset"] == "sp10" and language_pair is None:
        raise ValueError(
            "SP-10 one_to_one runs require --language-pair SRC:TGT (e.g. csl:zh) — "
            "its index file holds all 10 pairs together, see scripts/run_one_to_one.sh."
        )

    pairs = language_filter(config, language_pair)
    tokenizer, lid_tokenizer = build_tokenizers(config, splits_dir, pairs)
    feature_fn = build_feature_fn(config, allow_stub=args.allow_stub_features)
    train_ds = build_split(config, "train", splits_dir, tokenizer, pairs, feature_fn)
    dev_ds = build_split(config, "dev", splits_dir, tokenizer, pairs, feature_fn)

    os.makedirs(args.model_dir, exist_ok=True)
    tokenizer.to_file(os.path.join(args.model_dir, TXT_VOCAB_FILE))
    run_info = {"config": args.config, "language_pairs": pairs, "language_pair": language_pair,
                "n_train": len(train_ds), "n_dev": len(dev_ds), "txt_vocab_size": len(tokenizer),
                "stub_features": args.allow_stub_features}
    with open(os.path.join(args.model_dir, "config.resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({**config, "run_info": run_info}, f, sort_keys=False, allow_unicode=True)

    collate = lambda b: collate_slt_batch(b, pad_id=tokenizer.pad_id)  # noqa: E731
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(
        dev_ds, batch_size=config["training"].get("eval_batch_size", config["training"]["batch_size"]),
        shuffle=False, collate_fn=collate,
    )

    lid_vocab_size = len(lid_tokenizer) if lid_tokenizer else None
    model, family_router = build_model(config, len(tokenizer), tokenizer.pad_id, lid_vocab_size)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainer = Trainer(config, model, tokenizer, lid_tokenizer, family_router, args.model_dir, args.device)
    trainer.logger.info("run info: %s", json.dumps({**run_info, "trainable_params": n_params}, ensure_ascii=False))
    if args.resume:
        trainer.resume(args.resume)

    dev_device = config["decode"].get("dev_device") or args.device

    def dev_eval_fn(model) -> float:
        # Greedy decoding is many tiny sequential steps: faster on CPU than MPS
        # on an M1. Params are moved in place, so optimizer state stays valid.
        model.to(dev_device)
        try:
            hyps, refs, _, _ = decode_loader(model, dev_loader, tokenizer, config, dev_device, family_router,
                                             strategy=config["decode"].get("dev_strategy", "beam"))
        finally:
            model.to(args.device)
        return compute_bleu(hyps, refs)

    trainer.fit(train_loader, dev_eval_fn=dev_eval_fn)


if __name__ == "__main__":
    main()
