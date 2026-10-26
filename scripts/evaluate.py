#!/usr/bin/env python
"""Evaluate a trained checkpoint on dev/test (PLAN_LID_SELF_CONDITIONING.md §5).

Writes to <output-dir> (default: <model-dir>/eval_<split>/):
  metrics.json     BLEU-4 (SacreBLEU) + ROUGE-L overall and per source sign
                   language, LID frame/utterance accuracy, self-conditioning
                   gate, parameter count
  hypotheses.tsv   video_id, src_lang, reference, hypothesis (input to
                   scripts/compare_runs.py's paired bootstrap)

    python scripts/evaluate.py --config configs/proposed_lid_selfcond.yaml \
        --model-dir experiments/proposed/seed1 --split test --device cuda
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# nn.CTCLoss has no MPS kernel; let it run on CPU when --device mps (must be set before importing torch)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sacrebleu
import torch
from torch.utils.data import DataLoader

from src.datasets.common import collate_slt_batch
from src.evaluation.decode import decode_loader
from src.evaluation.metrics import compute_bleu, compute_rouge
from src.models.build import build_model
from src.models.layers import key_padding_mask_from_lengths
from src.training.checkpoint import load_checkpoint
from src.training.setup import TXT_VOCAB_FILE, build_feature_fn, build_split, build_tokenizers, language_filter
from src.utils.config import load_config


@torch.no_grad()
def lid_accuracy(model, loader, lid_tokenizer, device: str) -> dict:
    """Accuracy of the LID-CTC head. Frame-level: over frames whose argmax is
    not the blank (index 0). Utterance-level: majority vote of those frames."""
    model.eval()
    frame_correct = frame_total = utt_correct = utt_total = 0
    for batch in loader:
        feats = batch["features"].to(device)
        lengths = batch["feature_lengths"].to(device)
        _, lid_log_probs = model.encode(feats, key_padding_mask_from_lengths(lengths, feats.size(1)))
        preds = lid_log_probs.argmax(dim=-1).cpu()
        for i, lang in enumerate(batch["src_langs"]):
            gold = lid_tokenizer.token_for(lang)
            p = preds[i, : int(lengths[i])]
            p = p[p != 0]
            frame_correct += int((p == gold).sum())
            frame_total += int(p.numel())
            utt_total += 1
            if p.numel() and Counter(p.tolist()).most_common(1)[0][0] == gold:
                utt_correct += 1
    return {
        "lid_frame_acc": 100.0 * frame_correct / max(frame_total, 1),
        "lid_utterance_acc": 100.0 * utt_correct / max(utt_total, 1),
        "lid_nonblank_frames": frame_total,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True, help="training output dir (holds best.ckpt and txt_vocab.txt)")
    parser.add_argument("--checkpoint", default=None, help="default: <model-dir>/best.ckpt")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-stub-features", action="store_true")
    parser.add_argument("--strategy", choices=["greedy", "beam"], default=None,
                        help="decoding; default: config decode.final_strategy (else beam)")
    parser.add_argument("--ctc-weight", type=float, default=None, help="override config decode.ctc_weight (beam only)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.ctc_weight is not None:
        config["decode"]["ctc_weight"] = args.ctc_weight
    strategy = args.strategy or config["decode"].get("final_strategy", "beam")
    splits_dir = args.splits_dir or config["data"]["splits_dir"]
    checkpoint = args.checkpoint or os.path.join(args.model_dir, "best.ckpt")
    output_dir = args.output_dir or os.path.join(args.model_dir, f"eval_{args.split}")
    os.makedirs(output_dir, exist_ok=True)

    pairs = language_filter(config)
    tokenizer, lid_tokenizer = build_tokenizers(
        config, splits_dir, pairs, vocab_path=os.path.join(args.model_dir, TXT_VOCAB_FILE)
    )
    feature_fn = build_feature_fn(config, allow_stub=args.allow_stub_features)
    ds = build_split(config, args.split, splits_dir, tokenizer, pairs, feature_fn)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate_slt_batch(b, pad_id=tokenizer.pad_id))

    model, family_router = build_model(config, len(tokenizer), tokenizer.pad_id,
                                       len(lid_tokenizer) if lid_tokenizer else None)
    state = load_checkpoint(checkpoint, model)
    model.to(args.device).eval()

    hyps, refs, src_langs, video_ids = decode_loader(model, loader, tokenizer, config, args.device, family_router, strategy=strategy)

    by_lang = defaultdict(lambda: ([], []))
    for h, r, lang in zip(hyps, refs, src_langs):
        by_lang[lang][0].append(h)
        by_lang[lang][1].append(r)

    bleu_metric = sacrebleu.BLEU()
    bleu = bleu_metric.corpus_score(hyps, [refs])
    metrics = {
        "split": args.split,
        "decoding": strategy if strategy == "greedy" else f"beam{config['decode']['beam_size']}+ctc{config['decode']['ctc_weight']}",
        "checkpoint": checkpoint,
        "best_epoch": state.get("epoch"),
        "best_dev_bleu_during_training": state.get("best_score"),
        "n": len(hyps),
        "bleu": bleu.score,
        "rouge_l": compute_rouge(hyps, refs),
        "bleu_signature": str(bleu_metric.get_signature()),
        "per_language": {
            lang: {"n": len(h), "bleu": compute_bleu(h, r), "rouge_l": compute_rouge(h, r)}
            for lang, (h, r) in sorted(by_lang.items())
        },
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if getattr(model, "use_lid_branch", False):
        metrics.update(lid_accuracy(model, loader, lid_tokenizer, args.device))
    conditioner = getattr(model, "lid_conditioner", None)
    if conditioner is not None:
        metrics["selfcond_gate"] = conditioner.gate.item()
        metrics["selfcond_params"] = sum(p.numel() for p in conditioner.parameters())

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "hypotheses.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["video_id", "src_lang", "reference", "hypothesis"])
        w.writerows(zip(video_ids, src_langs, refs, hyps))

    print(json.dumps({k: v for k, v in metrics.items() if k != "per_language"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
