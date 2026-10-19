#!/usr/bin/env python
"""Baseline vs. proposed comparison (PLAN_LID_SELF_CONDITIONING.md §5).

Aggregates scripts/evaluate.py outputs over seeds (mean ± std) and runs a
paired bootstrap test (Koehn 2004) on corpus BLEU for each matched seed pair
(same test set, same order). Writes a JSON summary and a Markdown table.

    python scripts/compare_runs.py --split test \
        --baseline experiments/baseline/seed{1,2,3} \
        --proposed experiments/proposed/seed{1,2,3} \
        --output experiments/results/comparison_test
"""
import argparse
import csv
import json
import os
import statistics

import numpy as np
import sacrebleu


def load_run(model_dir: str, split: str):
    eval_dir = os.path.join(model_dir, f"eval_{split}")
    with open(os.path.join(eval_dir, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    with open(os.path.join(eval_dir, "hypotheses.tsv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return metrics, rows


def paired_bootstrap(base_hyps, prop_hyps, refs, n_samples: int, seed: int) -> dict:
    """p = fraction of resampled test sets on which proposed BLEU <= baseline."""
    rng = np.random.default_rng(seed)
    n = len(refs)
    deltas = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, n)
        r = [refs[i] for i in idx]
        b = sacrebleu.corpus_bleu([base_hyps[i] for i in idx], [r]).score
        p = sacrebleu.corpus_bleu([prop_hyps[i] for i in idx], [r]).score
        deltas.append(p - b)
    deltas = np.array(deltas)
    return {
        "delta_mean": float(deltas.mean()),
        "delta_95ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_value": float((deltas <= 0).mean()),
    }


def mean_std(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "values": values}


def summarise(runs):
    metrics = [m for m, _ in runs]
    out = {k: mean_std([m[k] for m in metrics]) for k in ("bleu", "rouge_l") }
    for k in ("lid_utterance_acc", "selfcond_gate", "trainable_params"):
        if all(k in m for m in metrics):
            out[k] = mean_std([m[k] for m in metrics])
    langs = sorted(metrics[0]["per_language"])
    out["per_language_bleu"] = {l: mean_std([m["per_language"][l]["bleu"] for m in metrics]) for l in langs}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", nargs="+", required=True, help="baseline model dirs, one per seed")
    parser.add_argument("--proposed", nargs="+", required=True, help="proposed model dirs, same seed order")
    parser.add_argument("--split", default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output", required=True, help="output path prefix (.json and .md are appended)")
    args = parser.parse_args()
    if len(args.baseline) != len(args.proposed):
        parser.error("need the same number of baseline and proposed runs (matched seeds)")

    base = [load_run(d, args.split) for d in args.baseline]
    prop = [load_run(d, args.split) for d in args.proposed]

    tests = []
    for (bm, brows), (pm, prows), bdir, pdir in zip(base, prop, args.baseline, args.proposed):
        if [r["video_id"] for r in brows] != [r["video_id"] for r in prows]:
            raise ValueError(f"{bdir} and {pdir} were evaluated on different examples/order")
        refs = [r["reference"] for r in brows]
        t = paired_bootstrap([r["hypothesis"] for r in brows], [r["hypothesis"] for r in prows],
                             refs, args.bootstrap_samples, seed=0)
        tests.append({"baseline": bdir, "proposed": pdir, **t})

    summary = {"split": args.split, "baseline": summarise(base), "proposed": summarise(prop), "paired_bootstrap": tests}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fmt = lambda s: f"{s['mean']:.2f} ± {s['std']:.2f}"  # noqa: E731
    b, p = summary["baseline"], summary["proposed"]
    lines = [
        f"## {args.split} — baseline (Sign2(LID+Text)) vs. proposed (+LID self-conditioning), {len(base)} seed(s)",
        "",
        "| Metric | Baseline | Proposed |",
        "|---|---|---|",
        f"| BLEU-4 | {fmt(b['bleu'])} | {fmt(p['bleu'])} |",
        f"| ROUGE-L | {fmt(b['rouge_l'])} | {fmt(p['rouge_l'])} |",
    ]
    if "lid_utterance_acc" in b and "lid_utterance_acc" in p:
        lines.append(f"| LID utterance acc. (%) | {fmt(b['lid_utterance_acc'])} | {fmt(p['lid_utterance_acc'])} |")
    if "selfcond_gate" in p:
        lines.append(f"| Learned gate g | – | {p['selfcond_gate']['mean']:.3f} |")
    lines.append(f"| Trainable params | {b['trainable_params']['mean']:,.0f} | {p['trainable_params']['mean']:,.0f} |")
    lines += ["", "| Lang | Baseline BLEU | Proposed BLEU | Δ |", "|---|---|---|---|"]
    for lang, bs in b["per_language_bleu"].items():
        ps = p["per_language_bleu"][lang]
        lines.append(f"| {lang} | {fmt(bs)} | {fmt(ps)} | {ps['mean'] - bs['mean']:+.2f} |")
    lines += ["", "Paired bootstrap (proposed − baseline BLEU):", ""]
    for t in tests:
        lines.append(f"- {os.path.basename(t['proposed'])}: Δ={t['delta_mean']:+.2f}, "
                     f"95% CI [{t['delta_95ci'][0]:+.2f}, {t['delta_95ci'][1]:+.2f}], p={t['p_value']:.3f}")
    with open(args.output + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
