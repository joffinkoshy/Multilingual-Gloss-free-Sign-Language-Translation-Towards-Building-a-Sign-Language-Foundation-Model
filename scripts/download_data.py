#!/usr/bin/env python
"""Dataset acquisition entry point (Task 6, Section 10).

NOT runnable end-to-end from this sandboxed environment: SP-10 (SpreadTheSign),
PHOENIX14T, and CSL-Daily are not fetchable via a single script call — each
has its own access process, and this session has no dataset-hosting network
access or storage for multi-GB video corpora. This script documents exactly
what to run on the target (RTX 3050) machine and validates the result once
downloaded, rather than performing the download itself.

Sources (verified by directly fetching each project's own page/repo):
  - SP-10: fully open, no request needed. https://github.com/MLSLT/SP-10 —
    clone it, `pip install -r requirements.txt` + ffmpeg, then
    `python video_downloader.py` downloads every clip from
    media.spreadthesign.com (URLs are public). Text/label annotations
    (dataset/{train,dev,test}.json) are already converted into this repo's
    schema by scripts/convert_sp10_index.py -> data/splits/sp10/*.json
    (830/142/214 groups x up to 10 languages = 8,300/1,420/2,024 records,
    matching Section 2.1's stated 8,300/1,420 train/dev sizes almost exactly;
    test is 2,024 vs the paper's 2,021 — a handful of groups are missing one
    language's video). Only the actual video files still need downloading.
  - PHOENIX14T: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/
    — direct download (39GB tarball), no request form found; cite Camgoz et
    al. 2018. Text annotations alone (small, standalone CSVs, independent of
    the video archive) are separately mirrored at
    https://huggingface.co/datasets/quagrain/RWTH-Phoenix-2014T and already
    converted by scripts/convert_phoenix14t_index.py -> real
    data/splits/phoenix14t/{train,dev,test}.json (7,096/519/642 records,
    exact match to Section 2.1). Only the actual video frames still need the
    official RWTH release or an equivalent mirror.
  - CSL-Daily: https://ustc-slr.github.io/datasets/2021_csl_daily/ — requires
    a signed release agreement (must be signed by a full-time staff member,
    not a student) emailed to ustc_vslrg@126.com, CC'd to the dataset's PI;
    cite Zhou et al. 2021 (CVPR, "Improving Sign Language Translation with
    Monolingual Data by Sign Back-Translation"). DEFERRED for now per user
    decision (2026-08-26) — no readily available staff signatory. Per
    Section 9's own dataset-priority order this is "least central to the
    multilingual story," so work proceeds on SP-10 + PHOENIX14T first.

Once real video is available, follow SlowFastSign's own data-prep instructions
(https://github.com/hulianyuyy/CorrNet, linked from kaistmm/SlowFastSign's
README) for frame extraction/resizing before feature extraction
(scripts/extract_features.py) — video preprocessing is treated as SlowFastSign's
frozen, external responsibility (Section 2.2), not reimplemented here.
"""
import argparse
import json
import os
import sys


def validate_index(dataset: str, splits_dir: str) -> None:
    for split in ("train", "dev", "test"):
        path = os.path.join(splits_dir, dataset, f"{split}.json")
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        required_keys = {"video_id", "src_lang", "tgt_lang", "text"}
        bad = [r for r in records if not required_keys.issubset(r)]
        print(f"{dataset}/{split}: {len(records)} records, {len(bad)} malformed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="check existing index files instead of printing instructions")
    parser.add_argument("--splits-dir", default="data/splits")
    args = parser.parse_args()

    if args.validate_only:
        for dataset in ("sp10", "phoenix14t", "csl_daily"):
            validate_index(dataset, args.splits_dir)
        return

    print(__doc__)
    print(
        "\nOnce raw video + transcripts are obtained, build index files at "
        f"{args.splits_dir}/<dataset>/{{train,dev,test}}.json — each a JSON "
        "list of {video_id, src_lang, tgt_lang, text} records matching "
        "src/datasets/common.py's SLTExample schema. Then run this script "
        "again with --validate-only to sanity-check the result."
    )


if __name__ == "__main__":
    sys.exit(main())
