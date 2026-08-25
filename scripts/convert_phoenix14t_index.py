#!/usr/bin/env python
"""Convert PHOENIX14T's real annotation CSVs (references/phoenix14t_{split}_raw.csv,
pipe-delimited: name|video|start|end|speaker|orth|translation) into this
repo's index schema (data/splits/phoenix14t/{split}.json).

Uses ONLY the `translation` column (spoken German text) as the target text.
`orth` is the gloss annotation — this reproduction is gloss-free by design
(Section 2.1: "No gloss annotations are used anywhere"), so `orth` is read
only for optional bookkeeping, never fed to the model as supervision.

The CSVs here are the small, standalone annotation files (a Hugging Face
mirror separates them from the ~39GB video archive), not the official RWTH
tarball — sourced from
https://huggingface.co/datasets/quagrain/RWTH-Phoenix-2014T. Actual video
frames still need to come from the official RWTH release or an equivalent
mirror; this only gets the text/label side working ahead of that.
"""
import argparse
import csv
import json
import os


def convert(raw_path: str) -> list:
    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            records.append({
                "video_id": row["name"],
                "src_lang": "gsg",
                "tgt_lang": "de",
                "text": row["translation"],
                "video_url": row["video"],  # relative path within the official archive, not a direct URL
            })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="references")
    parser.add_argument("--out-dir", default="data/splits/phoenix14t")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ("train", "dev", "test"):
        raw_path = os.path.join(args.raw_dir, f"phoenix14t_{split}_raw.csv")
        records = convert(raw_path)
        out_path = os.path.join(args.out_dir, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        print(f"{split}: {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
