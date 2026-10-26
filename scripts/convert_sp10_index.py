#!/usr/bin/env python
"""Convert MLSLT/SP-10's raw annotation JSON (references/sp10_{split}_raw.json,
fetched from https://github.com/MLSLT/SP-10) into this repo's index schema
(data/splits/sp10/{split}.json).

SP-10's real structure: each group has an "id" and a `sign_list` of up to 10
entries, one per sign language, each `{lang, text, video_url}` — `lang` here
is the spoken-language code, and every entry in a group represents the SAME
underlying meaning signed/spoken in a different language. This produces two
kinds of records:

  1. "same-language" pairs — src sign language -> its own spoken text
     (e.g. csl video -> zh text). These are exactly Table 2's 10 one-to-one
     pairs and Table 5's many-to-many diagonal pairs.
  2. "cross-to-English" pairs — every non-English entry's video paired with
     the group's English text. This is an explicit implementation choice
     (not spelled out by the paper) for how many-to-one's "English target"
     data is constructed from SP-10's actual parallel structure — the only
     real English annotation available per group is the "en" entry's text.
     Flag this as an assumption if verified numbers diverge from Table 4.

`video_id` is left as `{group_id}_{lang}` and the original `video_url` is
kept as metadata for the separate (not-yet-implemented-here) video download
step (see MLSLT/SP-10's own video_downloader.py — that step needs ffmpeg and
real bandwidth, deliberately not done in this pass).
"""
import argparse
import json
import os

# spoken-language code (SP-10 JSON's "lang") -> this repo's sign-language code
SPOKEN_TO_SIGN_LANG = {
    "zh": "csl", "uk": "ukl", "ru": "rsl", "is": "icl", "de": "gsg",
    "it": "ise", "bg": "bqn", "sv": "swl", "lt": "lls", "en": "bfi",
}


def convert(raw_path: str) -> list:
    with open(raw_path, "r", encoding="utf-8") as f:
        groups = json.load(f)

    records = []
    for group in groups:
        gid = group["id"]
        by_lang = {e["lang"]: e for e in group["sign_list"]}
        english = by_lang.get("en")

        for entry in group["sign_list"]:
            spoken_lang = entry["lang"]
            sign_lang = SPOKEN_TO_SIGN_LANG.get(spoken_lang)
            if sign_lang is None:
                continue
            if not entry.get("video_url"):
                continue  # no video upstream (3 csl test groups): nothing to translate from
            # (1) same-language pair: this SL's own video -> this SL's own text
            records.append({
                "video_id": f"{gid}_{spoken_lang}",
                "src_lang": sign_lang,
                "tgt_lang": spoken_lang,
                "text": entry["text"],
                "video_url": entry["video_url"],
            })
            # (2) cross-to-English pair, skipping English itself (already
            # covered by (1)) and groups missing an English annotation.
            if english is not None and spoken_lang != "en":
                records.append({
                    "video_id": f"{gid}_{spoken_lang}_to_en",
                    "src_lang": sign_lang,
                    "tgt_lang": "en",
                    "text": english["text"],
                    "video_url": entry["video_url"],
                })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="references")
    parser.add_argument("--out-dir", default="data/splits/sp10")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ("train", "dev", "test"):
        raw_path = os.path.join(args.raw_dir, f"sp10_{split}_raw.json")
        records = convert(raw_path)
        out_path = os.path.join(args.out_dir, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        n_same_lang = sum(1 for r in records if not r["video_id"].endswith("_to_en"))
        print(f"{split}: {len(records)} total records ({n_same_lang} same-language, "
              f"{len(records) - n_same_lang} cross-to-English) -> {out_path}")


if __name__ == "__main__":
    main()
