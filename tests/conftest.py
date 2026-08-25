import json
import os

import pytest

from src.datasets.tokenizer import LIDTokenizer, WordTokenizer

SP10_LANGS = ["csl", "ukl", "rsl", "icl", "gsg", "ise", "bqn", "swl", "lls", "bfi"]

SAMPLE_TEXTS = [
    "the weather tomorrow will be cold and windy",
    "it is snowing in the mountains today",
    "please translate this sentence into your language",
    "hello how are you doing today",
    "the sun is shining brightly this afternoon",
]


def make_index(tmp_path, dataset: str, split: str, n: int = 8):
    records = []
    for i in range(n):
        records.append({
            "video_id": f"{dataset}_{split}_{i:03d}",
            "src_lang": SP10_LANGS[i % len(SP10_LANGS)],
            "tgt_lang": "en",
            "text": SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)],
        })
    d = tmp_path / "splits" / dataset
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{split}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    return str(path)


@pytest.fixture
def tokenizer():
    tok = WordTokenizer(min_freq=1, max_size=1000)
    tok.build(SAMPLE_TEXTS)
    return tok


@pytest.fixture
def lid_tokenizer():
    return LIDTokenizer(SP10_LANGS)
