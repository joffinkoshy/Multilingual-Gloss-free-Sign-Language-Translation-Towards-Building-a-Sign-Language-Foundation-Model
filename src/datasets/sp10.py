"""SP-10 (SpreadTheSign, Yin et al. 2022) dataset loader.

10 SLs, 830 train examples/language (8,300 total), 1,420 dev, 2,021 test,
16.7k aggregate multilingual vocab, 79 signers (Section 2.1).
"""
import os

from src.datasets.common import SLTDataset
from src.datasets.tokenizer import WordTokenizer

LANGUAGES = ["csl", "ukl", "rsl", "icl", "gsg", "ise", "bqn", "swl", "lls", "bfi"]
SPLIT_SIZES = {"train": 8300, "dev": 1420, "test": 2021}


def build(split: str, splits_dir: str, tokenizer: WordTokenizer, d_feat: int, **kwargs) -> SLTDataset:
    index_path = os.path.join(splits_dir, "sp10", f"{split}.json")
    return SLTDataset(index_path, tokenizer, d_feat=d_feat, **kwargs)
