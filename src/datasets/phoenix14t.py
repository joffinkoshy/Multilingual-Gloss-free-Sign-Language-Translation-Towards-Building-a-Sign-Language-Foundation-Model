"""PHOENIX14T (Camgoz et al. 2018) dataset loader. gsg (German SL), weather
domain, standard splits: 7,096 train / 519 dev / 642 test, 2.9k vocab, 9
signers (Section 2.1)."""
import os

from src.datasets.common import SLTDataset
from src.datasets.tokenizer import WordTokenizer

SPLIT_SIZES = {"train": 7096, "dev": 519, "test": 642}


def build(split: str, splits_dir: str, tokenizer: WordTokenizer, d_feat: int, **kwargs) -> SLTDataset:
    index_path = os.path.join(splits_dir, "phoenix14t", f"{split}.json")
    return SLTDataset(index_path, tokenizer, d_feat=d_feat, **kwargs)
