"""CSL-Daily (Zhou et al. 2021) dataset loader. csl (Chinese SL), 18,401
train / 1,077 dev / 1,176 test, 2.3k vocab, 10 signers (Section 2.1)."""
import os

from src.datasets.common import SLTDataset
from src.datasets.tokenizer import WordTokenizer

SPLIT_SIZES = {"train": 18401, "dev": 1077, "test": 1176}


def build(split: str, splits_dir: str, tokenizer: WordTokenizer, d_feat: int, **kwargs) -> SLTDataset:
    index_path = os.path.join(splits_dir, "csl_daily", f"{split}.json")
    return SLTDataset(index_path, tokenizer, d_feat=d_feat, **kwargs)
