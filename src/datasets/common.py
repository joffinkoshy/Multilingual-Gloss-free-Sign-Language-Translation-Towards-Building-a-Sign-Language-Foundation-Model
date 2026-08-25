"""Shared dataset/collation logic for SP-10, PHOENIX14T, CSL-Daily.

Index files (data/splits/<dataset>/<split>.json) are plain JSON lists of
records: {"video_id", "src_lang", "tgt_lang", "text"}. Stage 2 of the handoff
is explicit that this stage works with index files, not raw video parsing —
actual frame decoding is SlowFastSign's job (Section 2.2), which is frozen
and external. Until a real checkpoint is available, feature vectors come from
StubFeatureGenerator (Section 12 item 10).
"""
import json
import os
from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset

from src.datasets.feature_extraction import StubFeatureGenerator
from src.datasets.tokenizer import WordTokenizer


class SLTExample:
    __slots__ = ("video_id", "src_lang", "tgt_lang", "text", "video_url")

    def __init__(self, video_id: str, src_lang: str, tgt_lang: str, text: str, video_url: Optional[str] = None):
        self.video_id = video_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.text = text
        self.video_url = video_url  # present once converted from a real source (e.g. SP-10); None for synthetic fixtures


def load_index(path: str) -> List[SLTExample]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [SLTExample(**r) for r in records]


class SLTDataset(Dataset):
    """Generic sign-language-translation dataset over an index file.

    `feature_fn(video_id, n_text_tokens) -> Tensor[T, d_feat]` supplies the
    per-frame features — a StubFeatureGenerator during pipeline development,
    a real feature-cache reader once SlowFastSign features exist.
    """

    def __init__(
        self,
        index_path: str,
        tokenizer: WordTokenizer,
        feature_fn: Optional[Callable[[str, int], torch.Tensor]] = None,
        d_feat: int = 1024,
        language_filter: Optional[List[tuple]] = None,
    ):
        self.examples = load_index(index_path)
        if language_filter is not None:
            allowed = set(language_filter)
            self.examples = [e for e in self.examples if (e.src_lang, e.tgt_lang) in allowed]
        self.tokenizer = tokenizer
        self.feature_fn = feature_fn or StubFeatureGenerator(d_feat=d_feat)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        text_ids = self.tokenizer.encode(ex.text, add_bos=False, add_eos=True)
        features = self.feature_fn(ex.video_id, len(text_ids))
        return {
            "video_id": ex.video_id,
            "src_lang": ex.src_lang,
            "tgt_lang": ex.tgt_lang,
            "features": features,          # [T_frames, d_feat]
            "text_ids": text_ids,           # [T_text] (ends with <eos>)
            "text": ex.text,
        }


def collate_slt_batch(batch: List[dict], pad_id: int) -> dict:
    d_feat = batch[0]["features"].shape[-1]
    max_frames = max(b["features"].shape[0] for b in batch)
    max_text = max(len(b["text_ids"]) for b in batch)
    bsz = len(batch)

    features = torch.zeros(bsz, max_frames, d_feat)
    feature_lengths = torch.zeros(bsz, dtype=torch.long)
    text_ids = torch.full((bsz, max_text), pad_id, dtype=torch.long)
    text_lengths = torch.zeros(bsz, dtype=torch.long)

    for i, b in enumerate(batch):
        t = b["features"].shape[0]
        features[i, :t] = b["features"]
        feature_lengths[i] = t
        n = len(b["text_ids"])
        text_ids[i, :n] = torch.tensor(b["text_ids"], dtype=torch.long)
        text_lengths[i] = n

    return {
        "features": features,               # [B, |F|, d_feat]
        "feature_lengths": feature_lengths,   # [B]
        "text_ids": text_ids,                  # [B, |T|]
        "text_lengths": text_lengths,            # [B]
        "src_langs": [b["src_lang"] for b in batch],
        "tgt_langs": [b["tgt_lang"] for b in batch],
        "video_ids": [b["video_id"] for b in batch],
        "texts": [b["text"] for b in batch],
    }
