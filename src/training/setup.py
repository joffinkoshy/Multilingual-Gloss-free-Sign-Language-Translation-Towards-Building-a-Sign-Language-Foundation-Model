"""Run setup shared by scripts/train.py and scripts/evaluate.py, so training
and evaluation always see the same language-pair filter, vocabulary and
feature source."""
import os
import warnings
from typing import List, Optional, Tuple

from src.datasets import csl_daily, phoenix14t, sp10
from src.datasets.common import load_index
from src.datasets.feature_cache import CachedFeatureReader
from src.datasets.tokenizer import LIDTokenizer, WordTokenizer

DATASET_MODULES = {"sp10": sp10, "phoenix14t": phoenix14t, "csl_daily": csl_daily}
ALL_SP10_LANGUAGES = ["csl", "ukl", "rsl", "icl", "gsg", "ise", "bqn", "swl", "lls", "bfi"]
TXT_VOCAB_FILE = "txt_vocab.txt"


def language_filter(config: dict, language_pair: Optional[Tuple[str, str]] = None) -> Optional[List[tuple]]:
    """(src_lang, tgt_lang) pairs a run trains/evaluates on.

    SP-10's index files hold both same-language pairs (csl->zh) and
    cross-to-English pairs (csl->en) — 15,770 train records. many_to_one must
    keep only the 10 x 830 = 8,300 records with the English target (paper
    Table 4 / Appendix C), otherwise native-language targets leak in.
    """
    if language_pair is not None:
        return [tuple(language_pair)]
    data = config["data"]
    if config["task"] == "many_to_one" and data.get("target_language"):
        sources = data.get("source_languages", ALL_SP10_LANGUAGES)
        return [(src, data["target_language"]) for src in sources]
    if data.get("language_pairs"):
        return [tuple(p) for p in data["language_pairs"]]
    return None


def build_tokenizers(config: dict, splits_dir: str, pairs: Optional[List[tuple]] = None, vocab_path: Optional[str] = None):
    tok_cfg = config["data"]["tokenizer"]
    if vocab_path is not None and os.path.exists(vocab_path):
        tokenizer = WordTokenizer.from_file(vocab_path, min_freq=tok_cfg["min_freq"], max_size=tok_cfg["max_size"])
    else:
        train_index_path = os.path.join(splits_dir, config["data"]["dataset"], "train.json")
        examples = load_index(train_index_path)
        if pairs is not None:
            allowed = set(pairs)
            examples = [e for e in examples if (e.src_lang, e.tgt_lang) in allowed]
            if not examples:
                raise ValueError(f"No training examples for language pairs {pairs} in {train_index_path}")
        tokenizer = WordTokenizer(min_freq=tok_cfg["min_freq"], max_size=tok_cfg["max_size"])
        tokenizer.build([e.text for e in examples])

    lid_tokenizer = None
    if config["model"]["use_lid_branch"]:
        lid_tokenizer = LIDTokenizer(config["data"].get("source_languages", ALL_SP10_LANGUAGES))
    return tokenizer, lid_tokenizer


def build_feature_fn(config: dict, allow_stub: bool = False):
    """CachedFeatureReader over config.data.feature_cache_dir, or None (=
    StubFeatureGenerator inside SLTDataset) only when explicitly allowed."""
    if allow_stub:
        warnings.warn("Using StubFeatureGenerator (random features) — smoke test only, numbers are meaningless.")
        return None
    cache_dir = config["data"]["feature_cache_dir"]
    if config["data"].get("require_real_features", False) or os.path.isdir(cache_dir):
        return CachedFeatureReader(cache_dir, d_feat=config["model"]["d_feat"])
    raise FileNotFoundError(
        f"No feature cache at '{cache_dir}'. Extract features first, or pass "
        "--allow-stub-features for a pipeline smoke test."
    )


def build_split(config: dict, split: str, splits_dir: str, tokenizer, pairs, feature_fn):
    module = DATASET_MODULES[config["data"]["dataset"]]
    kwargs = {"language_filter": pairs} if pairs is not None else {}
    if feature_fn is not None:
        kwargs["feature_fn"] = feature_fn
    ds = module.build(split, splits_dir, tokenizer, d_feat=config["model"]["d_feat"], **kwargs)
    if isinstance(feature_fn, CachedFeatureReader):
        # fail now, not an hour into the first epoch
        missing = [e.video_id for e in ds.examples if not os.path.exists(feature_fn.path_for(e.video_id))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(ds.examples)} {split} records have no cached features in "
                f"{feature_fn.cache_dir} (e.g. {missing[:3]})"
            )
    return ds
