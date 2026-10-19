"""Reader for cached per-video features written by
scripts/extract_features_dinov2.py: `<cache_dir>/<feature_key>.pt`, each a
torch.save'd [T, d_feat] tensor (float16 on disk, returned as float32).

SP-10 index records reuse one video for several targets ("410_zh" and
"410_zh_to_en" are the same csl clip), so features are keyed by the
underlying video, not the record: `feature_key` strips the "_to_<lang>"
suffix. Missing files raise — never silently fall back to stub features,
which would produce a run that "trains" on noise.
"""
import os

import torch


def feature_key(video_id: str) -> str:
    return video_id.split("_to_", 1)[0]


class CachedFeatureReader:
    def __init__(self, cache_dir: str, d_feat: int):
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(
                f"Feature cache dir '{cache_dir}' does not exist — run "
                "scripts/extract_features_dinov2.py first."
            )
        self.cache_dir = cache_dir
        self.d_feat = d_feat

    def path_for(self, video_id: str) -> str:
        return os.path.join(self.cache_dir, f"{feature_key(video_id)}.pt")

    def __call__(self, video_id: str, n_text_tokens: int) -> torch.Tensor:
        path = self.path_for(video_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing cached features for '{video_id}': {path}")
        feats = torch.load(path, map_location="cpu").float()
        if feats.dim() != 2 or feats.shape[1] != self.d_feat:
            raise ValueError(f"{path}: expected [T, {self.d_feat}], got {tuple(feats.shape)} — check model.d_feat")
        return feats
