#!/usr/bin/env python
"""Extract frozen DINOv2 frame features for SP-10 (PLAN_LID_SELF_CONDITIONING.md §3).

Replaces the unavailable spoken-text-supervised SlowFastSign features (see
scripts/extract_features.py). DINOv2 is self-supervised on images — no
gloss or text supervision — so the gloss-free constraint holds. Baseline and
proposed models read the same cache, so their comparison is fair.

Input layout (MLSLT/SP-10's video_downloader.py + video_converter.py, 25 fps):
    <video-dir>/<split>/<group_id>/<spoken_lang>.mp4
Output: <output-dir>/<group_id>_<spoken_lang>.pt  — float16 [T, d_feat], the
key src/datasets/feature_cache.py::feature_key derives from index video_ids.

    python scripts/extract_features_dinov2.py --video-dir data/raw/sp10 \
        --output-dir data/features/sp10_dinov2 --device cuda

Re-running skips videos whose .pt already exists, so it can be interrupted.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_frames(path: str, stride: int, size: int) -> np.ndarray:
    """Every `stride`-th frame, resized (whole frame, no crop — hands reach the
    edges) to size x size, RGB, ImageNet-normalised: [N, 3, size, size]."""
    import cv2

    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frame = cv2.cvtColor(cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            frames.append(frame)
        i += 1
    cap.release()
    if not frames:
        return np.zeros((0, 3, size, size), dtype=np.float32)
    x = np.stack(frames).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(0, 3, 1, 2)


def load_dinov2(name: str, device: str) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", name)
    return model.eval().to(device)


@torch.no_grad()
def encode_frames(encoder, frames: np.ndarray, device: str, batch_size: int) -> torch.Tensor:
    """encoder: images [B, 3, H, W] -> CLS embeddings [B, d_feat]."""
    outs = []
    use_amp = device.startswith("cuda")
    for start in range(0, len(frames), batch_size):
        x = torch.from_numpy(frames[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outs.append(encoder(x).float().cpu())
    return torch.cat(outs, dim=0)


def find_videos(video_dir: str, splits):
    """Yields (split, feature_key, path) for <video_dir>/<split>/<gid>/<lang>.mp4."""
    for split in splits:
        for path in sorted(Path(video_dir, split).glob("*/*.mp4")):
            yield split, f"{path.parent.name}_{path.stem}", str(path)


def extract_all(videos, encoder, output_dir: str, device: str, stride: int, size: int, batch_size: int) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    stats = {"written": 0, "skipped_existing": 0, "empty": [], "n_frames": []}
    seen = {}
    for split, key, path in videos:
        if key in seen and seen[key] != split:
            raise ValueError(f"feature key {key} appears in both {seen[key]} and {split}")
        seen[key] = split
        out_path = os.path.join(output_dir, f"{key}.pt")
        if os.path.exists(out_path):
            stats["skipped_existing"] += 1
            continue
        frames = read_frames(path, stride, size)
        if len(frames) == 0:
            stats["empty"].append(path)
            continue
        feats = encode_frames(encoder, frames, device, batch_size)
        torch.save(feats.half(), out_path)
        stats["written"] += 1
        stats["n_frames"].append(int(feats.shape[0]))
        if stats["written"] % 200 == 0:
            print(f"[{split}] {stats['written']} videos written (last: {key}, {feats.shape[0]} frames)", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--model", default="dinov2_vits14", help="torch.hub DINOv2 variant (vits14 -> d_feat=384)")
    parser.add_argument("--frame-stride", type=int, default=2, help="keep every Nth frame (25 fps -> 12.5 fps)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encoder = load_dinov2(args.model, args.device)
    stats = extract_all(
        find_videos(args.video_dir, args.splits), encoder, args.output_dir,
        args.device, args.frame_stride, args.image_size, args.batch_size,
    )
    lengths = stats.pop("n_frames")
    summary = {
        **vars(args), **stats,
        "frames_per_video": {"min": min(lengths), "mean": float(np.mean(lengths)), "max": max(lengths)} if lengths else None,
    }
    with open(os.path.join(args.output_dir, "extraction_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "empty"}, indent=2))
    if stats["empty"]:
        print(f"WARNING: {len(stats['empty'])} unreadable/empty videos, listed in extraction_summary.json", file=sys.stderr)


if __name__ == "__main__":
    main()
