#!/usr/bin/env python
"""Feature extraction entry point (Task 7, Section 10).

Confirmed by directly inspecting kaistmm/SlowFastSign and
Claire874/Gloss-free-MLSLT's readme (see
experiments/results/reference_inspection_notes.md, Section 12 item 10):
no public precomputed features exist for SP-10, PHOENIX14T, or CSL-Daily in
the form this paper needs. The public SlowFastSign checkpoints are gloss-
supervised recognition models trained only on PHOENIX2014/2014-T/CSL-Daily
(their README reports WER, a recognition metric) — the MLSLT paper explicitly
retrains/uses a *spoken-text-supervised* variant instead (Section 2.2), which
is not public, and there's no SP-10 checkpoint at all.

Real path forward (to run on the target RTX 3050 machine, not here):
  1. Clone kaistmm/SlowFastSign and get it running per its own README
     (PyTorch 1.13, ctcdecode, PySlowFast pretrained backbone).
  2. Retrain (or fine-tune) it with spoken-text targets instead of gloss
     targets on each dataset's transcripts, OR confirm with the paper's
     authors whether they release the actual checkpoint used.
  3. Run inference over each dataset's videos to produce per-frame features,
     writing them to data/features/<dataset>/<video_id>.pt as a plain
     torch.save'd [T, d_feat] tensor.
  4. Point config.data.feature_cache_dir at that directory and set
     config.model.d_feat to the real per-frame dimensionality (read off the
     checkpoint, per Section 12 item 1) — do not hardcode it elsewhere.

This script is the wiring point for step 3/4 once step 1-2 exist elsewhere;
it is intentionally not runnable today (no GPU, no SlowFastSign checkpoint,
no raw video in this environment).
"""
import argparse
import sys

from src.datasets.feature_extraction import SlowFastSignExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="path to a spoken-text-supervised SlowFastSign checkpoint")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    extractor = SlowFastSignExtractor(args.checkpoint)  # raises NotImplementedError today
    print(f"d_feat inferred from checkpoint: {extractor.d_feat}")


if __name__ == "__main__":
    sys.exit(main())
