#!/usr/bin/env bash
# Sequential ablation ladder (Section 9): baseline -> +routing ->
# +routing+augmentation -> +routing+augmentation+pretraining (V1 combined).
# 4 runs, cumulative, NOT the full 8-run factorial — a deliberate compute
# decision (Section 11), don't "complete" this into a factorial sweep.
#
# Requires real data/splits/sp10/{train,dev,test}.json and real cached
# SlowFastSign features to produce meaningful numbers (Section 12 item 10) —
# runs today only exercise the pipeline against StubFeatureGenerator.
set -euo pipefail

SPLITS_DIR="${SPLITS_DIR:-data/splits}"
DEVICE="${DEVICE:-cuda}"

run() {
  local name="$1" config="$2"
  echo "=== ${name} (${config}) ==="
  python scripts/train.py \
    --config "${config}" \
    --splits-dir "${SPLITS_DIR}" \
    --model-dir "experiments/${name}/sp10_many_to_one" \
    --device "${DEVICE}"
}

run baseline               configs/sp10_many_to_one.yaml
run v1_family_routing      configs/v1_family_routing.yaml
run v1_augmentation        configs/v1_augmentation.yaml
run v1_combined            configs/v1_combined.yaml

echo "All four ladder runs complete. Compare with:"
echo "  python -m src.evaluation.compare_to_paper experiments/results/<run>/metrics.json --table table_4_3"
