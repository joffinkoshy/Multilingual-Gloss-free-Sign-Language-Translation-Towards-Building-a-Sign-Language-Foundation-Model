#!/usr/bin/env bash
# Runs every one_to_one model the paper trains as an independent model
# (Section 2.1/Table 2): all 10 SP-10 pairs + PHOENIX14T's gsg->de.
# CSL-Daily's csl->zh pair is DEFERRED (2026-08-26 decision — no readily
# available staff signatory for its access agreement, see
# REPRODUCTION_STATUS.md) and intentionally excluded here; add it back with
# the same pattern once data/splits/csl_daily/ exists.
#
# Each pair gets its own model dir under experiments/baseline/one_to_one/ —
# these are 11 independent models, not one multi-task model (Section 2.1:
# "one-to-one task is evaluated per-language-pair").
set -euo pipefail

SPLITS_DIR="${SPLITS_DIR:-data/splits}"
DEVICE="${DEVICE:-cuda}"

SP10_PAIRS=(
  "csl:zh" "ukl:uk" "rsl:ru" "icl:is" "gsg:de"
  "ise:it" "bqn:bg" "swl:sv" "lls:lt" "bfi:en"
)

for pair in "${SP10_PAIRS[@]}"; do
  echo "=== SP-10 one_to_one ${pair} ==="
  python scripts/train.py \
    --config configs/sp10_one_to_one.yaml \
    --splits-dir "${SPLITS_DIR}" \
    --model-dir "experiments/baseline/one_to_one/sp10_${pair/:/_}" \
    --language-pair "${pair}" \
    --device "${DEVICE}"
done

echo "=== PHOENIX14T one_to_one gsg:de ==="
python scripts/train.py \
  --config configs/phoenix14t.yaml \
  --splits-dir "${SPLITS_DIR}" \
  --model-dir "experiments/baseline/one_to_one/phoenix14t_gsg_de" \
  --device "${DEVICE}"

echo "All one_to_one runs complete (CSL-Daily's csl:zh pair deferred). Compare with:"
echo "  python -m src.evaluation.compare_to_paper experiments/results/<run>/metrics.json --table table_4_1"
echo "  python -m src.evaluation.compare_to_paper experiments/results/<run>/metrics.json --table table_4_2"
