#!/usr/bin/env bash
# Reproduce the reported test-set numbers from the released checkpoints.
#
# Needs (see README §6):
#   experiments/{baseline,proposed}/seed{1,2,3}/{best.ckpt,txt_vocab.txt}   (Google Drive: mlslt_checkpoints.zip)
#   data/features/sp10_dinov2/*.pt                                           (Google Drive: sp10_dinov2_features.zip,
#                                                                             or rebuild with scripts/extract_features_dinov2.py)
#
#   DEVICE=cpu bash scripts/reproduce_reported_metrics.sh      # ~8 min on an M1 CPU (greedy decoding)
#   DEVICE=mps ...                                             # Apple GPU
#
# Writes <run>/eval_test/ (metrics.json, hypotheses.tsv) and
# experiments/results/reproduced_test.{md,json}, which should match
# experiments/results/comparison_test.md (same checkpoints, greedy decoding).
set -euo pipefail

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cpu}"
SEEDS="${SEEDS:-1 2 3}"
EXP_DIR="${EXP_DIR:-experiments}"

baseline_dirs=() proposed_dirs=()
for seed in ${SEEDS}; do
  for pair in "baseline:configs/sp10_m2o_dinov2.yaml" "proposed:configs/proposed_lid_selfcond.yaml"; do
    name="${pair%%:*}"; config="${pair##*:}"
    dir="${EXP_DIR}/${name}/seed${seed}"
    [[ -f "${dir}/best.ckpt" ]] || { echo "missing ${dir}/best.ckpt (download mlslt_checkpoints.zip, see README §6)"; exit 1; }
    echo "=== evaluate ${name} seed ${seed} (test) ==="
    "${PYTHON}" scripts/evaluate.py --config "${config}" --model-dir "${dir}" --split test --device "${DEVICE}" \
      --output-dir "${dir}/eval_test" | grep -E '"(bleu|rouge_l|decoding)"'
  done
  baseline_dirs+=("${EXP_DIR}/baseline/seed${seed}")
  proposed_dirs+=("${EXP_DIR}/proposed/seed${seed}")
done

"${PYTHON}" scripts/compare_runs.py --split test \
  --baseline "${baseline_dirs[@]}" --proposed "${proposed_dirs[@]}" \
  --output "${EXP_DIR}/results/reproduced_test"
echo "Compare with ${EXP_DIR}/results/comparison_test.md"
