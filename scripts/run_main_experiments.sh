#!/usr/bin/env bash
# Main course-project experiments (PLAN_LID_SELF_CONDITIONING.md §5):
#   E1 baseline  = paper's Sign2(LID+Text)          configs/sp10_m2o_dinov2.yaml
#   E2 proposed  = E1 + LID self-conditioning        configs/proposed_lid_selfcond.yaml
# SP-10 many-to-one (10 SLs -> English), same features/hyper-parameters,
# matched seeds; then test-set evaluation and the baseline-vs-proposed table.
#
# Prerequisite: data/features/sp10_dinov2/ from scripts/extract_features_dinov2.py
#
#   SEEDS="1 2 3" DEVICE=cuda bash scripts/run_main_experiments.sh
#   WITH_UNIVERSAL=1 ...   also trains optional E0 (vanilla Transformer, universal)
#   EXTRA_ARGS="--allow-stub-features --max-epochs 1"   pipeline smoke test
#   CONFIG_DIR=<dir>   read the same-named configs from another directory
# Re-running skips runs whose best.ckpt / metrics.json already exist.
set -euo pipefail

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-1 2 3}"
EXP_DIR="${EXP_DIR:-experiments}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

train_and_eval() {
  local name="$1" config="$2" seed="$3"
  local dir="${EXP_DIR}/${name}/seed${seed}"
  if [[ ! -f "${dir}/best.ckpt" ]]; then
    echo "=== train ${name} seed ${seed} (${config}) ==="
    # shellcheck disable=SC2086
    "${PYTHON}" scripts/train.py --config "${CONFIG_DIR:-configs}/$(basename ${config})" --model-dir "${dir}" --seed "${seed}" --device "${DEVICE}" ${EXTRA_ARGS}
  fi
  for split in dev test; do
    if [[ ! -f "${dir}/eval_${split}/metrics.json" ]]; then
      echo "=== eval ${name} seed ${seed} ${split} ==="
      # shellcheck disable=SC2086
      "${PYTHON}" scripts/evaluate.py --config "${CONFIG_DIR:-configs}/$(basename ${config})" --model-dir "${dir}" --split "${split}" --device "${DEVICE}" \
        $( [[ "${EXTRA_ARGS}" == *--allow-stub-features* ]] && echo --allow-stub-features )
    fi
  done
}

baseline_dirs=() proposed_dirs=()
for seed in ${SEEDS}; do
  train_and_eval baseline configs/sp10_m2o_dinov2.yaml "${seed}"
  train_and_eval proposed configs/proposed_lid_selfcond.yaml "${seed}"
  baseline_dirs+=("${EXP_DIR}/baseline/seed${seed}")
  proposed_dirs+=("${EXP_DIR}/proposed/seed${seed}")
done

if [[ "${WITH_UNIVERSAL:-0}" == "1" ]]; then
  train_and_eval universal configs/sp10_m2o_dinov2_universal.yaml "$(echo ${SEEDS} | cut -d' ' -f1)"
fi

for split in dev test; do
  "${PYTHON}" scripts/compare_runs.py --split "${split}" \
    --baseline "${baseline_dirs[@]}" --proposed "${proposed_dirs[@]}" \
    --output "${EXP_DIR}/results/comparison_${split}"
done
