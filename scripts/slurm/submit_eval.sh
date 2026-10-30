#!/bin/bash
# Submit the evaluation matrix: test-time steps K' in {1,2,3} for every trained
# model, plus an initial-pose-noise robustness sweep for the main model.
#
#   bash scripts/slurm/submit_eval.sh
#   DRY_RUN=1 bash scripts/slurm/submit_eval.sh
#
# Environment: MODEL_NAME (checkpoint stem, default epoch_002 = the last epoch
# of the default 3-epoch run), STEPS (default "1 2 3"), NOISE_DEG (default
# "5 10 20"), MRC_VENV/MRC_CONDA_ENV as in train.sbatch.
set -euo pipefail

cd "$(dirname "$0")/../.."

SBATCH="${SBATCH:-sbatch}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_NAME="${MODEL_NAME:-epoch_002}"
STEPS="${STEPS:-1 2 3}"
NOISE_DEG="${NOISE_DEG:-5 10 20}"

submit() {
  local run_name="$1" stereo_mode="$2" fusion="$3" n_refine="$4" suffix="$5"
  local extra="${6:-}"
  echo "== $run_name K'=$n_refine ($suffix)"
  local env_args=(
    "RUN_NAME=$run_name" "DATASET=stereobj-1m" "SPLIT=val"
    "STEREO_MODE=$stereo_mode" "FUSION=$fusion" "N_REFINE=$n_refine"
    "MODEL_NAME=$MODEL_NAME" "SUFFIX=$suffix" "EVAL=1")
  [ -n "$extra" ] && env_args+=("$extra")
  if [ "$DRY_RUN" = "1" ]; then
    echo "   env ${env_args[*]} $SBATCH scripts/slurm/infer.sbatch"
  else
    env "${env_args[@]}" "$SBATCH" scripts/slurm/infer.sbatch
  fi
}

# model: run_name stereo_mode fusion
models=(
  "stereo_cross_K3 stereo cross_view"
  "stereo_cross_K1 stereo cross_view"
  "stereo_mean_K3  stereo mean"
  "stereo_left_K3  stereo left_only"
  "stereo_right_K3 stereo right_only"
  "mono_K1         mono   -"
  "mono_K3         mono   -"
)

for entry in "${models[@]}"; do
  read -r run stereo fusion <<<"$entry"
  [ "$fusion" = "-" ] && fusion=""
  for k in $STEPS; do
    submit "$run" "$stereo" "$fusion" "$k" "k${k}"
  done
done

# Initial-pose-noise robustness (main model, K=3).
for nd in $NOISE_DEG; do
  submit stereo_cross_K3 stereo cross_view 3 "k3_noise${nd}" \
    "INIT_ROT_NOISE=$nd"
done

echo "Done. Then: python tools/collect_results.py --manifest <manifest.json>"
