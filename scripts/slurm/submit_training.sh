#!/bin/bash
# Submit the fixed-scope training matrix (see the experiment plan) to SLURM.
#
#   bash scripts/slurm/submit_training.sh            # submit all cells
#   DRY_RUN=1 bash scripts/slurm/submit_training.sh  # print sbatch commands
#
# Override any shared setting via the environment, e.g.
#   N_EPOCHS=75 BATCH_SIZE=16 MRC_VENV=$PWD/.venv \
#     bash scripts/slurm/submit_training.sh
#
# The probe fine-tune arm reuses the mono stereobj-1m checkpoint; set
# PROBE_PRETRAINED to that checkpoint path (defaults to the mono_K3 run's last
# epoch).  The stereo probe arm needs a stereo probe dataset definition and is
# intentionally not submitted here (see README/plan).
set -euo pipefail

cd "$(dirname "$0")/../.."

SBATCH="${SBATCH:-sbatch}"
DRY_RUN="${DRY_RUN:-0}"
N_EPOCHS="${N_EPOCHS:-75}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WARMUP="${WARMUP:-1000}"
N_GPUS="${N_GPUS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:---no_compile}"

submit() {
  local run_name="$1" dataset="$2" stereo_mode="$3" fusion="$4" n_refine="$5"
  local pretrained="${6:-}"
  echo "== $run_name (dataset=$dataset stereo=$stereo_mode fusion=$fusion K=$n_refine)"
  local env_args=(
    "RUN_NAME=$run_name" "DATASET=$dataset" "STEREO_MODE=$stereo_mode"
    "FUSION=$fusion" "N_REFINE=$n_refine" "N_EPOCHS=$N_EPOCHS"
    "BATCH_SIZE=$BATCH_SIZE" "NUM_WORKERS=$NUM_WORKERS" "WARMUP=$WARMUP"
    "N_GPUS=$N_GPUS" "EXTRA_ARGS=$EXTRA_ARGS")
  [ -n "$pretrained" ] && env_args+=("PRETRAINED=$pretrained")
  if [ "$DRY_RUN" = "1" ]; then
    echo "   env ${env_args[*]} $SBATCH scripts/slurm/train.sbatch"
  else
    env "${env_args[@]}" "$SBATCH" scripts/slurm/train.sbatch
  fi
}

# --- Main model + train-K vs test-K' ---------------------------------------
submit stereo_cross_K3 stereobj-1m stereo cross_view 3
submit stereo_cross_K1 stereobj-1m stereo cross_view 1

# --- Fusion ablations (at the main K=3) ------------------------------------
submit stereo_mean_K3  stereobj-1m stereo mean      3
submit stereo_left_K3  stereobj-1m stereo left_only 3
submit stereo_right_K3 stereobj-1m stereo right_only 3

# --- Mono arm (same schedule) ----------------------------------------------
submit mono_K1 stereobj-1m mono "" 1
submit mono_K3 stereobj-1m mono "" 3

# --- Probe transfer (fine-tune on the probe training split) -----------------
# Mono arm: init from the mono stereobj-1m model, matched schedule.
PROBE_PRETRAINED="${PROBE_PRETRAINED:-chkpt_stereobj-1m_mono_K3/epoch_070.pth}"
submit probe_mono_ft usprobe mono "" 3 "$PROBE_PRETRAINED"

echo "Done. Track with: squeue -u \$USER"
