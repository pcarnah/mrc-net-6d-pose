#!/bin/bash
# Submit the fixed-scope training matrix (see the experiment plan) to SLURM.
#
#   bash scripts/slurm/submit_training.sh            # submit all cells
#   DRY_RUN=1 bash scripts/slurm/submit_training.sh  # print sbatch commands
#
# Override any shared setting via the environment, e.g.
#   N_EPOCHS=3 BATCH_SIZE=16 TIME=168:00:00 \
#     bash scripts/slurm/submit_training.sh
#
# stereobj-1m is ~24h/epoch, so the default schedule is a single 7-day job
# running N_EPOCHS=3 with checkpoints on a step interval (SAVE_EVERY_STEPS) plus
# every epoch (SAVE_EVERY_EPOCHS=1).  Each job builds its own venv on
# $SLURM_TMPDIR and stages the dataset there (see scripts/slurm/setup_env.sh).
# Training runs with bf16 autocast by default (BF16=1; ~1.2x faster, ~25 percent
# less memory); set BF16=0 to fall back to fp32.
#
# The probe fine-tune arm reuses the mono stereobj-1m checkpoint; set
# PROBE_PRETRAINED to that checkpoint path (defaults to the mono_K3 run's last
# epoch).  It trains on usprobe, so point USPROBE_SRC at the staged usprobe
# dataset (defaults to the sibling project dir).  The stereo probe arm needs a
# stereo probe dataset definition and is intentionally not submitted here (see
# README/plan).
set -euo pipefail

cd "$(dirname "$0")/../.."

SBATCH="${SBATCH:-sbatch}"
DRY_RUN="${DRY_RUN:-0}"
TIME="${TIME:-48:00:00}"
N_EPOCHS="${N_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-12}"
WARMUP="${WARMUP:-1000}"
N_GPUS="${N_GPUS:-1}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-5000}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
BF16="${BF16:-1}"
EXTRA_ARGS="${EXTRA_ARGS:---no_compile}"
USPROBE_SRC="${USPROBE_SRC:-$HOME/projects/def-chene/pcarnah/usprobe}"

submit() {
  local run_name="$1" dataset="$2" stereo_mode="$3" fusion="$4" n_refine="$5"
  local pretrained="${6:-}" stage_name="${7:-}" stage_src="${8:-}"
  echo "== $run_name (dataset=$dataset stereo=$stereo_mode fusion=$fusion K=$n_refine)"
  local env_args=(
    "RUN_NAME=$run_name" "DATASET=$dataset" "STEREO_MODE=$stereo_mode"
    "FUSION=$fusion" "N_REFINE=$n_refine" "N_EPOCHS=$N_EPOCHS"
    "BATCH_SIZE=$BATCH_SIZE" "NUM_WORKERS=$NUM_WORKERS" "WARMUP=$WARMUP"
    "N_GPUS=$N_GPUS" "EXTRA_ARGS=$EXTRA_ARGS" "BF16=$BF16"
    "SAVE_EVERY_STEPS=$SAVE_EVERY_STEPS" "SAVE_EVERY_EPOCHS=$SAVE_EVERY_EPOCHS")
  [ -n "$pretrained" ] && env_args+=("PRETRAINED=$pretrained")
  [ -n "$stage_name" ] && env_args+=("MRC_STAGE_NAME=$stage_name")
  [ -n "$stage_src" ] && env_args+=("MRC_STAGE_SRC=$stage_src")
  if [ "$DRY_RUN" = "1" ]; then
    echo "   env ${env_args[*]} $SBATCH --time=$TIME scripts/slurm/train.sbatch"
  else
    env "${env_args[@]}" "$SBATCH" --time="$TIME" scripts/slurm/train.sbatch
  fi
}

PRETRAINED="chkpt_ycbv-ds/ycbv_real.pth"
# --- Main model + train-K vs test-K' ---------------------------------------
submit stereo_cross_K3 stereobj-1m stereo cross_view 3 "$PRETRAINED"
submit stereo_cross_K1 stereobj-1m stereo cross_view 1 "$PRETRAINED"

# --- Fusion ablations (at the main K=3) ------------------------------------
submit stereo_mean_K3  stereobj-1m stereo mean      3 "$PRETRAINED"
submit stereo_left_K3  stereobj-1m stereo left_only 3 "$PRETRAINED"
submit stereo_right_K3 stereobj-1m stereo right_only 3 "$PRETRAINED"

# --- Mono arm (same schedule) ----------------------------------------------
submit mono_K1 stereobj-1m mono "" 1 "$PRETRAINED"
submit mono_K3 stereobj-1m mono "" 3 "$PRETRAINED"

# --- Probe transfer (fine-tune on the probe training split) -----------------
# Mono arm: init from the mono stereobj-1m model, matched schedule.
#PROBE_PRETRAINED="${PROBE_PRETRAINED:-chkpt_stereobj-1m_mono_K3/epoch_002.pth}"
#submit probe_mono_ft usprobe mono "" 3 "$PROBE_PRETRAINED" usprobe "$USPROBE_SRC"

echo "Done. Track with: squeue -u \$USER"
