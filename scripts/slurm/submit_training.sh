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
# The submitted matrix follows RUN_PLAN.md:
#   mono_K1 / mono_K3          stereobj-1m mono sources (3 epochs, MONO_N_EPOCHS)
#   fr_{reg,regclf,regcorr,full}_k{1,3}
#                              frozen fusion-component matrix (1 epoch/arm,
#                              --freeze_except_stereo, K-matched mono source)
#   fm_left_k1 / fm_mean_k1    fusion-mode baselines (frozen, reg axis only)
# The usprobe corollary arm is winner-dependent and left as a template.
# Override the mono source checkpoint paths via PRETRAINED_MONO_K1/K3.
set -euo pipefail

cd "$(dirname "$0")/../.."

SBATCH="${SBATCH:-sbatch}"
DRY_RUN="${DRY_RUN:-0}"
TIME="${TIME:-12:00:00}"
N_EPOCHS="${N_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-12}"
WARMUP="${WARMUP:-1000}"
N_GPUS="${N_GPUS:-1}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-10000}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
BF16="${BF16:-1}"
EXTRA_ARGS="${EXTRA_ARGS:---no_compile}"
USPROBE_SRC="${USPROBE_SRC:-$HOME/projects/def-chene/pcarnah/usprobe}"

submit() {
  local run_name="$1" dataset="$2" stereo_mode="$3" fusion="$4" n_refine="$5"
  local pretrained="${6:-}" stage_name="${7:-}" stage_src="${8:-}" \
      extra="${9:-}"
  echo "== $run_name (dataset=$dataset stereo=$stereo_mode fusion=$fusion K=$n_refine)"
  local extra_args="$EXTRA_ARGS"
  [ -n "$extra" ] && extra_args="$extra_args $extra"
  local env_args=(
    "RUN_NAME=$run_name" "DATASET=$dataset" "STEREO_MODE=$stereo_mode"
    "FUSION=$fusion" "N_REFINE=$n_refine" "N_EPOCHS=$N_EPOCHS"
    "BATCH_SIZE=$BATCH_SIZE" "NUM_WORKERS=$NUM_WORKERS" "WARMUP=$WARMUP"
    "N_GPUS=$N_GPUS" "EXTRA_ARGS=$extra_args" "BF16=$BF16"
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

# Sources: stereobj-1m mono runs at matched K (the frozen arms init from their
# first epoch checkpoint; adjust the file names to the actual saves).
PRETRAINED_MONO_K1="${PRETRAINED_MONO_K1:-chkpt_stereobj-1m_mono_K1/epoch_001.pth}"
PRETRAINED_MONO_K3="${PRETRAINED_MONO_K3:-chkpt_stereobj-1m_mono_K3/epoch_001.pth}"
FREEZE_FLAGS="--freeze_except_stereo"

# --- Mono sources (dependency for every frozen arm; 3 epochs per plan) ------
# (N_EPOCHS="${MONO_N_EPOCHS:-3}" submit mono_K1 stereobj-1m mono "" 1 "")
# (N_EPOCHS="${MONO_N_EPOCHS:-3}" submit mono_K3 stereobj-1m mono "" 3 "")

# --- Frozen fusion-component matrix (stereobj-1m, 1 epoch per arm) ----------
# reg-only / reg+clf / reg+corr / full, x K in {1,3}. All arms init from the
# K-matched mono source and freeze everything outside the stereo fusion
# blocks + regressor (--freeze_except_stereo). clf_fusion/fusion_corr default
# to the dataset config (True); arms toggle them off explicitly.
# K=1
submit fr_reg_k1     stereobj-1m stereo cross_view 1 "$PRETRAINED_MONO_K1" \
  "" "" "$FREEZE_FLAGS --no-clf_fusion --no-fusion_corr"
submit fr_regclf_k1  stereobj-1m stereo cross_view 1 "$PRETRAINED_MONO_K1" \
  "" "" "$FREEZE_FLAGS --no-fusion_corr"
submit fr_regcorr_k1 stereobj-1m stereo cross_view 1 "$PRETRAINED_MONO_K1" \
  "" "" "$FREEZE_FLAGS --no-clf_fusion"
submit fr_full_k1    stereobj-1m stereo cross_view 1 "$PRETRAINED_MONO_K1" \
  "" "" "$FREEZE_FLAGS"
# K=3
# submit fr_reg_k3     stereobj-1m stereo cross_view 3 "$PRETRAINED_MONO_K3" \
#   "" "" "$FREEZE_FLAGS --no-clf_fusion --no-fusion_corr"
# submit fr_regclf_k3  stereobj-1m stereo cross_view 3 "$PRETRAINED_MONO_K3" \
#   "" "" "$FREEZE_FLAGS --no-fusion_corr"
# submit fr_regcorr_k3 stereobj-1m stereo cross_view 3 "$PRETRAINED_MONO_K3" \
#   "" "" "$FREEZE_FLAGS --no-clf_fusion"
# submit fr_full_k3    stereobj-1m stereo cross_view 3 "$PRETRAINED_MONO_K3" \
#   "" "" "$FREEZE_FLAGS"

# --- Fusion-mode baselines (frozen K=1; reg axis only) ----------------------
# left_only = honest no-reg-fusion control; mean = fixed averaging. Both
# bypass stereo_fuse entirely (only the regressor trains).
submit fm_left_k1 stereobj-1m stereo left_only 1 "$PRETRAINED_MONO_K1" \
  "" "" "$FREEZE_FLAGS --no-clf_fusion --no-fusion_corr"
submit fm_mean_k1 stereobj-1m stereo mean      1 "$PRETRAINED_MONO_K1" \
  "" "" "$FREEZE_FLAGS --no-clf_fusion --no-fusion_corr"

# --- Usprobe corollary (winner-dependent; fill in before submitting) --------
# Train the winning arm's config on usprobe from the usprobe mono source,
# same frozen protocol; eval via inference_usprobe_stereo.py.
#submit fr_<winner>_usprobe usprobe stereo cross_view <K> \
#  "<usprobe mono checkpoint>" usprobe "$USPROBE_SRC" "$FREEZE_FLAGS"

echo "Done. Track with: squeue -u \$USER"
