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
# Runs per job / MPS:
#   RUNS_PER_JOB=2 packs two matrix cells into one sbatch job; they train
#   concurrently on the same GPU and train.sbatch starts MPS so they share the
#   SMs.  Each run keeps its own batch size (BATCH_SIZE is per run) and its own
#   checkpoint/log dirs.  Grouping is by consecutive `submit` order, so keep
#   runs of similar duration together (a short run finishing early leaves the
#   GPU half idle).  RUNS_PER_JOB=1 = one cell per job.
#   GRES=h100:1 selects the GPU type (passed to sbatch as --gpus); empty uses
#   the train.sbatch directive.  CPUS_PER_TASK/MEM default to 8*RUNS_PER_JOB
#   and 60*RUNS_PER_JOB GB; MPS=0 disables MPS (see train.sbatch).
#
# stereobj-1m is ~24h/epoch, so the default schedule is a single 7-day job
# running N_EPOCHS=3 with checkpoints on a step interval (SAVE_EVERY_STEPS) plus
# every epoch (SAVE_EVERY_EPOCHS=1).  Each job builds its own venv on
# $SLURM_TMPDIR and stages the dataset there (see scripts/slurm/setup_env.sh).
# Training runs with bf16 autocast by default (BF16=1; ~1.2x faster, ~25 percent
# less memory); set BF16=0 to fall back to fp32.
#
# The submitted matrix follows RUN_PLAN.md section 4 (unfrozen primary):
#   mono_K1                    stereobj-1m mono source (3 epochs, MONO_N_EPOCHS)
#   uf_left_k1 / uf_mean_k1    controls: no cross-view pathway / fixed averaging
#   uf_full_k1                 reg residual + clf fusion + pool correlation
#   (reg-only arm already run: chkpt_stereobj-1m_stereo_cross_K1_resid_reg_only_unfrozen)
# K=3 follow-ups and the frozen attribution matrix are commented templates.
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

# Concurrent runs per job (MPS-shared) and the per-job resource envelope.
RUNS_PER_JOB="${RUNS_PER_JOB:-2}"
GRES="${GRES:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * RUNS_PER_JOB))}"
MEM="${MEM:-$((60 * RUNS_PER_JOB))G}"
MPS="${MPS:-auto}"

# Cells accumulated for the next job; flushed every RUNS_PER_JOB submits.
PENDING=()

submit() {
  local run_name="$1" dataset="$2" stereo_mode="$3" fusion="$4" n_refine="$5"
  local pretrained="${6:-}" stage_name="${7:-}" stage_src="${8:-}" \
      extra="${9:-}"
  local n_epochs="${N_EPOCHS}"
  echo "== $run_name (dataset=$dataset stereo=$stereo_mode fusion=$fusion " \
       "K=$n_refine epochs=$n_epochs)"
  PENDING+=("$run_name|$dataset|$stereo_mode|$fusion|$n_refine|$pretrained|$stage_name|$stage_src|$extra|$n_epochs")
  if [ "${#PENDING[@]}" -ge "$RUNS_PER_JOB" ]; then
    flush
  fi
}

flush() {
  local n="${#PENDING[@]}"
  [ "$n" -eq 0 ] && return 0
  local env_args=(
    "N_RUNS=$n" "BATCH_SIZE=$BATCH_SIZE" "NUM_WORKERS=$NUM_WORKERS"
    "WARMUP=$WARMUP" "N_GPUS=$N_GPUS" "BF16=$BF16" "MPS=$MPS"
    "SAVE_EVERY_STEPS=$SAVE_EVERY_STEPS" "SAVE_EVERY_EPOCHS=$SAVE_EVERY_EPOCHS")
  local i=0 cell rn ds sm fu nr pt sn ss ex ne
  for cell in "${PENDING[@]}"; do
    IFS='|' read -r rn ds sm fu nr pt sn ss ex ne <<<"$cell"
    env_args+=(
      "RUN${i}_NAME=$rn" "RUN${i}_DATASET=$ds" "RUN${i}_STEREO_MODE=$sm"
      "RUN${i}_FUSION=$fu" "RUN${i}_N_REFINE=$nr" "RUN${i}_N_EPOCHS=$ne"
      "RUN${i}_EXTRA_ARGS=$EXTRA_ARGS $ex")
    if [ -n "$pt" ]; then env_args+=("RUN${i}_PRETRAINED=$pt"); fi
    if [ -n "$sn" ]; then env_args+=("RUN${i}_STAGE_NAME=$sn"); fi
    if [ -n "$ss" ]; then env_args+=("RUN${i}_STAGE_SRC=$ss"); fi
    i=$((i + 1))
  done
  local sbatch_opts=("--time=$TIME")
  if [ -n "$GRES" ]; then sbatch_opts+=("--gpus=$GRES"); fi
  if [ -n "$CPUS_PER_TASK" ]; then
    sbatch_opts+=("--cpus-per-task=$CPUS_PER_TASK")
  fi
  if [ -n "$MEM" ]; then sbatch_opts+=("--mem=$MEM"); fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "   env ${env_args[*]} $SBATCH ${sbatch_opts[*]} scripts/slurm/train.sbatch"
  else
    env "${env_args[@]}" "$SBATCH" "${sbatch_opts[@]}" scripts/slurm/train.sbatch
  fi
  PENDING=()
}

# Sources: the stereobj-1m mono-K1 run; every primary arm resumes from its
# epoch_002 checkpoint (adjust the file name to the actual save).
PRETRAINED_MONO_K1="${PRETRAINED_MONO_K1:-chkpt_stereobj-1m_mono_K1/epoch_002.pth}"
PRETRAINED_MONO_K3="${PRETRAINED_MONO_K3:-chkpt_stereobj-1m_mono_K3/epoch_001.pth}"

# --- Mono source (dependency for every primary arm; 3 epochs per plan) ------
# Its 3-epoch duration matches no other cell, so submit it as its own job.
N_EPOCHS="${MONO_N_EPOCHS:-3}" RUNS_PER_JOB=1 \
  submit mono_K1 stereobj-1m mono "" 1 ""

# --- Primary matrix (unfrozen, 1 epoch, K=1, resumed from mono ep1) ---------
# See RUN_PLAN.md section 4. reg-only is already run
# (chkpt_stereobj-1m_stereo_cross_K1_resid_reg_only_unfrozen).
# left_only / mean: controls for the stereo-eval operating point and two-view
# data effects (no cross-view regression pathway / fixed averaging).
submit uf_left_k1 stereobj-1m stereo left_only 1 "$PRETRAINED_MONO_K1" \
  "" "" "--no-clf_fusion --no-fusion_corr"
submit uf_mean_k1 stereobj-1m stereo mean 1 "$PRETRAINED_MONO_K1" \
  "" "" "--no-clf_fusion --no-fusion_corr"
# full: reg residual + clf fusion + pool correlation (dataset defaults).
submit uf_full_k1 stereobj-1m stereo cross_view 1 "$PRETRAINED_MONO_K1"

submit uf_reg_k1     stereobj-1m stereo cross_view 1 "$PRETRAINED_MONO_K1" \
 "" "" "--no-clf_fusion --no-fusion_corr"
submit uf_regclf_k1  stereobj-1m stereo cross_view 1 "$PRETRAINED_MONO_K1" \
 "" "" "--no-fusion_corr"
submit uf_regcorr_k1 stereobj-1m stereo cross_view 1 "$PRETRAINED_MONO_K1" \
 "" "" "--no-clf_fusion"

# Flush any partial group left over from the matrix above.
flush

# --- K=3 follow-ups (after the K=1 verdict) ---------------------------------
#submit uf_left_k3  stereobj-1m stereo left_only  3 "$PRETRAINED_MONO_K3" \
#  "" "" "--no-clf_fusion --no-fusion_corr"
#submit uf_mean_k3  stereobj-1m stereo mean       3 "$PRETRAINED_MONO_K3" \
#  "" "" "--no-clf_fusion --no-fusion_corr"
#submit uf_full_k3  stereobj-1m stereo cross_view 3 "$PRETRAINED_MONO_K3"

# --- Frozen matrix (demoted: optional cheap per-component attribution) ------
# Caveat: frozen results conflate freezing with pathway count. See RUN_PLAN 4.
#submit fr_reg_k1     stereobj-1m stereo cross_view 1 "$PRETRAINED_MONO_K1" \
#  "" "" "--freeze_except_stereo --no-clf_fusion --no-fusion_corr"
#submit fr_regclf_k1  stereobj-1m stereo cross_view 1 "$PRETRAINED_MONO_K1" \
#  "" "" "--freeze_except_stereo --no-fusion_corr"
#submit fr_regcorr_k1 stereobj-1m stereo cross_view 1 "$PRETRAINED_MONO_K1" \
#  "" "" "--freeze_except_stereo --no-clf_fusion"
#submit fr_full_k1    stereobj-1m stereo cross_view 1 "$PRETRAINED_MONO_K1" \
#  "" "" "--freeze_except_stereo"

# --- Usprobe corollary (winner-dependent; fill in before submitting) --------
# Train the winning arm's config on usprobe from the usprobe mono source;
# eval via inference_usprobe_stereo.py.
#submit uf_<winner>_usprobe usprobe stereo cross_view <K> \
#  "<usprobe mono checkpoint>" usprobe "$USPROBE_SRC"

echo "Done. Track with: squeue -u \$USER"
