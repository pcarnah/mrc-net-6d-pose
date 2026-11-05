#!/bin/bash
#SBATCH --job-name=mrcnet-train
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --mem=60G
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --mail-user=pcarnah@uwo.ca
#SBATCH --mail-type=ALL

# Generic MRC-Net training job.  All configuration is passed through the
# environment so `submit_training.sh` can fan out the matrix; every value has a
# sensible default.  Submit directly with:
#
#   RUN_NAME=stereo_cross_K3 DATASET=stereobj-1m STEREO_MODE=stereo \
#     FUSION=cross_view N_REFINE=3 N_GPUS=1 sbatch scripts/slurm/train.sbatch
#
# One job can run several training configs concurrently on the same GPU; MPS
# (Multi-Process Service) is started automatically so the runs share the GPU's
# SMs instead of time-slicing at the kernel boundary.  Set N_RUNS>1 and give
# each run its own RUN<i>_* variables (the unsuffixed variables are the run-0
# defaults):
#
#   N_RUNS=2 BATCH_SIZE=24 \
#     RUN0_NAME=uf_left_k1 RUN0_FUSION=left_only RUN0_EXTRA_ARGS="--no-clf_fusion" \
#     RUN1_NAME=uf_mean_k1 RUN1_FUSION=mean      RUN1_EXTRA_ARGS="--no-clf_fusion" \
#     sbatch scripts/slurm/train.sbatch
#
# Each run writes to its own logs/slurm/<jobname>-<jobid>-run<i>.out and its own
# checkpoint/log dir.  MPS is on iff N_RUNS>1 by default (MPS=0 disables it,
# MPS=1 forces it on for a single run); MPS_THREAD_PCT caps each client's SM
# share (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE).  The daemon is started once for the
# job and stopped on exit; if one is already running on the node the job keeps
# going without starting a second one.
#
# The job builds a per-job venv on $SLURM_TMPDIR and stages the dataset there
# (see scripts/slurm/setup_env.sh).  stereobj-1m is ~24h/epoch, so the default
# wall time is the 7-day maximum and checkpoints are written on a step interval
# (plus every epoch) so a timeout/crash does not lose the whole run.
#
# Environment (per run i; unsuffixed values are the run-0 defaults):
#   RUN<i>_NAME          run label (checkpoint/log subdir)     [RUN_NAME|mrcnet]
#   RUN<i>_DATASET       DATASET_CONFIG key                    [DATASET|stereobj-1m]
#   RUN<i>_STEREO_MODE   mono|stereo (empty = dataset default) [STEREO_MODE|stereo]
#   RUN<i>_FUSION        cross_view|mean|left_only|right_only [FUSION|cross_view]
#   RUN<i>_N_REFINE      refinement iterations K               [N_REFINE|3]
#   RUN<i>_N_EPOCHS      epochs for this run                   [N_EPOCHS|3]
#   RUN<i>_PRETRAINED    checkpoint path for init (empty=fresh) [PRETRAINED|]
#   RUN<i>_EXTRA_ARGS    extra main.py flags                   [EXTRA_ARGS|]
#   RUN<i>_STAGE_NAME    dataset dir to stage                  [MRC_STAGE_NAME|]
#   RUN<i>_STAGE_SRC     staging source dir                    [MRC_STAGE_SRC|]
#   RUN<i>_CHKPT_DIR     checkpoint dir                        [chkpt_<dataset>_<name>]
#   RUN<i>_LOG_DIR       tensorboard log dir                   [logs/<dataset>/<name>]
# Shared across runs:
#   N_RUNS              number of concurrent runs              [1]
#   BATCH_SIZE          per-run global batch size              [24]
#   NUM_WORKERS         per-run dataloader workers             [8]
#   WARMUP              LR warmup steps (must be < steps/epoch)[1000]
#   N_GPUS              GPUs per run (torchrun DDP if >1)      [1]
#   SAVE_EVERY_STEPS    checkpoint every N optimizer steps     [10000]
#   SAVE_EVERY_EPOCHS   checkpoint every N epochs              [1]
#   BF16                1 to train with bf16 autocast          [1]
#   MPS                 auto|0|1 (auto = on iff N_RUNS>1)      [auto]
#   MPS_THREAD_PCT      CUDA_MPS_ACTIVE_THREAD_PERCENTAGE      [unset]
#   MRC_VENV            pre-built venv to reuse (else $SLURM_TMPDIR/env)
#   MRC_CONDA_ENV       conda env to activate instead of a venv
#   MRC_DATASET_ROOT / MRC_EVAL_ROOT  data/eval roots  [default $SLURM_TMPDIR]
#   TORCH_HOME          torch hub cache dir  [/home/pcarnah/scratch/torch_hub]

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

N_RUNS="${N_RUNS:-1}"

# Resolve a per-run variable: RUN<i>_<SUFFIX> if set, else the fallback.
get_run_var() {
  local i="$1" suffix="$2" fallback="$3"
  local name="RUN${i}_${suffix}"
  local val="${!name:-}"
  if [ -n "$val" ]; then printf '%s' "$val"; else printf '%s' "$fallback"; fi
}

# Data/eval roots: an explicit MRC_* / DATASET_ROOT / EVAL_ROOT wins, else use
# the per-job staging dir $SLURM_TMPDIR when present.  (config.py also accepts
# --dataset_root / --eval_root on the command line.)
export MRC_DATASET_ROOT="${MRC_DATASET_ROOT:-${DATASET_ROOT:-$SLURM_TMPDIR}}"
export MRC_EVAL_ROOT="${MRC_EVAL_ROOT:-${EVAL_ROOT:-$SLURM_TMPDIR}}"
# Run 0's dataset drives the shared staging in setup_env.sh.
export MRC_DATASET="$(get_run_var 0 DATASET "${DATASET:-stereobj-1m}")"
export MRC_STAGE_NAME="${RUN0_STAGE_NAME:-${MRC_STAGE_NAME:-}}"
export MRC_STAGE_SRC="${RUN0_STAGE_SRC:-${MRC_STAGE_SRC:-}}"

# Modules, per-job venv (requirements-cluster.txt + the pytorch3d wheel) and the
# background dataset staging copy.
# shellcheck source=scripts/slurm/setup_env.sh
source scripts/slurm/setup_env.sh

# Shared defaults.
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WARMUP="${WARMUP:-1000}"
N_GPUS="${N_GPUS:-1}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-10000}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
BF16="${BF16:-1}"

# --- Resolve every run's configuration -------------------------------------
R_NAME=(); R_DATASET=(); R_STEREO=(); R_FUSION=(); R_K=(); R_EPOCHS=()
R_PRETRAINED=(); R_STAGE_NAME=(); R_STAGE_SRC=(); R_EXTRA=(); R_CHKPT=(); R_LOG=()
for i in $(seq 0 $((N_RUNS - 1))); do
  R_NAME[$i]="$(get_run_var "$i" NAME "${RUN_NAME:-mrcnet}")"
  R_DATASET[$i]="$(get_run_var "$i" DATASET "${DATASET:-stereobj-1m}")"
  R_STEREO[$i]="$(get_run_var "$i" STEREO_MODE "${STEREO_MODE:-stereo}")"
  R_FUSION[$i]="$(get_run_var "$i" FUSION "${FUSION:-cross_view}")"
  R_K[$i]="$(get_run_var "$i" N_REFINE "${N_REFINE:-3}")"
  R_EPOCHS[$i]="$(get_run_var "$i" N_EPOCHS "${N_EPOCHS:-3}")"
  R_PRETRAINED[$i]="$(get_run_var "$i" PRETRAINED "${PRETRAINED:-}")"
  R_STAGE_NAME[$i]="$(get_run_var "$i" STAGE_NAME "${MRC_STAGE_NAME:-}")"
  R_STAGE_SRC[$i]="$(get_run_var "$i" STAGE_SRC "${MRC_STAGE_SRC:-}")"
  R_EXTRA[$i]="$(get_run_var "$i" EXTRA_ARGS "${EXTRA_ARGS:-}")"
  R_CHKPT[$i]="$(get_run_var "$i" CHKPT_DIR "${CHKPT_DIR:-chkpt_${R_DATASET[$i]}_${R_NAME[$i]}}")"
  R_LOG[$i]="$(get_run_var "$i" LOG_DIR "${LOG_DIR:-logs/${R_DATASET[$i]}/${R_NAME[$i]}}")"
done

mkdir -p logs/slurm

# --- Stage any extra per-run datasets (setup_env staged run 0's) ------------
STAGE_PIDS=()
for i in $(seq 0 $((N_RUNS - 1))); do
  sn="${R_STAGE_NAME[$i]}"; ss="${R_STAGE_SRC[$i]}"
  if [ -n "$sn" ] && [ ! -d "$MRC_DATASET_ROOT/$sn" ] && [ -d "$ss" ]; then
    echo "staging $ss -> $MRC_DATASET_ROOT/$sn (background)"
    cp -r "$ss" "$MRC_DATASET_ROOT/" &
    STAGE_PIDS+=($!)
  fi
done
for pid in "${STAGE_PIDS[@]:-}"; do
  [ -n "$pid" ] && wait "$pid"
done

# --- MPS: share the GPU's SMs between the concurrent runs -------------------
MPS="${MPS:-auto}"
if [ "$MPS" = "auto" ]; then
  if [ "$N_RUNS" -gt 1 ]; then MPS=1; else MPS=0; fi
fi
MPS_STARTED=0
if [ "$MPS" = "1" ]; then
  export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
  export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if [ -n "${MPS_THREAD_PCT:-}" ]; then
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$MPS_THREAD_PCT"
  fi
  if nvidia-cuda-mps-control -d 2>/dev/null; then
    MPS_STARTED=1
    echo "MPS control daemon started (pipe=$CUDA_MPS_PIPE_DIRECTORY)"
    sleep 2
  else
    echo "WARNING: could not start MPS daemon; continuing without it (one may" \
         "already be running on this node)" >&2
  fi
fi
cleanup() {
  if [ "$MPS_STARTED" = "1" ]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# --- Launch every run concurrently -----------------------------------------
launch_run() {
  local i="$1"
  local run_name="${R_NAME[$i]}" dataset="${R_DATASET[$i]}"
  local stereo_mode="${R_STEREO[$i]}" fusion="${R_FUSION[$i]}"
  local n_refine="${R_K[$i]}" n_epochs="${R_EPOCHS[$i]}"
  local pretrained="${R_PRETRAINED[$i]}" extra_args="${R_EXTRA[$i]}"
  local chkpt_dir="${R_CHKPT[$i]}" log_dir="${R_LOG[$i]}"

  mkdir -p "$chkpt_dir" "$log_dir"
  local run_log="logs/slurm/${SLURM_JOB_NAME:-mrcnet}-${SLURM_JOB_ID:-local}-run${i}.out"
  echo "run[$i] $run_name: dataset=$dataset stereo=$stereo_mode fusion=$fusion K=$n_refine epochs=$n_epochs -> $run_log"

  local stereo_arg="" fusion_arg="" pretrained_arg="" bf16_arg=""
  [ -n "$stereo_mode" ] && stereo_arg="--stereo_mode $stereo_mode"
  [ -n "$fusion" ] && fusion_arg="--fusion $fusion"
  [ -n "$pretrained" ] && pretrained_arg="--pretrained $pretrained"
  [ "$BF16" = "1" ] && bf16_arg="--bf16"

  {
    if [ "$N_GPUS" -gt 1 ]; then
      torchrun --nproc_per_node="$N_GPUS" main.py \
        --is_parallel \
        --dataset "$dataset" $stereo_arg $fusion_arg \
        --n_refine_iters "$n_refine" --n_epochs "$n_epochs" \
        --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --warmup_step "$WARMUP" --chkpt_dir "$chkpt_dir" --log_dir "$log_dir" \
        --save_every_steps "$SAVE_EVERY_STEPS" \
        --save_every_epochs "$SAVE_EVERY_EPOCHS" $bf16_arg \
        $pretrained_arg $extra_args
    else
      python main.py \
        --dataset "$dataset" $stereo_arg $fusion_arg \
        --n_refine_iters "$n_refine" --n_epochs "$n_epochs" \
        --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --warmup_step "$WARMUP" --chkpt_dir "$chkpt_dir" --log_dir "$log_dir" \
        --save_every_steps "$SAVE_EVERY_STEPS" \
        --save_every_epochs "$SAVE_EVERY_EPOCHS" $bf16_arg \
        --no_compile \
        $pretrained_arg $extra_args
    fi
  } >"$run_log" 2>&1
}

PIDS=()
for i in $(seq 0 $((N_RUNS - 1))); do
  launch_run "$i" &
  PIDS+=($!)
done

STATUS=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || STATUS=$?
done
exit "$STATUS"
