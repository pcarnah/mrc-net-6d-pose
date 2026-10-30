#!/bin/bash
#SBATCH --job-name=mrcnet-train
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --time=1:00:00
#SBATCH --mem=60G
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Generic MRC-Net training job.  All configuration is passed through the
# environment so `submit_training.sh` can fan out the matrix; every value has a
# sensible default.  Submit directly with:
#
#   RUN_NAME=stereo_cross_K3 DATASET=stereobj-1m STEREO_MODE=stereo \
#     FUSION=cross_view N_REFINE=3 N_GPUS=1 sbatch scripts/slurm/train.sbatch
#
# The job builds a per-job venv on $SLURM_TMPDIR and stages the dataset there
# (see scripts/slurm/setup_env.sh).  stereobj-1m is ~24h/epoch, so the default
# wall time is the 7-day maximum and checkpoints are written on a step interval
# (plus every epoch) so a timeout/crash does not lose the whole run.
#
# Environment:
#   RUN_NAME            run label (checkpoint/log subdir)          [mrcnet]
#   DATASET             DATASET_CONFIG key                         [stereobj-1m]
#   STEREO_MODE         mono|stereo (empty = dataset default)      [stereo]
#   FUSION              cross_view|mean|left_only|right_only       [cross_view]
#   N_REFINE            refinement iterations K                    [3]
#   N_EPOCHS            epochs                                     [3]
#   BATCH_SIZE          global batch size                          [24]
#   NUM_WORKERS         dataloader workers                         [8]
#   WARMUP              LR warmup steps (must be < steps/epoch)    [1000]
#   N_GPUS              GPUs on the node (torchrun DDP if >1)      [1]
#   SAVE_EVERY_STEPS    checkpoint every N optimizer steps (0=off) [2000]
#   SAVE_EVERY_EPOCHS   checkpoint every N epochs (0=off)          [1]
#   BF16                1 to train with bf16 autocast (renderer fp32) [1]
#   PRETRAINED          checkpoint path for init (empty = fresh)   []
#   EXTRA_ARGS          extra main.py flags (e.g. --no_compile)    []
#   MRC_VENV            pre-built venv to reuse (else $SLURM_TMPDIR/env)
#   MRC_CONDA_ENV       conda env to activate instead of a venv
#   MRC_DATASET_ROOT / MRC_EVAL_ROOT  data/eval roots  [default $SLURM_TMPDIR]
#   MRC_STAGE_SRC / MRC_STAGE_NAME    dataset to stage [stereobj-1m]
#   TORCH_HOME          torch hub cache dir  [/home/pcarnah/scratch/torch_hub]

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

# Data/eval roots: an explicit MRC_* / DATASET_ROOT / EVAL_ROOT wins, else use
# the per-job staging dir $SLURM_TMPDIR when present.  (config.py also accepts
# --dataset_root / --eval_root on the command line.)
export MRC_DATASET_ROOT="${MRC_DATASET_ROOT:-${DATASET_ROOT:-$SLURM_TMPDIR}}"
export MRC_EVAL_ROOT="${MRC_EVAL_ROOT:-${EVAL_ROOT:-$SLURM_TMPDIR}}"
export MRC_DATASET="${DATASET:-stereobj-1m}"

# Modules, per-job venv (requirements-cluster.txt + the pytorch3d wheel) and the
# background dataset staging copy.
# shellcheck source=scripts/slurm/setup_env.sh
source scripts/slurm/setup_env.sh

RUN_NAME="${RUN_NAME:-mrcnet}"
DATASET="${DATASET:-stereobj-1m}"
STEREO_MODE="${STEREO_MODE:-stereo}"
FUSION="${FUSION:-cross_view}"
N_REFINE="${N_REFINE:-3}"
N_EPOCHS="${N_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WARMUP="${WARMUP:-1000}"
N_GPUS="${N_GPUS:-1}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-10000}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
BF16="${BF16:-1}"
PRETRAINED="${PRETRAINED:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CHKPT_DIR="${CHKPT_DIR:-chkpt_${DATASET}_${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-logs/${DATASET}/${RUN_NAME}}"
mkdir -p "$CHKPT_DIR" "$LOG_DIR" logs/slurm

STEREO_ARG=""
[ -n "$STEREO_MODE" ] && STEREO_ARG="--stereo_mode $STEREO_MODE"
FUSION_ARG=""
[ -n "$FUSION" ] && FUSION_ARG="--fusion $FUSION"
PRETRAINED_ARG=""
[ -n "$PRETRAINED" ] && PRETRAINED_ARG="--pretrained $PRETRAINED"
BF16_ARG=""
[ "$BF16" = "1" ] && BF16_ARG="--bf16"

echo "run=$RUN_NAME dataset=$DATASET stereo=$STEREO_MODE fusion=$FUSION K=$N_REFINE gpus=$N_GPUS"

if [ "$N_GPUS" -gt 1 ]; then
  torchrun --nproc_per_node="$N_GPUS" main.py \
    --is_parallel \
    --dataset "$DATASET" $STEREO_ARG $FUSION_ARG \
    --n_refine_iters "$N_REFINE" --n_epochs "$N_EPOCHS" \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --warmup_step "$WARMUP" --chkpt_dir "$CHKPT_DIR" --log_dir "$LOG_DIR" \
    --save_every_steps "$SAVE_EVERY_STEPS" \
    --save_every_epochs "$SAVE_EVERY_EPOCHS" $BF16_ARG \
    $PRETRAINED_ARG $EXTRA_ARGS
else
  python main.py \
    --dataset "$DATASET" $STEREO_ARG $FUSION_ARG \
    --n_refine_iters "$N_REFINE" --n_epochs "$N_EPOCHS" \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --warmup_step "$WARMUP" --chkpt_dir "$CHKPT_DIR" --log_dir "$LOG_DIR" \
    --save_every_steps "$SAVE_EVERY_STEPS" \
    --save_every_epochs "$SAVE_EVERY_EPOCHS" $BF16_ARG \
    --no_compile \
    $PRETRAINED_ARG $EXTRA_ARGS
fi
