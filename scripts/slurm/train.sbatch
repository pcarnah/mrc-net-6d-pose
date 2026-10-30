#!/bin/bash
#SBATCH --job-name=mrcnet-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Generic MRC-Net training job.  All configuration is passed through the
# environment so `submit_training.sh` can fan out the matrix; every value has a
# sensible default.  Submit directly with:
#
#   RUN_NAME=stereo_cross_K3 DATASET=stereobj-1m STEREO_MODE=stereo \
#     FUSION=cross_view N_REFINE=3 N_GPUS=1 sbatch scripts/slurm/train.sbatch
#
# Environment:
#   RUN_NAME     run label (checkpoint/log subdir)          [mrcnet]
#   DATASET      DATASET_CONFIG key                         [stereobj-1m]
#   STEREO_MODE  mono|stereo (empty = dataset default)      [stereo]
#   FUSION       cross_view|mean|left_only|right_only       [cross_view]
#   N_REFINE     refinement iterations K                    [3]
#   N_EPOCHS     epochs                                     [75]
#   BATCH_SIZE   global batch size                          [16]
#   NUM_WORKERS  dataloader workers                         [8]
#   WARMUP       LR warmup steps (must be < steps/epoch)    [1000]
#   N_GPUS       GPUs on the node (torchrun DDP if >1)      [1]
#   PRETRAINED   checkpoint path for init (empty = fresh)   []
#   EXTRA_ARGS   extra main.py flags (e.g. --no_compile)    []
#   MRC_VENV     path to a venv to activate (optional)      []
#   MRC_CONDA_ENV  conda env name to activate (optional)    []

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

if [ -n "${MRC_VENV:-}" ]; then
  # shellcheck disable=SC1091
  source "${MRC_VENV}/bin/activate"
fi
if [ -n "${MRC_CONDA_ENV:-}" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "${MRC_CONDA_ENV}"
fi

RUN_NAME="${RUN_NAME:-mrcnet}"
DATASET="${DATASET:-stereobj-1m}"
STEREO_MODE="${STEREO_MODE:-stereo}"
FUSION="${FUSION:-cross_view}"
N_REFINE="${N_REFINE:-3}"
N_EPOCHS="${N_EPOCHS:-75}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WARMUP="${WARMUP:-1000}"
N_GPUS="${N_GPUS:-1}"
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

echo "run=$RUN_NAME dataset=$DATASET stereo=$STEREO_MODE fusion=$FUSION K=$N_REFINE gpus=$N_GPUS"

if [ "$N_GPUS" -gt 1 ]; then
  torchrun --nproc_per_node="$N_GPUS" main.py \
    --is_parallel \
    --dataset "$DATASET" $STEREO_ARG $FUSION_ARG \
    --n_refine_iters "$N_REFINE" --n_epochs "$N_EPOCHS" \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --warmup_step "$WARMUP" --chkpt_dir "$CHKPT_DIR" --log_dir "$LOG_DIR" \
    $PRETRAINED_ARG $EXTRA_ARGS
else
  python main.py \
    --dataset "$DATASET" $STEREO_ARG $FUSION_ARG \
    --n_refine_iters "$N_REFINE" --n_epochs "$N_EPOCHS" \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --warmup_step "$WARMUP" --chkpt_dir "$CHKPT_DIR" --log_dir "$LOG_DIR" \
    $PRETRAINED_ARG $EXTRA_ARGS
fi
