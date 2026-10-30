#!/bin/bash
#SBATCH --job-name=mrcnet-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Stereobj-1m inference (GT-detection mode) for one checkpoint / one test-time
# step count.  Writes a BOP-format CSV under $EVAL_ROOT and, optionally, runs
# the ADD(-S)/projection evaluator on it.
#
# Environment:
#   RUN_NAME       run label (matches the training run)      [mrcnet]
#   DATASET        DATASET_CONFIG key                        [stereobj-1m]
#   SPLIT          val|test                                  [val]
#   STEREO_MODE    mono|stereo (empty = dataset default)     [stereo]
#   FUSION         fusion override (empty = checkpoint's)    []
#   N_REFINE       test-time steps K                         [1]
#   MODEL_NAME     checkpoint file stem (no .pth)            [epoch_074]
#   SUFFIX         CSV output suffix                         [k1]
#   INIT_ROT_NOISE bootstrap rotation noise (deg)            [0]
#   INIT_TRANS_NOISE bootstrap translation noise (frac)      [0]
#   EVAL           1 to run tools/eval_stereobj.py after     [1]
#   EXTRA_ARGS     extra inference flags (e.g. --no_compile) []
#   DATASET_ROOT / EVAL_ROOT  data/eval roots; default $SLURM_TMPDIR  []

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

# Data/eval roots: explicit MRC_* / DATASET_ROOT / EVAL_ROOT wins, else the
# per-job staging dir $SLURM_TMPDIR.  config.py also accepts --dataset_root /
# --eval_root on the command line.
if [ -n "${SLURM_TMPDIR:-}" ]; then
  export MRC_DATASET_ROOT="${MRC_DATASET_ROOT:-${DATASET_ROOT:-$SLURM_TMPDIR}}"
  export MRC_EVAL_ROOT="${MRC_EVAL_ROOT:-${EVAL_ROOT:-$SLURM_TMPDIR}}"
fi

RUN_NAME="${RUN_NAME:-mrcnet}"
DATASET="${DATASET:-stereobj-1m}"
SPLIT="${SPLIT:-val}"
STEREO_MODE="${STEREO_MODE:-stereo}"
FUSION="${FUSION:-}"
N_REFINE="${N_REFINE:-1}"
MODEL_NAME="${MODEL_NAME:-epoch_070}"
SUFFIX="${SUFFIX:-k${N_REFINE}}"
INIT_ROT_NOISE="${INIT_ROT_NOISE:-0}"
INIT_TRANS_NOISE="${INIT_TRANS_NOISE:-0}"
EVAL="${EVAL:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CHKPT_NAME="chkpt_${DATASET}_${RUN_NAME}"
mkdir -p logs/slurm

STEREO_ARG=""
[ -n "$STEREO_MODE" ] && STEREO_ARG="--stereo_mode $STEREO_MODE"
FUSION_ARG=""
[ -n "$FUSION" ] && FUSION_ARG="--fusion $FUSION"

python inference_stereobj.py \
  --dataset "$DATASET" --split "$SPLIT" \
  --checkpoint_name "$CHKPT_NAME" --model_name "$MODEL_NAME" \
  --output_suffix "$SUFFIX" \
  --n_refine_iters "$N_REFINE" \
  --init_rot_noise_deg "$INIT_ROT_NOISE" \
  --init_trans_noise "$INIT_TRANS_NOISE" \
  $STEREO_ARG $FUSION_ARG $EXTRA_ARGS

if [ "$EVAL" = "1" ]; then
  # inference_stereobj.py writes
  #   <EVAL_ROOT>/mrcnet_<dataset>-<split>_<checkpoint_name>_<suffix>.csv
  EVAL_ROOT="${EVAL_ROOT:-${MRC_EVAL_ROOT:-$(python -c 'import config; print(config.EVAL_ROOT)')}}"
  CSV="${EVAL_ROOT}/mrcnet_${DATASET}-${SPLIT}_${CHKPT_NAME}_${SUFFIX}.csv"
  OUT="reports/${RUN_NAME}_${SUFFIX}.eval.json"
  mkdir -p reports
  python tools/eval_stereobj.py \
    --results "$CSV" --dataset "$DATASET" --split "$SPLIT" \
    --stereo_mode "$STEREO_MODE" --output "$OUT"
fi
