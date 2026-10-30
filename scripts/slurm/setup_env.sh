#!/bin/bash
# Shared SLURM environment setup for MRC-Net jobs on the Alliance cluster.
#
# Sourced (not executed) by train.sbatch / infer.sbatch after `cd` to the repo
# root.  It:
#   1. loads the required modules,
#   2. stages the training dataset into node-local scratch in the background,
#   3. builds a per-job virtualenv on $SLURM_TMPDIR and installs the cluster
#      requirements (pytorch3d comes from a locally built wheel, not an index).
#
# Environment:
#   SLURM_TMPDIR     node-local scratch (required; set by SLURM)
#   MRC_DATASET      dataset being staged                 [stereobj-1m]
#   MRC_DATASET_ROOT staging destination root             [$SLURM_TMPDIR]
#   MRC_STAGE_NAME   dataset dir to stage                 [MRC_DATASET]
#   MRC_STAGE_SRC    source dir to copy                   [stereobj-1m default]
#   MRC_VENV         pre-built venv to reuse              [$SLURM_TMPDIR/env]
#   MRC_CONDA_ENV    conda env to activate instead        []
#   MRC_BOP_TOOLKIT  sibling bop_toolkit path to install  []
#   TORCH_HOME       torch hub cache dir                  [/home/pcarnah/scratch/torch_hub]
#
# The dataset copy is launched in the background so it overlaps the (slower)
# venv build; the job waits for it before training starts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${SLURM_TMPDIR:?setup_env.sh requires SLURM_TMPDIR (run under sbatch)}"

module load python/3.12 opencv/4.10 cuda/13.2 cudnn

# Torch hub cache: compute nodes have no internet, so point torch at the
# pre-populated cache on scratch for any pretrained backbone weights.
export TORCH_HOME="${TORCH_HOME:-/home/pcarnah/scratch/torch_hub}"

# --- Stage the dataset into node-local scratch (background) -----------------
# Default to stereobj-1m (the main training dataset); other datasets must pass
# MRC_STAGE_NAME / MRC_STAGE_SRC explicitly.
STAGE_NAME="${MRC_STAGE_NAME:-}"
STAGE_SRC="${MRC_STAGE_SRC:-}"
if [ -z "$STAGE_NAME" ] && [ "${MRC_DATASET:-stereobj-1m}" = "stereobj-1m" ]; then
  STAGE_NAME="stereobj-1m"
  STAGE_SRC="${STAGE_SRC:-$HOME/projects/def-chene/pcarnah/stereobj-1m}"
fi

STAGE_ROOT="${MRC_DATASET_ROOT:-$SLURM_TMPDIR}"
STAGE_PID=""
if [ -n "$STAGE_NAME" ]; then
  if [ -d "$STAGE_ROOT/$STAGE_NAME" ]; then
    echo "dataset already staged: $STAGE_ROOT/$STAGE_NAME"
  elif [ -d "$STAGE_SRC" ]; then
    echo "staging $STAGE_SRC -> $STAGE_ROOT/$STAGE_NAME (background)"
    cp -r "$STAGE_SRC" "$STAGE_ROOT/" &
    STAGE_PID=$!
  else
    echo "WARNING: staging source not found: $STAGE_SRC" >&2
  fi
fi

# --- Per-job environment (venv on node-local scratch) ----------------------
if [ -n "${MRC_CONDA_ENV:-}" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "$MRC_CONDA_ENV"
elif [ -n "${MRC_VENV:-}" ] && [ -x "${MRC_VENV}/bin/python" ]; then
  # shellcheck disable=SC1091
  source "${MRC_VENV}/bin/activate"
else
  VENV="${MRC_VENV:-$SLURM_TMPDIR/env}"
  python -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install --no-index --upgrade pip
  pip install --no-index -r "$REPO_ROOT/requirements-cluster.txt"
  pip install --no-index --no-deps "$HOME"/wheels/pytorch3d-*.whl
fi

# --- Wait for the dataset copy before the job starts ------------------------
if [ -n "$STAGE_PID" ]; then
  echo "waiting for dataset copy (pid $STAGE_PID)"
  wait "$STAGE_PID"
fi

# Optional: the sibling bop_toolkit fork (needed by inference/eval, not by
# training).  Point MRC_BOP_TOOLKIT at its path on the cluster to install it.
if [ -n "${MRC_BOP_TOOLKIT:-}" ]; then
  pip install --no-index "$MRC_BOP_TOOLKIT"
fi

echo "environment ready: python=$(command -v python) dataset=${MRC_DATASET:-}"
