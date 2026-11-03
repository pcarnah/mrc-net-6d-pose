#!/usr/bin/env bash
# Quick abbreviated stereobj-1m eval: GT-detection inference + scoring on a
# val subset. Writes the BOP CSV and the metrics JSON into EVAL_ROOT.
#
# Usage:
#   scripts/eval_stereobj_subset.sh <ckpt_subdir> <model_name> <stereo_mode> \
#                                   [max_instances] [output_suffix]
#
#   <ckpt_subdir>   directory under CKPT_ROOT, e.g. chkpt_stereobj-1m_mono_K1
#   <model_name>    checkpoint file stem, e.g. epoch_000 or step_00030000
#   <stereo_mode>   mono | stereo   (must match the checkpoint's training mode)
#   [max_instances] val-subset cap, 0 = all (default 500)
#   [output_suffix] extra tag for the output files (default: auto)
#
# Examples:
#   scripts/eval_stereobj_subset.sh chkpt_stereobj-1m_mono_K1 epoch_000 mono 500
#   scripts/eval_stereobj_subset.sh chkpt_stereobj-1m_stereo_cross_K1 epoch_000 stereo 5000
#
# Environment overrides:
#   CKPT_ROOT   checkpoint root           (default: <repo>/stereo_checkpoints)
#   DATA_ROOT   dataset root              (default: /workspaces/datasets)
#   EVAL_ROOT   output dir for CSV/JSON   (default: <repo>/reports)
#   TORCH_HOME  writable torch-hub cache  (default: /tmp/kilo/torch)
#   PYTHON      interpreter               (default: <repo>/.venv/bin/python;
#                                         on Windows use .venv/Scripts/python.exe)
#   CROP        single | multi            (default: single). multi re-enables the
#                                         4-crop confidence argmax selection.
#
# Inference runs with --no_compile always, and --single_crop unless CROP=multi
# (the 4-crop confidence argmax is miscalibrated; see reports/ history).

set -euo pipefail

if [ "$#" -lt 3 ]; then
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_SUBDIR="$1"
MODEL="$2"
MODE="$3"
N="${4:-500}"
SUFFIX="${5:-}"

CKPT_ROOT="${CKPT_ROOT:-$REPO_DIR/stereo_checkpoints}"
DATA_ROOT="${DATA_ROOT:-/workspaces/datasets}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_DIR/reports}"
TORCH_HOME="${TORCH_HOME:-/tmp/kilo/torch}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
CROP="${CROP:-single}"

crop_flags=()
if [ "$CROP" != "multi" ]; then
    crop_flags+=(--single_crop)
fi

ckpt_dir="$CKPT_ROOT/$CKPT_SUBDIR"
if [ ! -d "$ckpt_dir" ]; then
    echo "checkpoint dir not found: $ckpt_dir" >&2
    exit 1
fi

run_tag="${CKPT_SUBDIR#chkpt_stereobj-1m_}"
model_tag="${MODEL%.pth}"
tag="${run_tag}_${model_tag}_n${N}"
if [ -n "$SUFFIX" ]; then
    tag="${tag}_${SUFFIX}"
fi

mkdir -p "$EVAL_ROOT"

echo "== inference: $CKPT_SUBDIR / $MODEL ($MODE, n=$N) =="
cd "$CKPT_ROOT"
TORCH_HOME="$TORCH_HOME" "$PYTHON" "$REPO_DIR/inference_stereobj.py" \
    --dataset stereobj-1m --split val \
    --checkpoint_name "$CKPT_SUBDIR" --model_name "$MODEL" \
    --output_suffix "$tag" --stereo_mode "$MODE" \
    --max_instances "$N" --no_compile "${crop_flags[@]}" \
    --dataset_root "$DATA_ROOT" --eval_root "$EVAL_ROOT"

echo "== scoring =="
cd "$REPO_DIR"
"$PYTHON" "$REPO_DIR/tools/eval_stereobj.py" \
    --results "$EVAL_ROOT/mrcnet_stereobj-1m-val_${CKPT_SUBDIR}_${tag}.csv" \
    --dataset stereobj-1m --split val --stereo_mode "$MODE" \
    --output "$EVAL_ROOT/${tag}.json"

echo "== done: $EVAL_ROOT/${tag}.json =="