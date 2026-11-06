#!/usr/bin/env bash
# Quick abbreviated stereobj-1m eval: GT-detection inference + scoring on a
# val subset. Writes the BOP CSV and the metrics JSON into EVAL_ROOT.
#
# Usage:
#   scripts/eval_stereobj_subset.sh <ckpt_subdir> <model_name> <stereo_mode> \
#                                   [max_instances] [output_suffix]
#
#   <ckpt_subdir>   directory under CKPT_ROOT, e.g. chkpt_stereobj-1m_mono_K1
#                   (subdirectories like ablation/chkpt_... are allowed; the
#                   output tag uses the basename)
#   <model_name>    checkpoint file stem, e.g. epoch_000 or step_00030000
#   <stereo_mode>   mono | stereo   (must match the checkpoint's training mode)
#   [max_instances] val-subset cap, 0 = all (default 500). The subset is
#                   drawn by stratified sampling across all object types
#                   (proportional quota, min 1 per type) over the common
#                   both-eye population: mono and stereo runs select the
#                   identical instance set for a given seed (mono scores the
#                   left eye), so checkpoints are directly comparable.
#                   STRATIFIED=0 restores the legacy first-frames cap.
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
#   NUM_WORKERS DataLoader workers for the CPU prep (default: 2). Overlaps
#                                         decode + crop building with the GPU
#                                         call; 0 = serial.
#   STRATIFIED  1 | 0                    (default: 1). 0 disables stratified
#                                         sampling (legacy first-N-frames
#                                         subset). The tag gets "_strat" when
#                                         enabled.
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
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
CROP="${CROP:-single}"
NUM_WORKERS="${NUM_WORKERS:-2}"
STRATIFIED="${STRATIFIED:-1}"

crop_flags=()
if [ "$CROP" != "multi" ]; then
    crop_flags+=(--single_crop)
fi
worker_flags=(--num_workers "$NUM_WORKERS")
strat_flags=()
if [ "$STRATIFIED" != "0" ]; then
    strat_flags+=(--stratified)
fi

ckpt_dir="$CKPT_ROOT/$CKPT_SUBDIR"
if [ ! -d "$ckpt_dir" ]; then
    echo "checkpoint dir not found: $ckpt_dir" >&2
    exit 1
fi

run_tag="${CKPT_SUBDIR##*/}"
run_tag="${run_tag#chkpt_stereobj-1m_}"
ckpt_csv="${CKPT_SUBDIR//\//_}"
model_tag="${MODEL%.pth}"
tag="${run_tag}_${model_tag}_n${N}"
if [ "$STRATIFIED" != "0" ]; then
    tag="${tag}_strat"
fi
if [ -n "$SUFFIX" ]; then
    tag="${tag}_${SUFFIX}"
fi

mkdir -p "$EVAL_ROOT"

echo "== inference: $CKPT_SUBDIR / $MODEL ($MODE, n=$N) =="
cd "$CKPT_ROOT"
"$PYTHON" "$REPO_DIR/inference_stereobj.py" \
    --dataset stereobj-1m --split val \
    --checkpoint_name "$CKPT_SUBDIR" --model_name "$MODEL" \
    --output_suffix "$tag" --stereo_mode "$MODE" \
    --max_instances "$N" --no_compile "${crop_flags[@]}" \
    "${worker_flags[@]}" "${strat_flags[@]}" \
    --dataset_root "$DATA_ROOT" --eval_root "$EVAL_ROOT"

echo "== scoring =="
cd "$REPO_DIR"
"$PYTHON" "$REPO_DIR/tools/eval_stereobj.py" \
    --results "$EVAL_ROOT/mrcnet_stereobj-1m-val_${ckpt_csv}_${tag}.csv" \
    --dataset stereobj-1m --split val --stereo_mode "$MODE" \
    --output "$EVAL_ROOT/${tag}.json"

echo "== done: $EVAL_ROOT/${tag}.json =="