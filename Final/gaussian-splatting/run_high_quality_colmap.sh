#!/usr/bin/env bash
# High-quality 3D Gaussian Splatting pipeline for COLMAP datasets
# (Tanks & Temples, Deep Blending, MipNeRF360, etc.)
#
# Usage:
#   ./run_high_quality_colmap.sh <source_path> <output_path> [iterations]
#
# Example:
#   ./run_high_quality_colmap.sh \
#     /home/arthur/storage/tandt_db/tandt/truck \
#     /home/arthur/storage/tandt_db/output/truck
#
# Optional environment variables:
#   CONDA_ENV=gaussian_splatting   # auto-activate conda env if set
#   SKIP_TRAIN=1                   # skip training, only render + metrics
#   RENDER_TRAIN=1                 # also render train split (default: test only)
#   IMAGES_SUBDIR=images           # image folder name (default: images)
#   SKIP_PRUNE=1                   # skip post-training PLY prune
#   SCENE_PROFILE=tnt_outdoor      # override: tnt_indoor | tnt_outdoor

set -eo pipefail

SOURCE_PATH="${1:-}"
OUTPUT_PATH="${2:-}"
ITERATIONS="${3:-30000}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images}"

usage() {
    echo "Usage: $0 <source_path> <output_path> [iterations]"
    echo ""
    echo "  source_path   COLMAP dataset directory (must contain sparse/0/ and images/)"
    echo "  output_path   Where to save the trained model and renders"
    echo "  iterations    Training iterations (default: 30000)"
    echo ""
    echo "Environment variables:"
    echo "  CONDA_ENV       Conda environment name (default: gaussian_splatting)"
    echo "  SKIP_TRAIN=1    Skip training, only render + metrics"
    echo "  RENDER_TRAIN=1  Also render the train split"
    echo "  IMAGES_SUBDIR   Image subdirectory (default: images)"
    exit 1
}

[[ -z "$SOURCE_PATH" || -z "$OUTPUT_PATH" ]] && usage

GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$GS_ROOT/scripts/train_profiles.sh"
SOURCE_PATH="$(realpath "$SOURCE_PATH")"
SCENE_NAME="$(basename "$SOURCE_PATH")"
mkdir -p "$OUTPUT_PATH"
OUTPUT_PATH="$(realpath "$OUTPUT_PATH")"

# --- conda ---
if command -v conda &>/dev/null && [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        conda activate "$CONDA_ENV"
    fi
fi

PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not found. Activate the gaussian_splatting conda env first."
    exit 1
fi

echo "=============================================="
echo " 3DGS — COLMAP High Quality Pipeline"
echo "=============================================="
echo " Source:     $SOURCE_PATH"
echo " Output:     $OUTPUT_PATH"
echo " Iterations: $ITERATIONS"
echo " Images dir: $IMAGES_SUBDIR"
echo " GS root:    $GS_ROOT"
echo " Python:     $($PYTHON --version 2>&1) ($(which $PYTHON))"
echo "=============================================="

# --- validate COLMAP layout ---
if [[ ! -d "$SOURCE_PATH/sparse/0" ]]; then
    echo "ERROR: Missing sparse/0/ in $SOURCE_PATH"
    echo "  Expected COLMAP structure: sparse/0/{cameras,images,points3D}.bin + images/"
    exit 1
fi
if [[ ! -d "$SOURCE_PATH/$IMAGES_SUBDIR" ]]; then
    echo "ERROR: Missing $IMAGES_SUBDIR/ in $SOURCE_PATH"
    exit 1
fi
echo "[INFO] COLMAP dataset detected → --eval, black background (no -w)"

# --- verify rasterizer ---
RASTERIZER=$("$PYTHON" -c "import diff_gaussian_rasterization; print(diff_gaussian_rasterization.__file__)" 2>/dev/null || true)
if [[ -z "$RASTERIZER" ]]; then
    echo "ERROR: diff_gaussian_rasterization not installed."
    echo "  cd $GS_ROOT && $PYTHON -m pip install submodules/diff-gaussian-rasterization submodules/simple-knn"
    exit 1
fi
if [[ "$RASTERIZER" == *"Endoscopic"* ]]; then
    echo "ERROR: Wrong rasterizer detected: $RASTERIZER"
    echo "  Install the official one: $PYTHON -m pip install $GS_ROOT/submodules/diff-gaussian-rasterization"
    exit 1
fi
echo "[INFO] Rasterizer: $RASTERIZER"

# --- build args (matches full_eval.py settings for tandt/db) ---
TRAIN_ARGS=(
    -s "$SOURCE_PATH"
    -m "$OUTPUT_PATH"
    -i "$IMAGES_SUBDIR"
    --iterations "$ITERATIONS"
    --eval
    --disable_viewer
    --antialiasing
    --test_iterations 7000 "$ITERATIONS"
    --save_iterations 7000 "$ITERATIONS"
)

apply_colmap_profile "$SCENE_NAME"

# --- training ---
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    echo ""
    echo ">>> [1/4] Training ($ITERATIONS iterations)..."
    cd "$GS_ROOT"
    "$PYTHON" train.py "${TRAIN_ARGS[@]}"
else
    echo ""
    echo ">>> [1/4] Training skipped (SKIP_TRAIN=1)"
    PRUNE_PROFILE="${SCENE_PROFILE:-tnt_indoor}"
fi

# --- post-prune ---
if [[ "${SKIP_PRUNE:-0}" != "1" ]]; then
    run_post_prune "$OUTPUT_PATH" "$ITERATIONS" "${PRUNE_PROFILE:-tnt_indoor}"
else
    echo ""
    echo ">>> [2/4] Post-prune skipped (SKIP_PRUNE=1)"
fi

# --- render ---
echo ""
echo ">>> [3/4] Rendering test set (iteration $ITERATIONS)..."
RENDER_ARGS=(
    -s "$SOURCE_PATH"
    -m "$OUTPUT_PATH"
    -i "$IMAGES_SUBDIR"
    --iteration "$ITERATIONS"
    --antialiasing
)
if [[ "${RENDER_TRAIN:-0}" != "1" ]]; then
    RENDER_ARGS+=(--skip_train)
fi

cd "$GS_ROOT"
"$PYTHON" render.py "${RENDER_ARGS[@]}"

# --- metrics (PSNR / SSIM / LPIPS) ---
echo ""
echo ">>> [4/4] Computing metrics..."
cd "$GS_ROOT"
"$PYTHON" metrics.py -m "$OUTPUT_PATH"

# --- summary ---
echo ""
echo "=============================================="
echo " Done!"
echo "=============================================="
echo " Model:       $OUTPUT_PATH/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
echo " Pruned:      $OUTPUT_PATH/point_cloud/iteration_$ITERATIONS/point_cloud_pruned.ply"
echo " Renders:     $OUTPUT_PATH/test/ours_$ITERATIONS/renders/"
echo " GT:          $OUTPUT_PATH/test/ours_$ITERATIONS/gt/"
echo " Metrics:     $OUTPUT_PATH/results.json"
echo " Per-view:    $OUTPUT_PATH/per_view.json"
echo "=============================================="
