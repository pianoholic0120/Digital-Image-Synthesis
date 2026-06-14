#!/usr/bin/env bash
# High-quality 3D Gaussian Splatting training + render pipeline.
#
# Usage:
#   ./run_high_quality.sh <source_path> <output_path> [iterations]
#
# Example:
#   ./run_high_quality.sh \
#     /home/arthur/storage/nerf_synthetic/lego \
#     /home/arthur/storage/nerf_synthetic/output/lego
#
# Optional environment variables:
#   CONDA_ENV=gaussian_splatting   # auto-activate conda env if set
#   SKIP_TRAIN=1                   # skip training, only render/composite
#   RENDER_TRAIN=1                 # also render train split (default: test only)
#   SKIP_PRUNE=1                   # skip post-training PLY prune
#   SCENE_PROFILE=synthetic_edge   # override: synthetic | synthetic_edge

set -eo pipefail

SOURCE_PATH="${1:-}"
OUTPUT_PATH="${2:-}"
ITERATIONS="${3:-30000}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"

usage() {
    echo "Usage: $0 <source_path> <output_path> [iterations]"
    echo ""
    echo "  source_path   Dataset directory (NeRF Synthetic or COLMAP)"
    echo "  output_path   Where to save the trained model and renders"
    echo "  iterations    Training iterations (default: 30000)"
    echo ""
    echo "Environment variables:"
    echo "  CONDA_ENV      Conda environment name (default: gaussian_splatting)"
    echo "  SKIP_TRAIN=1   Skip training, only render + composite"
    echo "  RENDER_TRAIN=1 Also render/composite the train split"
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

# --- conda (skip if already in the target env; conda hooks break under `set -u`) ---
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
echo " 3D Gaussian Splatting — High Quality Pipeline"
echo "=============================================="
echo " Source:     $SOURCE_PATH"
echo " Output:     $OUTPUT_PATH"
echo " Iterations: $ITERATIONS"
echo " GS root:    $GS_ROOT"
echo " Python:     $($PYTHON --version 2>&1) ($(which $PYTHON))"
echo "=============================================="

# --- detect dataset type ---
IS_SYNTHETIC=false
if [[ -f "$SOURCE_PATH/transforms_train.json" ]]; then
    IS_SYNTHETIC=true
    echo "[INFO] NeRF Synthetic / Blender dataset detected → --eval"
else
    echo "[INFO] COLMAP-style dataset detected"
fi

# --- verify rasterizer ---
RASTERIZER=$("$PYTHON" -c "import diff_gaussian_rasterization; print(diff_gaussian_rasterization.__file__)" 2>/dev/null || true)
if [[ -z "$RASTERIZER" ]]; then
    echo "ERROR: diff_gaussian_rasterization not installed."
    echo "  cd $GS_ROOT && $PYTHON -m pip install submodules/diff-gaussian-rasterization submodules/simple-knn"
    exit 1
fi
if [[ "$RASTERIZER" == *"Endoscopic"* ]]; then
    echo "WARNING: Wrong rasterizer detected: $RASTERIZER"
    echo "  Install the official one: $PYTHON -m pip install $GS_ROOT/submodules/diff-gaussian-rasterization"
    exit 1
fi
echo "[INFO] Rasterizer: $RASTERIZER"

# --- build train args ---
TRAIN_ARGS=(
    -s "$SOURCE_PATH"
    -m "$OUTPUT_PATH"
    --iterations "$ITERATIONS"
    --disable_viewer
    --antialiasing
    --test_iterations 7000 "$ITERATIONS"
    --save_iterations 7000 "$ITERATIONS"
)

if $IS_SYNTHETIC; then
    TRAIN_ARGS+=(--eval)
    apply_synthetic_profile "$SCENE_NAME"
else
    apply_colmap_profile "$SCENE_NAME"
fi

# --- training ---
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    echo ""
    echo ">>> [1/4] Training ($ITERATIONS iterations)..."
    cd "$GS_ROOT"
    "$PYTHON" train.py "${TRAIN_ARGS[@]}"
else
    echo ""
    echo ">>> [1/4] Training skipped (SKIP_TRAIN=1)"
    PRUNE_PROFILE="${SCENE_PROFILE:-synthetic}"
fi

# --- post-prune ---
if [[ "${SKIP_PRUNE:-0}" != "1" ]]; then
    run_post_prune "$OUTPUT_PATH" "$ITERATIONS" "${PRUNE_PROFILE:-synthetic}"
else
    echo ""
    echo ">>> [2/4] Post-prune skipped (SKIP_PRUNE=1)"
fi

# --- render ---
echo ""
echo ">>> [3/4] Rendering (iteration $ITERATIONS)..."
RENDER_ARGS=(
    -s "$SOURCE_PATH"
    -m "$OUTPUT_PATH"
    --iteration "$ITERATIONS"
    --antialiasing
)
if [[ "${RENDER_TRAIN:-0}" != "1" ]]; then
    RENDER_ARGS+=(--skip_train)
fi

cd "$GS_ROOT"
"$PYTHON" render.py "${RENDER_ARGS[@]}"

# --- alpha composite (clean backgrounds for synthetic data) ---
echo ""
echo ">>> [4/4] Alpha-compositing renders..."
COMPOSITE_ARGS=(
    -m "$OUTPUT_PATH"
    --iteration "$ITERATIONS"
    --antialiasing
)
if [[ "${RENDER_TRAIN:-0}" != "1" ]]; then
    COMPOSITE_ARGS+=(--skip_train)
fi

cd "$GS_ROOT"
"$PYTHON" scripts/composite_renders.py "${COMPOSITE_ARGS[@]}"

# --- summary ---
echo ""
echo "=============================================="
echo " Done!"
echo "=============================================="
echo " Model:              $OUTPUT_PATH/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
echo " Pruned model:       $OUTPUT_PATH/point_cloud/iteration_$ITERATIONS/point_cloud_pruned.ply"
echo " Raw renders:        $OUTPUT_PATH/test/ours_$ITERATIONS/renders/"
echo " Composited renders: $OUTPUT_PATH/test/ours_$ITERATIONS/composited/renders/"
echo " Composited GT:      $OUTPUT_PATH/test/ours_$ITERATIONS/composited/gt/"
echo " Metrics:            $OUTPUT_PATH/metrics_composited_$ITERATIONS.json"
echo ""
echo " Tip: renders/ should now be clean directly (upstream #1038 fix)."
echo "       composited/ is kept for metrics comparison."
echo "=============================================="
