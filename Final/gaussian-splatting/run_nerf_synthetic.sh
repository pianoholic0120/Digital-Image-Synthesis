#!/usr/bin/env bash
# Batch runner for all 8 NeRF Synthetic scenes.
#
# Usage:
#   ./run_nerf_synthetic.sh
#
# Optional environment variables:
#   DATA_ROOT=/home/arthur/storage/nerf_synthetic
#   OUTPUT_ROOT=/home/arthur/storage/nerf_synthetic/output
#   ITERATIONS=30000
#   SKIP_TRAIN=1 / SKIP_PRUNE=1 / RENDER_TRAIN=1

set -eo pipefail

GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/arthur/storage/nerf_synthetic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/output}"
ITERATIONS="${ITERATIONS:-30000}"

SCENES=(chair drums ficus hotdog lego materials mic ship)

mkdir -p "$OUTPUT_ROOT"

echo "=============================================="
echo " NeRF Synthetic — ${#SCENES[@]} scenes"
echo "=============================================="
echo " Data root:   $DATA_ROOT"
echo " Output root: $OUTPUT_ROOT"
echo " Iterations:  $ITERATIONS"
echo "=============================================="

for name in "${SCENES[@]}"; do
    source="$DATA_ROOT/$name"
    output="$OUTPUT_ROOT/$name"
    profile="synthetic"
    if [[ "$name" == "lego" || "$name" == "materials" ]]; then
        profile="synthetic_edge"
    fi

    echo ""
    echo "##############################################"
    echo " Scene: $name  (profile=$profile)"
    echo "##############################################"

    if [[ ! -f "$source/transforms_train.json" ]]; then
        echo "ERROR: Missing $source/transforms_train.json — skip"
        exit 1
    fi

    SCENE_PROFILE="$profile" \
        "$GS_ROOT/run_high_quality.sh" "$source" "$output" "$ITERATIONS"
done

echo ""
echo "=============================================="
echo " All ${#SCENES[@]} NeRF Synthetic scenes complete!"
echo "=============================================="
for name in "${SCENES[@]}"; do
    echo "   $OUTPUT_ROOT/$name/"
done
echo "=============================================="
