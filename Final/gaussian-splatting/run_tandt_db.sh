#!/usr/bin/env bash
# Batch runner for Tanks & Temples + Deep Blending (4 scenes).
#
# Usage:
#   ./run_tandt_db.sh
#
# Optional environment variables:
#   DATA_ROOT=/home/arthur/storage/tandt_db
#   OUTPUT_ROOT=/home/arthur/storage/tandt_db/output
#   ITERATIONS=30000
#   SKIP_TRAIN=1          # passed through to each scene
#   RENDER_TRAIN=1        # passed through to each scene

set -eo pipefail

GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/arthur/storage/tandt_db}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/output}"
ITERATIONS="${ITERATIONS:-30000}"

mkdir -p "$OUTPUT_ROOT"

echo "=============================================="
echo " Tanks & Temples + Deep Blending — 4 scenes"
echo "=============================================="
echo " Data root:   $DATA_ROOT"
echo " Output root: $OUTPUT_ROOT"
echo " Iterations:  $ITERATIONS"
echo "=============================================="

run_scene() {
    local name="$1"
    local source="$2"
    local profile="$3"
    local output="$OUTPUT_ROOT/$name"

    echo ""
    echo "##############################################"
    echo " Scene: $name  (profile=$profile)"
    echo "##############################################"

    SCENE_PROFILE="$profile" \
        "$GS_ROOT/run_high_quality_colmap.sh" "$source" "$output" "$ITERATIONS"
}

# Tanks & Temples
run_scene truck     "$DATA_ROOT/tandt/truck"     tnt_outdoor
run_scene train     "$DATA_ROOT/tandt/train"     tnt_outdoor

# Deep Blending
run_scene drjohnson "$DATA_ROOT/db/drjohnson"    tnt_indoor
run_scene playroom  "$DATA_ROOT/db/playroom"     tnt_indoor

echo ""
echo "=============================================="
echo " All 4 scenes complete!"
echo "=============================================="
echo " Outputs:"
echo "   $OUTPUT_ROOT/truck/"
echo "   $OUTPUT_ROOT/train/"
echo "   $OUTPUT_ROOT/drjohnson/"
echo "   $OUTPUT_ROOT/playroom/"
echo "=============================================="
