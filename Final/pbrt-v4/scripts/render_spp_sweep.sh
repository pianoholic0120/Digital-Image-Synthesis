#!/usr/bin/env bash
# Usage: bash render_spp_sweep.sh scene.pbrt 1 4 16 64 256 1024
set -euo pipefail
SCENE="$1"
shift
PBRT="${PBRT:-build/Release/pbrt}"
for SPP in "$@"; do
  OUT="render_spp${SPP}.exr"
  echo "Rendering ${SCENE} at ${SPP} spp -> ${OUT}"
  "${PBRT}" "${SCENE}" "--pixelsamples=${SPP}" "--outfile=${OUT}"
done
