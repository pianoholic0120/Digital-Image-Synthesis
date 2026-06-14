#!/usr/bin/env bash
# Retrain ALL scenes: 8 NeRF Synthetic + 4 TnT/DB.
#
# Usage:
#   ./run_all.sh
#
# Recommended: run inside tmux (see below). Total runtime: many hours.
#
# Optional:
#   ITERATIONS=30000
#   ONLY_NERF=1      # skip tandt_db
#   ONLY_TANDT=1     # skip nerf_synthetic

set -eo pipefail

GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START=$(date +%s)

echo "=============================================="
echo " Full retrain: NeRF Synthetic + TnT/DB"
echo " Started: $(date)"
echo "=============================================="

if [[ "${ONLY_TANDT:-0}" != "1" ]]; then
    echo ""
    echo "========== Phase 1/2: NeRF Synthetic (8) =========="
    "$GS_ROOT/run_nerf_synthetic.sh"
fi

if [[ "${ONLY_NERF:-0}" != "1" ]]; then
    echo ""
    echo "========== Phase 2/2: TnT + DB (4) =========="
    "$GS_ROOT/run_tandt_db.sh"
fi

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "=============================================="
echo " All training finished in $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m"
echo " Finished: $(date)"
echo "=============================================="
