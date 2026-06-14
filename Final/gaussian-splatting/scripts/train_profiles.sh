#!/usr/bin/env bash
# Shared training profile helpers for anti-floater configs.
# Sourced by run_high_quality.sh and run_high_quality_colmap.sh — do not execute directly.

# Apply profile-specific train.py arguments to the TRAIN_ARGS array (by name).
apply_synthetic_profile() {
    local scene_name="$1"
    local profile="${SCENE_PROFILE:-}"

    if [[ -z "$profile" ]]; then
        case "$scene_name" in
            lego|materials) profile="synthetic_edge" ;;
            *)              profile="synthetic" ;;
        esac
    fi

    case "$profile" in
        synthetic)
            TRAIN_ARGS+=(
                --white_background
                --densify_until_iter 15000
                --densify_grad_threshold 0.0004
                --opacity_reset_interval 3000
            )
            ;;
        synthetic_edge)
            TRAIN_ARGS+=(
                --white_background
                --densify_until_iter 15000
                --densify_grad_threshold 0.0005
                --opacity_reset_interval 3000
            )
            ;;
        *)
            echo "ERROR: Unknown synthetic profile: $profile"
            exit 1
            ;;
    esac
    PRUNE_PROFILE="$profile"
    echo "[INFO] Synthetic profile: $profile (scene=$scene_name)"
}

apply_colmap_profile() {
    local scene_name="$1"
    local profile="${SCENE_PROFILE:-}"

    if [[ -z "$profile" ]]; then
        case "$scene_name" in
            train|truck) profile="tnt_outdoor" ;;
            *)           profile="tnt_indoor" ;;
        esac
    fi

    case "$profile" in
        tnt_indoor)
            TRAIN_ARGS+=(
                --densify_until_iter 15000
                --densify_grad_threshold 0.0004
                --opacity_reset_interval 3000
            )
            ;;
        tnt_outdoor)
            TRAIN_ARGS+=(
                --densify_until_iter 10000
                --densify_grad_threshold 0.0008
                --opacity_reset_interval 3000
                --percent_dense 0.01
            )
            ;;
        *)
            echo "ERROR: Unknown COLMAP profile: $profile"
            exit 1
            ;;
    esac
    PRUNE_PROFILE="$profile"
    echo "[INFO] COLMAP profile: $profile (scene=$scene_name) — black background (no -w)"
}

run_post_prune() {
    local output_path="$1"
    local iteration="$2"
    local profile="$3"

    local ply_in="$output_path/point_cloud/iteration_$iteration/point_cloud.ply"
    local ply_out="$output_path/point_cloud/iteration_$iteration/point_cloud_pruned.ply"
    local report="$output_path/prune_report_$iteration.json"

    if [[ ! -f "$ply_in" ]]; then
        echo "WARNING: PLY not found, skipping prune: $ply_in"
        return 0
    fi

    echo ""
    echo ">>> Post-prune PLY (profile=$profile)..."
    "$PYTHON" "$GS_ROOT/scripts/prune_ply.py" \
        "$ply_in" \
        -o "$ply_out" \
        --profile "$profile" \
        --report "$report"
}
