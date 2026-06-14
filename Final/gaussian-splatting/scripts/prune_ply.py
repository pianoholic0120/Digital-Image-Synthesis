#!/usr/bin/env python3
"""Post-training PLY pruning to remove floaters and oversized Gaussians."""

import argparse
import json
import os
import sys

import numpy as np
from plyfile import PlyData, PlyElement

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

C0 = 0.28209479177387814

PROFILES = {
    "synthetic": {
        "min_opacity": 0.008,
        "max_scale_mult": 3.0,
        "far_dist_pct": 85.0,
        "bright_dc_thresh": 3.5,
        "large_scale_opacity": 0.3,
    },
    "synthetic_edge": {
        "min_opacity": 0.008,
        "max_scale_mult": 2.5,
        "far_dist_pct": 80.0,
        "bright_dc_thresh": 3.0,
        "large_scale_opacity": 0.25,
    },
    "tnt_indoor": {
        "min_opacity": 0.005,
        "max_scale_mult": 3.0,
        "far_dist_pct": 90.0,
        "bright_dc_thresh": 4.0,
        "large_scale_opacity": 0.35,
    },
    "tnt_outdoor": {
        "min_opacity": 0.005,
        "max_scale_mult": 2.5,
        "far_dist_pct": 85.0,
        "bright_dc_thresh": 3.5,
        "large_scale_opacity": 0.3,
    },
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prune_mask(vertices, prop_names, cfg):
    xyz = np.stack(
        [np.asarray(vertices["x"]), np.asarray(vertices["y"]), np.asarray(vertices["z"])],
        axis=1,
    )
    n = len(xyz)

    opacity = sigmoid(np.asarray(vertices["opacity"]))
    f_dc = np.stack(
        [
            np.asarray(vertices["f_dc_0"]),
            np.asarray(vertices["f_dc_1"]),
            np.asarray(vertices["f_dc_2"]),
        ],
        axis=1,
    )
    dc_mag = np.linalg.norm(f_dc, axis=1)

    scale_cols = sorted(
        [p for p in prop_names if p.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    scales = np.stack([np.asarray(vertices[c]) for c in scale_cols], axis=1)
    max_scale = np.exp(scales).max(axis=1)

    center = xyz.mean(axis=0)
    dist = np.linalg.norm(xyz - center, axis=1)
    dist_thresh = np.percentile(dist, cfg["far_dist_pct"])
    scale_thresh = np.percentile(max_scale, 99) * cfg["max_scale_mult"]

    keep = np.ones(n, dtype=bool)

    # Rule 1: very low opacity
    keep &= opacity >= cfg["min_opacity"]

    # Rule 2: far + bright DC (classic floater)
    far_bright = (dist > dist_thresh) & (dc_mag > cfg["bright_dc_thresh"])
    keep &= ~far_bright

    # Rule 3: oversized + semi-opaque (sky blobs on outdoor scenes)
    large_blob = (max_scale > scale_thresh) & (opacity > cfg["large_scale_opacity"])
    keep &= ~large_blob

    return keep, {
        "total": int(n),
        "kept": int(keep.sum()),
        "removed": int((~keep).sum()),
        "removed_low_opacity": int((opacity < cfg["min_opacity"]).sum()),
        "removed_far_bright": int(far_bright.sum()),
        "removed_large_blob": int(large_blob.sum()),
        "max_scale_before": float(max_scale.max()),
        "max_scale_after": float(max_scale[keep].max()) if keep.any() else 0.0,
        "opacity_gt_0.5_before": float((opacity > 0.5).mean()),
        "opacity_gt_0.5_after": float((opacity[keep] > 0.5).mean()) if keep.any() else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Prune floaters from a 3DGS PLY.")
    parser.add_argument("input_ply", type=str, help="Input point_cloud.ply")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output PLY (default: <input>_pruned.ply)",
    )
    parser.add_argument(
        "--profile", type=str, default="synthetic",
        choices=list(PROFILES.keys()),
        help="Pruning profile (default: synthetic)",
    )
    parser.add_argument("--min-opacity", type=float, default=None)
    parser.add_argument("--max-scale-mult", type=float, default=None)
    parser.add_argument("--report", type=str, default=None, help="JSON report path")
    args = parser.parse_args()

    cfg = PROFILES[args.profile].copy()
    if args.min_opacity is not None:
        cfg["min_opacity"] = args.min_opacity
    if args.max_scale_mult is not None:
        cfg["max_scale_mult"] = args.max_scale_mult

    ply = PlyData.read(args.input_ply)
    vertex_el = ply["vertex"]
    vertices = vertex_el.data
    prop_names = [p.name for p in vertex_el.properties]
    keep, stats = prune_mask(vertices, prop_names, cfg)

    pruned = vertices[keep]
    out_path = args.output or args.input_ply.replace(".ply", "_pruned.ply")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    PlyData([PlyElement.describe(pruned, "vertex")], text=False).write(out_path)

    stats["profile"] = args.profile
    stats["input"] = args.input_ply
    stats["output"] = out_path
    stats["config"] = cfg

    print(f"Pruned PLY: {stats['removed']}/{stats['total']} removed "
          f"({100 * stats['removed'] / max(stats['total'], 1):.1f}%)")
    print(f"  low opacity:  {stats['removed_low_opacity']}")
    print(f"  far+bright:     {stats['removed_far_bright']}")
    print(f"  large blob:     {stats['removed_large_blob']}")
    print(f"  max scale:      {stats['max_scale_before']:.4f} → {stats['max_scale_after']:.4f}")
    print(f"  saved to:       {out_path}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
