#!/usr/bin/env python3
"""Create lightweight PLY previews (+ optional PNG) for every 3DGS scene asset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "scenes" / "assets"
DATASETS = ("3dgs", "tandt_db")

# ~3 MB binary PLY (62 float props × 4 bytes ≈ 248 B/vertex).
DEFAULT_TARGET_MB = 3.0
BYTES_PER_VERTEX = 62 * 4
PREVIEW_PLY = "point_cloud_preview.ply"
PREVIEW_PNG = "point_cloud_preview.png"
FULL_PNG = "point_cloud.png"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def scene_ply_paths() -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    for ds in DATASETS:
        base = ASSETS / ds
        if not base.exists():
            continue
        for scene_dir in sorted(base.iterdir()):
            if not scene_dir.is_dir():
                continue
            ply = scene_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            if ply.exists():
                out.append((ds, scene_dir.name, ply))
    return out


def max_vertices_for_target(target_mb: float) -> int:
    return max(5000, int(target_mb * 1024 * 1024 / BYTES_PER_VERTEX))


def subsample_indices(vertices, n_keep: int, seed: int) -> np.ndarray:
    n = len(vertices)
    if n_keep >= n:
        return np.arange(n, dtype=np.int64)

    opacity = sigmoid(np.asarray(vertices["opacity"], dtype=np.float64))
    # Prefer visible Gaussians; break ties with deterministic RNG.
    rng = np.random.default_rng(seed)
    jitter = rng.random(n) * 1e-6
    order = np.argsort(-(opacity + jitter))
    return order[:n_keep].astype(np.int64)


def write_point_cloud_png(vertices, dst: Path, *, size: int = 640, max_draw: int = 120_000) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    xyz = np.stack(
        [np.asarray(vertices["x"]), np.asarray(vertices["y"]), np.asarray(vertices["z"])],
        axis=1,
    )
    opacity = sigmoid(np.asarray(vertices["opacity"]))
    dc = np.stack(
        [np.asarray(vertices["f_dc_0"]), np.asarray(vertices["f_dc_1"]), np.asarray(vertices["f_dc_2"])],
        axis=1,
    )
    rgb = np.clip(0.5 + 0.28209479177387814 * dc, 0.0, 1.0)
    alpha = np.clip(opacity, 0.05, 1.0)

    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1) & np.isfinite(alpha)
    xyz, rgb, alpha = xyz[valid], rgb[valid], alpha[valid]
    if len(xyz) == 0:
        return

    n = len(xyz)
    step = max(1, n // max_draw)
    xyz, rgb, alpha = xyz[::step], rgb[::step], alpha[::step]

    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=rgb, s=0.15, alpha=alpha, linewidths=0, rasterized=True,
    )
    ax.set_axis_off()
    c = xyz.mean(axis=0)
    r = np.linalg.norm(xyz - c, axis=1).max() * 1.05
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout(pad=0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def write_preview_ply_from_vertices(vertices, dst: Path, n_keep: int, seed: int, src_mb: float) -> dict:
    n = len(vertices)
    idx = subsample_indices(vertices, n_keep, seed)
    preview = vertices[idx]
    dst.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(preview, "vertex")], text=False).write(str(dst))
    size_mb = dst.stat().st_size / (1024 * 1024)
    return {
        "input_vertices": n,
        "preview_vertices": int(len(preview)),
        "fraction": round(len(preview) / n, 4),
        "input_mb": round(src_mb, 2),
        "preview_mb": round(size_mb, 2),
    }


def process_scene(
    ds: str,
    scene: str,
    src: Path,
    *,
    target_mb: float,
    seed: int,
    png: bool,
    write_preview_ply: bool,
) -> dict:
    out_dir = src.parent
    src_mb = src.stat().st_size / (1024 * 1024)
    vertices = PlyData.read(str(src))["vertex"].data
    stats = {"input_vertices": len(vertices), "input_mb": round(src_mb, 2)}

    if write_preview_ply:
        dst_ply = out_dir / PREVIEW_PLY
        n_keep = max_vertices_for_target(target_mb)
        ply_stats = write_preview_ply_from_vertices(vertices, dst_ply, n_keep, seed, src_mb)
        stats.update(ply_stats)
        stats["preview_ply"] = str(dst_ply.relative_to(ROOT))
    else:
        preview_ply = out_dir / PREVIEW_PLY
        if preview_ply.exists():
            stats["preview_ply"] = str(preview_ply.relative_to(ROOT))
            stats["preview_vertices"] = len(PlyData.read(str(preview_ply))["vertex"].data)
            stats["preview_mb"] = round(preview_ply.stat().st_size / (1024 * 1024), 2)

    stats.update({"dataset": ds, "scene": scene})

    if png:
        full_png = out_dir / FULL_PNG
        write_point_cloud_png(vertices, full_png)
        if full_png.exists():
            stats["full_png"] = str(full_png.relative_to(ROOT))
            stats["full_png_kb"] = round(full_png.stat().st_size / 1024, 1)
            stats["full_png_vertices"] = len(vertices)

        preview_ply = out_dir / PREVIEW_PLY
        if preview_ply.exists():
            preview = PlyData.read(str(preview_ply))["vertex"].data
            preview_png = out_dir / PREVIEW_PNG
            write_point_cloud_png(preview, preview_png)
            if preview_png.exists():
                stats["preview_png"] = str(preview_png.relative_to(ROOT))
                stats["preview_png_kb"] = round(preview_png.stat().st_size / 1024, 1)

    if write_preview_ply:
        print(
            f"  {ds}/{scene}: {stats['input_vertices']:,} -> {stats['preview_vertices']:,} "
            f"({stats['preview_mb']} MB)",
            flush=True,
        )
    else:
        print(f"  {ds}/{scene}: full PNG from {stats['input_vertices']:,} Gaussians", flush=True)
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-mb", type=float, default=DEFAULT_TARGET_MB, help="Target preview PLY size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-png", action="store_true", help="Skip PNG generation")
    p.add_argument("--full-png-only", action="store_true", help="Only render full-resolution PNGs (skip preview PLY rewrite)")
    p.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = p.parse_args()

    scenes = [(ds, sc, ply) for ds, sc, ply in scene_ply_paths() if ds in args.datasets]
    if not scenes:
        print("No point_cloud.ply files found.", file=sys.stderr)
        return 1

    if not args.full_png_only:
        print(f"Target preview size ~{args.target_mb} MB (~{max_vertices_for_target(args.target_mb):,} Gaussians)", flush=True)
    results = []
    for ds, scene, src in scenes:
        print(f"=== {ds}/{scene} ===", flush=True)
        results.append(
            process_scene(
                ds, scene, src,
                target_mb=args.target_mb,
                seed=args.seed,
                png=not args.no_png,
                write_preview_ply=not args.full_png_only,
            )
        )

    manifest = {
        "target_mb": args.target_mb,
        "preview_ply_name": PREVIEW_PLY,
        "preview_png_name": PREVIEW_PNG,
        "full_png_name": FULL_PNG,
        "scenes": results,
    }
    manifest_path = ASSETS / "ply_preview_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
