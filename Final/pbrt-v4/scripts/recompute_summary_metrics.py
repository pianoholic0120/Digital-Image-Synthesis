#!/usr/bin/env python3
"""Recompute stochastic-vs-composite and fill missing quality blocks from report_assets EXRs."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT.parent
SUMMARY_PATH = FINAL / "report_results" / "benchmark_summary.json"

sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_assets import parse_scene_key  # noqa: E402
from report_benchmark_pipeline import (  # noqa: E402
    aggregate_metrics,
    metrics_vs_gt,
    view_dir,
)

SPP_NERF = [1, 32, 64, 256, 1024, 2048]
SPP_TANDT = [1, 32, 64, 256, 1024]


def max_export_spp(ds_name: str) -> int:
    return 2048 if ds_name == "3dgs" else 1024


def recompute_scene(key: str, entry: dict, n_views: int) -> dict:
    dataset, scene = parse_scene_key(key)
    ds_name = dataset.name
    scene_spp = SPP_TANDT if ds_name == "tandt_db" else SPP_NERF
    max_spp = max_export_spp(ds_name)
    quality = dict(entry.get("quality", {}))

    for spp in scene_spp:
        cross = []
        for frame in range(n_views):
            stoch = view_dir(ds_name, scene, spp, frame) / "stochastic.exr"
            comp = view_dir(ds_name, scene, max_spp, frame) / "composite.exr"
            if stoch.exists() and comp.exists():
                cross.append(metrics_vs_gt(stoch, comp))
        if cross:
            quality[f"stochastic_vs_composite_{spp}"] = asdict(aggregate_metrics(cross))

    stoch1024 = []
    comp1 = []
    rast = []
    for frame in range(n_views):
        gt = dataset.gt_path(scene, frame)
        s = view_dir(ds_name, scene, max_spp, frame) / "stochastic.exr"
        c = view_dir(ds_name, scene, max_spp, frame) / "composite.exr"
        if s.exists() and gt.exists():
            stoch1024.append(metrics_vs_gt(s, gt))
        if c.exists() and gt.exists():
            comp1.append(metrics_vs_gt(c, gt))
        rast_png = dataset.scene_dir(scene) / "test" / "ours_30000" / "renders" / f"{frame:05d}.png"
        if rast_png.exists() and gt.exists():
            rast.append(metrics_vs_gt(rast_png, gt))
    if stoch1024:
        quality["stochastic_1024"] = asdict(aggregate_metrics(stoch1024))
    if comp1:
        quality["composite_1"] = asdict(aggregate_metrics(comp1))
    if rast:
        quality["rasterizer"] = asdict(aggregate_metrics(rast))

    entry["quality"] = quality
    return entry


def main() -> int:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    n_views = len(summary.get("config", {}).get("views", list(range(10))))
    for key, entry in summary.get("scenes", {}).items():
        summary["scenes"][key] = recompute_scene(key, entry, n_views)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Updated {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
