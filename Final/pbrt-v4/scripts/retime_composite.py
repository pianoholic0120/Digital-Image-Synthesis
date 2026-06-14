#!/usr/bin/env python3
"""Re-render composite (1 spp) and record wall-clock time for scenes missing composite_1.time_s."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT.parent
SUMMARY_PATH = FINAL / "report_results" / "benchmark_summary.json"
SCENES_DIR = ROOT / "scenes" / "benchmark"

sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_assets import DATASETS, parse_scene_key, render_scene_pbrt  # noqa: E402
from report_benchmark_pipeline import (  # noqa: E402
    aggregate_metrics,
    metrics_vs_gt,
    pbrt_out_rel,
    run_pbrt,
    view_dir,
)


def retime_scene(key: str, n_views: int, pbrt: Path) -> float:
    dataset, scene = parse_scene_key(key)
    ds_name = dataset.name
    max_spp = 2048 if ds_name == "3dgs" else 1024
    times: list[float] = []
    metrics = []

    for frame in range(n_views):
        gt = dataset.gt_path(scene, frame)
        if not gt.exists():
            print(f"  skip frame {frame}: missing GT", flush=True)
            continue
        canon_vdir = view_dir(ds_name, scene, max_spp, frame)
        canon_vdir.mkdir(parents=True, exist_ok=True)
        rel_c = pbrt_out_rel(ds_name, scene, max_spp, frame, "composite.exr")
        scene_c = SCENES_DIR / f"_retime_{scene}_comp_f{frame}.pbrt"
        scene_c.write_text(
            render_scene_pbrt(dataset, scene, frame=frame, mode="reference", spp=1, out_exr=rel_c),
            encoding="utf-8",
        )
        t = run_pbrt(pbrt, scene_c)
        times.append(t)
        comp_exr = canon_vdir / "composite.exr"
        if comp_exr.exists():
            m = metrics_vs_gt(comp_exr, gt)
            m.time_s = t
            metrics.append(m)
        print(f"  frame {frame}: {t:.1f}s", flush=True)

    mean_t = float(np.mean(times)) if times else 0.0
    return mean_t, metrics


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", nargs="+", required=True, help="e.g. tandt_db/drjohnson tandt_db/playroom")
    p.add_argument("--views", type=int, default=10)
    p.add_argument("--pbrt", type=Path, default=ROOT / "build/Release/pbrt.exe")
    args = p.parse_args()

    if not SUMMARY_PATH.exists():
        print(f"missing {SUMMARY_PATH}", file=sys.stderr)
        return 1

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    for key in args.scenes:
        print(f"=== {key} ===", flush=True)
        mean_t, metrics = retime_scene(key, args.views, args.pbrt)
        entry = summary.setdefault("scenes", {}).setdefault(key, {})
        q = entry.setdefault("quality", {})
        if metrics:
            agg = asdict(aggregate_metrics(metrics))
            agg["time_s"] = mean_t
            q["composite_1"] = agg
        else:
            q.setdefault("composite_1", {})["time_s"] = mean_t
        print(f"  mean composite time: {mean_t:.1f}s", flush=True)

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Updated {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
