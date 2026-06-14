#!/usr/bin/env python3
"""Build 6×4 method comparison grid for NeRF Synthetic scenes (chair … ship)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT.parent
SCRIPTS = ROOT / "scripts"
REPORT_ASSETS = FINAL / "report_assets" / "nerf_synthetic"
REPORT_FIGURES = FINAL / "report_figures" / "nerf_synthetic"
SCENES_DIR = ROOT / "scenes" / "benchmark"
PBRT = ROOT / "build" / "Release" / "pbrt.exe"

sys.path.insert(0, str(SCRIPTS))
from benchmark_assets import DATASETS, render_scene_pbrt  # noqa: E402
from report_benchmark_pipeline import (  # noqa: E402
    pbrt_out_rel,
    rasterizer_path,
    save_preview_png,
    view_dir,
)

STOCH_SPP = 64
COMPOSITE_EXPORT_SPP = 64  # folder under report_assets with composite.png
THUMB_H = 180

SCENES = ["chair", "drums", "ficus", "materials", "mic", "ship"]
# One representative test view per scene.
SCENE_VIEW: dict[str, int] = {
    "chair": 0,
    "drums": 0,
    "ficus": 0,
    "materials": 0,
    "mic": 0,
    "ship": 0,
}

COL_LABELS = ["Stochastic 64 spp", "Composite", "3DGS rasterizer", "GT"]
DATASET = DATASETS["3dgs"]


def run_pbrt(scene_path: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run(
        [str(PBRT), str(scene_path), "--disable-pixel-jitter"],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )
    return time.perf_counter() - t0


def render_stochastic(scene: str, frame: int, *, force: bool) -> None:
    vdir = view_dir("3dgs", scene, STOCH_SPP, frame)
    exr = vdir / "stochastic.exr"
    if exr.exists() and not force:
        return
    vdir.mkdir(parents=True, exist_ok=True)
    rel = pbrt_out_rel("3dgs", scene, STOCH_SPP, frame, "stochastic.exr")
    scene_path = SCENES_DIR / f"_extra_{scene}_stoch64_f{frame}.pbrt"
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(
        render_scene_pbrt(DATASET, scene, frame=frame, mode="stochastic", spp=STOCH_SPP, out_exr=rel),
        encoding="utf-8",
    )
    dt = run_pbrt(scene_path)
    save_preview_png(exr, vdir / "stochastic.png")
    print(f"  stochastic64 {scene} view={frame}: {dt:.1f}s", flush=True)


def render_composite(scene: str, frame: int, *, force: bool) -> None:
    vdir = view_dir("3dgs", scene, COMPOSITE_EXPORT_SPP, frame)
    exr = vdir / "composite.exr"
    if exr.exists() and not force:
        if not (vdir / "composite.png").exists():
            save_preview_png(exr, vdir / "composite.png")
        return
    vdir.mkdir(parents=True, exist_ok=True)
    rel = pbrt_out_rel("3dgs", scene, COMPOSITE_EXPORT_SPP, frame, "composite.exr")
    scene_path = SCENES_DIR / f"_extra_{scene}_comp_f{frame}.pbrt"
    scene_path.write_text(
        render_scene_pbrt(DATASET, scene, frame=frame, mode="reference", spp=1, out_exr=rel),
        encoding="utf-8",
    )
    dt = run_pbrt(scene_path)
    save_preview_png(exr, vdir / "composite.png")
    print(f"  composite {scene} view={frame}: {dt:.1f}s", flush=True)


def ensure_assets(*, force: bool, scenes: list[str] | None = None) -> None:
    if not PBRT.exists():
        raise SystemExit(f"Missing {PBRT}")
    for scene in scenes or SCENES:
        frame = SCENE_VIEW[scene]
        print(f"=== {scene} view={frame} ===", flush=True)
        render_stochastic(scene, frame, force=force)
        render_composite(scene, frame, force=force)


def load_rgb(path: Path, height: int) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    im = Image.open(path).convert("RGB")
    if im.height != height:
        w = max(1, int(im.width * height / im.height))
        im = im.resize((w, height), Image.Resampling.LANCZOS)
    return im


def method_paths(scene: str, view: int) -> dict[str, Path]:
    vdir = REPORT_ASSETS / scene / f"spp_{COMPOSITE_EXPORT_SPP:04d}" / f"view_{view:05d}"
    return {
        "stochastic": vdir / "stochastic.png",
        "composite": vdir / "composite.png",
        "rasterizer": rasterizer_path(DATASET, scene, view),
        "gt": DATASET.gt_path(scene, view),
    }


def build_combined_grid(*, thumb_h: int = THUMB_H, scenes: list[str] | None = None) -> Image.Image:
    gap, label_w, header_h, row_label_h = 4, 100, 28, 22
    scene_list = scenes or SCENES
    rows: list[tuple[str, list[Image.Image]]] = []
    for scene in scene_list:
        view = SCENE_VIEW[scene]
        paths = method_paths(scene, view)
        cells = [load_rgb(paths[k], thumb_h) for k in ("stochastic", "composite", "rasterizer", "gt")]
        rows.append((scene, cells))

    col_w = max(im.width for _, cells in rows for im in cells)
    row_h = thumb_h + row_label_h
    canvas_w = label_w + len(COL_LABELS) * col_w + (len(COL_LABELS) - 1) * gap
    canvas_h = header_h + len(rows) * row_h + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for ci, hdr in enumerate(COL_LABELS):
        x = label_w + ci * (col_w + gap)
        draw.text((x + 4, 6), hdr, fill=(0, 0, 0))

    y = header_h
    for row_label, cells in rows:
        draw.text((4, y + 4), row_label, fill=(60, 60, 60))
        for ci, im in enumerate(cells):
            x = label_w + ci * (col_w + gap)
            canvas.paste(im, (x, y + row_label_h))
        y += row_h + gap
    return canvas


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--force-render", action="store_true")
    p.add_argument("--scenes", nargs="*", default=None)
    args = p.parse_args()

    scenes = args.scenes or SCENES
    if not args.skip_render:
        ensure_assets(force=args.force_render, scenes=scenes)

    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    combined = build_combined_grid(scenes=scenes)
    out = REPORT_FIGURES / "methods_comparison_6scenes.png"
    combined.save(out)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
