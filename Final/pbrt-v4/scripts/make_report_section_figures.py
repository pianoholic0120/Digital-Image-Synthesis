#!/usr/bin/env python3
"""Render assets and build Figures 1–4 for Final/report.docx (§5.4–5.5)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT.parent
SCRIPTS = ROOT / "scripts"
REPORT_ASSETS = FINAL / "report_assets"
REPORT_FIGURES = FINAL / "report_figures"
SCENES_DIR = ROOT / "scenes" / "benchmark"
PBRT = ROOT / "build" / "Release" / "pbrt.exe"
REPORT_DOCX = FINAL / "report.docx"

sys.path.insert(0, str(SCRIPTS))
from benchmark_assets import DATASETS, render_scene_pbrt  # noqa: E402
from report_benchmark_pipeline import (  # noqa: E402
    pbrt_out_rel,
    save_preview_png,
    view_dir,
)

SPP_SWEEP = [1, 32, 64, 256, 1024]
# Views shown in Figure 1–2 grids.
FIG12_VIEWS = [0, 3]

FIG34_VIEW = 0
# Canonical folder for composite reference PNG (composite is 1-spp, identical across export folders).
FIG34_COMPOSITE_SPP = 64

# Only render assets required for Figures 1–4 (not all 10 test views).
RENDER_JOBS: list[tuple[str, str, list[int]]] = [
    ("3dgs", "lego", FIG12_VIEWS),
    ("tandt_db", "drjohnson", FIG12_VIEWS),
    ("3dgs", "hotdog", [FIG34_VIEW]),
    ("tandt_db", "playroom", [FIG34_VIEW]),
]

FIG12_SCENES = [
    ("3dgs", "lego", "nerf_synthetic", 200),
    ("tandt_db", "drjohnson", "tandt_db", 180),
]
FIG34_SCENES = [
    ("3dgs", "hotdog", "nerf_synthetic", 220),
    ("tandt_db", "playroom", "tandt_db", 180),
]


def run_pbrt(scene_path: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run(
        [str(PBRT), str(scene_path), "--disable-pixel-jitter"],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )
    return time.perf_counter() - t0


def render_stochastic(ds_name: str, scene: str, spp: int, frame: int, *, force: bool) -> None:
    dataset = DATASETS[ds_name]
    vdir = view_dir(ds_name, scene, spp, frame)
    exr = vdir / "stochastic.exr"
    if exr.exists() and not force:
        return
    vdir.mkdir(parents=True, exist_ok=True)
    rel = pbrt_out_rel(ds_name, scene, spp, frame, "stochastic.exr")
    scene_path = SCENES_DIR / f"_fig_{scene}_stoch_f{frame}_s{spp}.pbrt"
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(
        render_scene_pbrt(dataset, scene, frame=frame, mode="stochastic", spp=spp, out_exr=rel),
        encoding="utf-8",
    )
    dt = run_pbrt(scene_path)
    save_preview_png(exr, vdir / "stochastic.png")
    print(f"  stochastic {ds_name}/{scene} view={frame} spp={spp}: {dt:.1f}s", flush=True)


def render_composite(ds_name: str, scene: str, frame: int, *, force: bool) -> None:
    dataset = DATASETS[ds_name]
    vdir = view_dir(ds_name, scene, FIG34_COMPOSITE_SPP, frame)
    exr = vdir / "composite.exr"
    if exr.exists() and not force:
        if not (vdir / "composite.png").exists():
            save_preview_png(exr, vdir / "composite.png")
        return
    vdir.mkdir(parents=True, exist_ok=True)
    rel = pbrt_out_rel(ds_name, scene, FIG34_COMPOSITE_SPP, frame, "composite.exr")
    scene_path = SCENES_DIR / f"_fig_{scene}_comp_f{frame}.pbrt"
    scene_path.write_text(
        render_scene_pbrt(dataset, scene, frame=frame, mode="reference", spp=1, out_exr=rel),
        encoding="utf-8",
    )
    dt = run_pbrt(scene_path)
    save_preview_png(exr, vdir / "composite.png")
    print(f"  composite {ds_name}/{scene} view={frame}: {dt:.1f}s", flush=True)


def render_fig34_assets(*, force: bool) -> None:
    for ds_name, scene, views in RENDER_JOBS:
        if scene not in ("hotdog", "playroom"):
            continue
        print(f"=== fig34 {ds_name}/{scene} ===", flush=True)
        for frame in views:
            render_composite(ds_name, scene, frame, force=force)
            for spp in SPP_SWEEP:
                render_stochastic(ds_name, scene, spp, frame, force=force)


def render_all(*, force: bool, scenes: list[tuple[str, str, list[int]]] | None = None) -> None:
    if not PBRT.exists():
        raise SystemExit(f"Missing {PBRT}; build pbrt first.")
    jobs = scenes or RENDER_JOBS
    for ds_name, scene, views in jobs:
        if scene in ("hotdog", "playroom"):
            continue  # handled by render_fig34_assets
        print(f"=== {ds_name}/{scene} views={views} ===", flush=True)
        for frame in views:
            for spp in SPP_SWEEP:
                render_stochastic(ds_name, scene, spp, frame, force=force)
    render_fig34_assets(force=force)


def load_rgb(path: Path, height: int) -> Image.Image | None:
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    if im.height != height:
        w = max(1, int(im.width * height / im.height))
        im = im.resize((w, height), Image.Resampling.LANCZOS)
    return im


def spp_sweep_grid(
    folder: str,
    scene: str,
    views: list[int],
    *,
    thumb_h: int,
) -> Image.Image:
    """Multi-row (views) × multi-column (SPP) grid matching spp_sweep_lego layout."""
    gap, label_w, header_h, row_label_h = 4, 100, 28, 22
    rows: list[tuple[str, list[Image.Image]]] = []
    for view in views:
        cells: list[Image.Image] = []
        for spp in SPP_SWEEP:
            path = REPORT_ASSETS / folder / scene / f"spp_{spp:04d}" / f"view_{view:05d}" / "stochastic.png"
            im = load_rgb(path, thumb_h)
            if im is None:
                raise FileNotFoundError(path)
            cells.append(im)
        rows.append((f"view {view}", cells))

    col_w = max(im.width for _, cells in rows for im in cells)
    row_h = thumb_h + row_label_h
    canvas_w = label_w + len(SPP_SWEEP) * col_w + (len(SPP_SWEEP) - 1) * gap
    canvas_h = header_h + len(rows) * row_h + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for ci, spp in enumerate(SPP_SWEEP):
        x = label_w + ci * (col_w + gap)
        draw.text((x + 4, 6), f"{spp} spp", fill=(0, 0, 0))

    y = header_h
    for row_label, cells in rows:
        draw.text((4, y + 4), row_label, fill=(60, 60, 60))
        for ci, im in enumerate(cells):
            x = label_w + ci * (col_w + gap)
            canvas.paste(im, (x, y + row_label_h))
        y += row_h + gap
    return canvas


def vertical_stoch_vs_composite_grid(
    scenes: list[tuple[str, str, str, int]],
    view: int,
    *,
    thumb_h: int,
) -> Image.Image:
    """5 rows (stochastic SPP) × 4 cols: per scene, stochastic | composite (view 0).

    Composite is exact 1-spp reference (SPP-independent); repeated in each row for alignment.
    """
    gap, scene_gap = 4, 20
    header_h, row_label_w = 28, 72
    col_modes = ["stochastic", "composite"]

    composite_cache: dict[tuple[str, str], Image.Image] = {}
    grid: list[list[Image.Image]] = []
    for spp in SPP_SWEEP:
        row: list[Image.Image] = []
        for _ds, scene, folder, _thumb in scenes:
            stoch_path = (
                REPORT_ASSETS / folder / scene / f"spp_{spp:04d}" / f"view_{view:05d}" / "stochastic.png"
            )
            stoch = load_rgb(stoch_path, thumb_h)
            if stoch is None:
                raise FileNotFoundError(stoch_path)
            row.append(stoch)

            key = (folder, scene)
            if key not in composite_cache:
                comp_path = (
                    REPORT_ASSETS / folder / scene / f"spp_{FIG34_COMPOSITE_SPP:04d}"
                    / f"view_{view:05d}" / "composite.png"
                )
                comp = load_rgb(comp_path, thumb_h)
                if comp is None:
                    raise FileNotFoundError(comp_path)
                composite_cache[key] = comp
            row.append(composite_cache[key])
        grid.append(row)

    col_w = max(im.width for row in grid for im in row)
    row_h = thumb_h
    n_scenes = len(scenes)
    scene_block_w = 2 * col_w + gap
    canvas_w = row_label_w + n_scenes * scene_block_w + (n_scenes - 1) * scene_gap
    canvas_h = header_h + len(SPP_SWEEP) * row_h + (len(SPP_SWEEP) - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = row_label_w
    for _ds, _scene, _folder, _ in scenes:
        for mi, mode in enumerate(col_modes):
            cx = x + mi * (col_w + gap)
            draw.text((cx + 4, 6), mode, fill=(0, 0, 0))
        x += scene_block_w + scene_gap

    y = header_h
    for ri, spp in enumerate(SPP_SWEEP):
        draw.text((4, y + 4), f"{spp} spp", fill=(0, 0, 0))
        x = row_label_w
        col_idx = 0
        for _ds, _scene, _folder, _ in scenes:
            for mi in range(2):
                im = grid[ri][col_idx]
                px = x + mi * (col_w + gap)
                py = y + (row_h - im.height) // 2
                canvas.paste(im, (px, py))
                col_idx += 1
            x += scene_block_w + scene_gap
        y += row_h + gap
    return canvas


def figure_hotdog_playroom_sweep() -> Image.Image:
    """5 rows (SPP) × 4 cols: hotdog stoch|comp, playroom stoch|comp (view 0)."""
    thumb_h = min(h for *_, h in FIG34_SCENES)
    return vertical_stoch_vs_composite_grid(FIG34_SCENES, FIG34_VIEW, thumb_h=thumb_h)


def build_figures() -> dict[str, Path]:
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    out = {
        "figure1": REPORT_FIGURES / "figure1_spp_sweep_lego.png",
        "figure2": REPORT_FIGURES / "figure2_spp_sweep_drjohnson.png",
        "figure34": REPORT_FIGURES / "figure3_4_hotdog_playroom.png",
    }
    fig34_backup = REPORT_FIGURES / "figure3_4_hotdog_playroom_stoch_views.png"
    if out["figure34"].exists() and not fig34_backup.exists():
        shutil.copy2(out["figure34"], fig34_backup)
        print(f"Backed up previous Figure 3–4 to {fig34_backup}", flush=True)
    _, _, folder, thumb_h = FIG12_SCENES[0]
    spp_sweep_grid(folder, "lego", FIG12_VIEWS, thumb_h=thumb_h).save(out["figure1"])
    print(f"Wrote {out['figure1']}", flush=True)
    _, _, folder, thumb_h = FIG12_SCENES[1]
    spp_sweep_grid(folder, "drjohnson", FIG12_VIEWS, thumb_h=thumb_h).save(out["figure2"])
    print(f"Wrote {out['figure2']}", flush=True)
    figure_hotdog_playroom_sweep().save(out["figure34"])
    print(f"Wrote {out['figure34']}", flush=True)
    return out


def insert_into_docx(figures: dict[str, Path]) -> None:
    if not REPORT_DOCX.exists():
        print(f"Skip docx: missing {REPORT_DOCX}", flush=True)
        return
    from docx import Document

    from update_report_from_results import find_paragraph, insert_figure_before, paragraph_has_image

    doc = Document(str(REPORT_DOCX))
    specs = [
        (
            "Figure 1. Different levels of SPP",
            figures["figure1"],
            "Figure 1. Different levels of SPP for the stochastic estimator on lego (NeRF Synthetic, 800×800).",
            6.5,
        ),
        (
            "Figure 2. Different levels of SPP",
            figures["figure2"],
            "Figure 2. Different levels of SPP for the stochastic estimator on drjohnson (Tanks & Temples, 1920×1080).",
            6.5,
        ),
        (
            "Figure 4",
            figures["figure34"],
            "Figure 4–5. Stochastic SPP sweep vs. composite reference on hotdog and playroom (view 0). "
            "Rows show stochastic at increasing SPP; composite is exact 1-spp reference (identical each row).",
            6.5,
        ),
    ]
    for start, path, caption, width in specs:
        anchor = find_paragraph(doc, startswith=start)
        if anchor is None:
            print(f"  docx anchor not found: {start!r}", flush=True)
            continue
        prev = anchor._element.getprevious()
        if prev is not None and prev.tag.endswith("p"):
            from docx.text.paragraph import Paragraph

            if paragraph_has_image(Paragraph(prev, anchor._parent)):
                prev.getparent().remove(prev)
                prev2 = anchor._element.getprevious()
                if prev2 is not None and prev2.tag.endswith("p"):
                    from docx.text.paragraph import Paragraph as P2

                    if caption.split(".")[0] in P2(prev2, anchor._parent).text:
                        prev2.getparent().remove(prev2)
        insert_figure_before(doc, anchor, path, caption, width_in=width)
    _save_docx_or_copy(doc, REPORT_DOCX)


def _save_docx_or_copy(doc, path: Path) -> Path:
    try:
        doc.save(str(path))
        print(f"Updated {path}", flush=True)
        return path
    except PermissionError:
        alt = path.with_name(path.stem + "_figures_updated.docx")
        doc.save(str(alt))
        print(f"Could not write {path} (file may be open in Word). Saved {alt}", flush=True)
        return alt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", nargs="*", default=None,
                   help="Limit render to scene names, e.g. hotdog playroom")
    p.add_argument("--skip-render", action="store_true", help="Only rebuild PNG grids from existing assets")
    p.add_argument("--skip-docx", action="store_true")
    p.add_argument("--force-render", action="store_true", help="Re-render even if EXR exists")
    args = p.parse_args()

    jobs = RENDER_JOBS
    if args.scenes:
        wanted = set(args.scenes)
        jobs = [j for j in RENDER_JOBS if j[1] in wanted]
        if not jobs:
            raise SystemExit(f"No render jobs for scenes {args.scenes}")

    if not args.skip_render:
        render_all(force=args.force_render, scenes=jobs)
    figures = build_figures()
    if not args.skip_docx:
        insert_into_docx(figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
