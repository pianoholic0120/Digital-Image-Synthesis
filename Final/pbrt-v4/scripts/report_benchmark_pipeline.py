#!/usr/bin/env python3
"""Full report benchmark: timings, metrics, PNG exports for Final/report_assets/."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT.parent
REPORT_ASSETS = FINAL / "report_assets"
SCENES_DIR = ROOT / "scenes" / "benchmark"
SUMMARY_PATH = FINAL / "report_results" / "benchmark_summary.json"
LOCK_PATH = FINAL / "report_results" / ".benchmark.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_benchmark_lock() -> None:
    """Refuse to start if another benchmark instance is already running."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            other = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except ValueError:
            other = -1
        if _pid_alive(other):
            print(
                f"Another benchmark is already running (pid {other}). "
                "Stop it before starting a second instance — concurrent pbrt runs invalidate timing.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        LOCK_PATH.unlink(missing_ok=True)
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(release_benchmark_lock)


def release_benchmark_lock() -> None:
    if not LOCK_PATH.exists():
        return
    try:
        if int(LOCK_PATH.read_text(encoding="utf-8").strip()) == os.getpid():
            LOCK_PATH.unlink(missing_ok=True)
    except (ValueError, OSError):
        pass

sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_assets import (  # noqa: E402
    DATASETS,
    all_scene_keys,
    parse_scene_key,
    render_scene_pbrt,
    use_2d_alpha,
)
from exr_to_png import read_exr  # noqa: E402

try:
    import OpenEXR
    import Imath
except ImportError as exc:
    raise SystemExit("pip install OpenEXR Imath numpy pillow") from exc

try:
    from skimage.metrics import structural_similarity as skimage_ssim

    def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
        return float(skimage_ssim(a, b, channel_axis=2, data_range=1.0))

except ImportError:

    def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
        c1, c2 = 0.01**2, 0.03**2
        mu_a, mu_b = a.mean(), b.mean()
        var_a, var_b = a.var(), b.var()
        cov = ((a - mu_a) * (b - mu_b)).mean()
        return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


DATASET_FOLDER = {"3dgs": "nerf_synthetic", "tandt_db": "tandt_db"}


@dataclass
class ViewMetrics:
    psnr: float
    ssim: float
    lpips: float | None
    time_s: float | None


def load_rgb(path: Path) -> np.ndarray:
    """Load image for metrics. EXR = linear; PNG = 3DGS convention (bytes treated as linear)."""
    if path.suffix.lower() == ".exr":
        return read_exr(path)
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def save_preview_png(src: Path, dst: Path) -> None:
    """Preview PNG matching 3DGS save convention (linear clamp → uint8, no gamma).

    GT and rasterizer PNGs from gaussian-splatting store linear RGB in [0,1] directly
    as 8-bit values (see composite_renders.save_tensor_image). Applying sRGB gamma here
    makes our images look washed-out / too white vs GT and renders/.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".exr":
        rgb = read_exr(src)
        linear = np.clip(rgb, 0.0, 1.0)
        Image.fromarray((linear * 255.0 + 0.5).astype(np.uint8)).save(dst)
    else:
        shutil.copy2(src, dst)


def align_pair(pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pred.shape[:2] == ref.shape[:2]:
        return pred, ref
    pred_im = Image.fromarray((np.clip(pred, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    pred_im = pred_im.resize((ref.shape[1], ref.shape[0]), Image.Resampling.LANCZOS)
    pred = np.asarray(pred_im, dtype=np.float64) / 255.0
    return pred, ref


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


_LPIPS_FN = None


def compute_lpips(a: np.ndarray, b: np.ndarray) -> float | None:
    global _LPIPS_FN
    try:
        import torch
        import lpips  # type: ignore

        if _LPIPS_FN is None:
            _LPIPS_FN = lpips.LPIPS(net="alex")
        ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            return float(_LPIPS_FN(ta, tb).item())
    except Exception:
        return None


def metrics_vs_gt(pred: Path, gt: Path) -> ViewMetrics:
    p = np.clip(load_rgb(pred), 0, 1)
    g = np.clip(load_rgb(gt), 0, 1)
    p, g = align_pair(p, g)
    return ViewMetrics(psnr=psnr(p, g), ssim=compute_ssim(p, g), lpips=compute_lpips(p, g), time_s=None)


def ply_gaussian_count_k(dataset_name: str, scene: str) -> float:
    ply = DATASETS[dataset_name].scene_dir(scene) / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    with ply.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.startswith("element vertex"):
                return int(line.split()[-1]) / 1000.0
    return 0.0


def gt_path(dataset, scene: str, frame: int) -> Path:
    return dataset.gt_path(scene, frame)


def rasterizer_path(dataset, scene: str, frame: int) -> Path:
    return dataset.scene_dir(scene) / "test" / "ours_30000" / "renders" / f"{frame:05d}.png"


def run_pbrt(pbrt: Path, scene_path: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run([str(pbrt), str(scene_path), "--disable-pixel-jitter"], check=True, cwd=ROOT)
    return time.perf_counter() - t0


def view_dir(dataset_name: str, scene: str, spp: int, frame: int) -> Path:
    return REPORT_ASSETS / DATASET_FOLDER[dataset_name] / scene / f"spp_{spp:04d}" / f"view_{frame:05d}"


def pbrt_out_rel(dataset_name: str, scene: str, spp: int, frame: int, filename: str) -> str:
    """EXR path relative to pbrt-v4 cwd (report_assets lives under Final/)."""
    folder = DATASET_FOLDER[dataset_name]
    return f"../report_assets/{folder}/{scene}/spp_{spp:04d}/view_{frame:05d}/{filename}"


def write_metrics_txt(path: Path, gt: Path, entries: dict[str, ViewMetrics]) -> None:
    lines = [f"ground_truth: {gt}", ""]
    for name, m in entries.items():
        lines.append(f"[{name}]")
        lines.append(f"  psnr_vs_gt: {m.psnr:.4f} dB")
        lines.append(f"  ssim_vs_gt: {m.ssim:.6f}")
        if m.lpips is not None:
            lines.append(f"  lpips_vs_gt: {m.lpips:.6f}")
        else:
            lines.append("  lpips_vs_gt: N/A (install torch+lpips)")
        if m.time_s is not None:
            lines.append(f"  render_time_s: {m.time_s:.3f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def scene_is_complete(
    entry: dict, spp_levels: list[int], export_spp: set[int], n_views: int, *, config: dict | None = None
) -> bool:
    if config and entry.get("benchmark_version") != config.get("benchmark_version"):
        return False
    timing = entry.get("timing_stochastic", {})
    if not all(str(s) in timing for s in spp_levels):
        return False
    q = entry.get("quality", {})
    needed = {"stochastic_1024", "composite_1", "rasterizer"}
    if not needed.issubset(q.keys()):
        return False
    for s in export_spp:
        if f"stochastic_vs_composite_{s}" not in q:
            return False
    if any(len(timing[str(s)].get("per_view_s", [])) < n_views for s in spp_levels):
        return False
    return True


BENCHMARK_VERSION = 3  # v3: T&T spp capped at 1024; composite at export-spp for analysis

SPP_NERF = [1, 32, 64, 256, 1024, 2048]
SPP_TANDT = [1, 32, 64, 256, 1024]


def spp_levels_for_dataset(ds_name: str, cli_spp: list[int] | None = None) -> list[int]:
    """NeRF Synthetic: up to 2048 spp; Tanks & Temples: stop at 1024."""
    base = SPP_TANDT if ds_name == "tandt_db" else SPP_NERF
    if not cli_spp:
        return base
    allowed = set(base)
    return [s for s in cli_spp if s in allowed]


def _benchmark_config(
    frames: list[int],
    cli_spp: list[int],
    export_spp: set[int],
    scene_keys: list[str],
) -> dict:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "views": frames,
        "spp_levels_cli": cli_spp,
        "spp_levels": {
            "3dgs": spp_levels_for_dataset("3dgs", cli_spp),
            "tandt_db": spp_levels_for_dataset("tandt_db", cli_spp),
        },
        "export_spp": sorted(export_spp),
        "scene_keys": scene_keys,
        "use_2d_alpha": {ds: use_2d_alpha(ds) for ds in DATASETS},
        "composite_mode": "reference (1 spp exact compositing at each export-spp folder)",
        "pixel_jitter": False,
        "sigma_cutoff": 2.828,
        "sh_degree": 3,
    }


def aggregate_metrics(items: list[ViewMetrics]) -> ViewMetrics:
    lp = [m.lpips for m in items if m.lpips is not None]
    ts = [m.time_s for m in items if m.time_s is not None]
    return ViewMetrics(
        psnr=float(np.mean([m.psnr for m in items])),
        ssim=float(np.mean([m.ssim for m in items])),
        lpips=float(np.mean(lp)) if lp else None,
        time_s=float(np.mean(ts)) if ts else None,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pbrt", type=Path, default=ROOT / "build/Release/pbrt.exe")
    p.add_argument("--datasets", nargs="*", default=["3dgs", "tandt_db"])
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--views", type=int, default=10, help="number of test views (frames 0..N-1)")
    p.add_argument("--spp", nargs="*", type=int, default=[1, 32, 64, 256, 1024, 2048])
    p.add_argument("--export-spp", nargs="*", type=int, default=[64, 1024], help="SPP levels that save PNGs")
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--resume", action="store_true", help="skip finished scenes / existing EXRs")
    p.add_argument("--timing-only", action="store_true", help="only record stochastic times, no PNG export")
    args = p.parse_args()

    if not args.pbrt.exists():
        print(f"missing {args.pbrt}", file=sys.stderr)
        return 1

    acquire_benchmark_lock()

    if args.scenes:
        scene_keys = []
        for s in args.scenes:
            if "/" in s:
                scene_keys.append(s)
            elif len(args.datasets) == 1:
                scene_keys.append(f"{args.datasets[0]}/{s}")
            else:
                raise SystemExit(f"--scenes {s!r} needs dataset/scene when multiple --datasets are set")
    else:
        scene_keys = all_scene_keys(args.datasets)
    frames = list(range(args.views))
    export_spp = set(args.export_spp)

    if args.resume and SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        summary.setdefault("scenes", {})
        summary["config"] = _benchmark_config(frames, args.spp, export_spp, scene_keys)
    else:
        summary = {"config": _benchmark_config(frames, args.spp, export_spp, scene_keys), "scenes": {}}

    max_export_spp = max(export_spp) if export_spp else 1024

    for ds in sorted({parse_scene_key(k)[0].name for k in scene_keys}):
        levels = spp_levels_for_dataset(ds, args.spp)
        print(f"SPP schedule ({ds}): {levels}", flush=True)

    for key in scene_keys:
        cfg = summary.get("config", {})
        dataset, scene = parse_scene_key(key)
        ds_name = dataset.name
        scene_spp = spp_levels_for_dataset(ds_name, args.spp)
        if args.resume and key in summary.get("scenes", {}):
            if scene_is_complete(summary["scenes"][key], scene_spp, export_spp, len(frames), config=cfg):
                print(f"=== {key} (skip, complete) ===", flush=True)
                continue
        folder = DATASET_FOLDER[ds_name]
        scene_entry: dict = dict(summary.get("scenes", {}).get(key, {}))
        scene_entry.update({
            "benchmark_version": BENCHMARK_VERSION,
            "dataset_folder": folder,
            "scene": scene,
            "gaussians_k": ply_gaussian_count_k(ds_name, scene),
        })
        scene_entry.setdefault("timing_stochastic", {})
        scene_entry.setdefault("quality", {})
        print(f"=== {key} (#G={scene_entry['gaussians_k']:.1f}k) ===", flush=True)

        gt_metrics_rast: list[ViewMetrics] = []
        gt_metrics_comp: list[ViewMetrics] = []
        comp_render_times: list[float] = []
        stoch_vs_comp: dict[int, list[ViewMetrics]] = {}

        for spp in scene_spp:
            stoch_times: list[float] = []
            stoch_metrics: list[ViewMetrics] = []
            comp_metrics: list[ViewMetrics] = []

            for frame in frames:
                gt = gt_path(dataset, scene, frame)
                if not gt.exists():
                    print(f"  skip frame {frame}: missing GT {gt}", flush=True)
                    continue

                vdir = view_dir(ds_name, scene, spp, frame)
                do_export = (not args.timing_only) and (spp in export_spp)

                # Stochastic
                stoch_exr = vdir / "stochastic.exr"
                stoch_png = vdir / "stochastic.png"
                # Always re-render stochastic for accurate wall-clock timing (deterministic w/ --disable-pixel-jitter).
                if not args.skip_render:
                    SCENES_DIR.mkdir(parents=True, exist_ok=True)
                    vdir.mkdir(parents=True, exist_ok=True)
                    rel = pbrt_out_rel(ds_name, scene, spp, frame, "stochastic.exr")
                    scene_path = SCENES_DIR / f"_report_{scene}_stoch_f{frame}_s{spp}.pbrt"
                    scene_path.write_text(
                        render_scene_pbrt(dataset, scene, frame=frame, mode="stochastic", spp=spp, out_exr=rel),
                        encoding="utf-8",
                    )
                    t_stoch = run_pbrt(args.pbrt, scene_path)
                else:
                    t_stoch = None

                if stoch_exr.exists():
                    if t_stoch is None and args.skip_render:
                        t_stoch = 0.0
                    stoch_times.append(t_stoch)
                    m = metrics_vs_gt(stoch_exr, gt)
                    m.time_s = t_stoch
                    stoch_metrics.append(m)
                    vdir.mkdir(parents=True, exist_ok=True)
                    save_preview_png(stoch_exr, stoch_png)
                    if do_export:
                        shutil.copy2(gt, vdir / "gt.png")
                        rast = rasterizer_path(dataset, scene, frame)
                        if rast.exists():
                            shutil.copy2(rast, vdir / "rasterizer.png")
                            if spp == max(export_spp):
                                gt_metrics_rast.append(metrics_vs_gt(rast, gt))

                # Composite reference (1 spp) — render once per view, copy to other export-spp folders
                if do_export:
                    canon_vdir = view_dir(ds_name, scene, max_export_spp, frame)
                    canon_comp = canon_vdir / "composite.exr"
                    comp_exr = vdir / "composite.exr"
                    t_comp = None
                    if not args.skip_render:
                        canon_vdir.mkdir(parents=True, exist_ok=True)
                        rel_c = pbrt_out_rel(ds_name, scene, max_export_spp, frame, "composite.exr")
                        scene_c = SCENES_DIR / f"_report_{scene}_comp_f{frame}.pbrt"
                        scene_c.write_text(
                            render_scene_pbrt(dataset, scene, frame=frame, mode="reference", spp=1, out_exr=rel_c),
                            encoding="utf-8",
                        )
                        t_comp = run_pbrt(args.pbrt, scene_c)
                        if t_comp is not None:
                            comp_render_times.append(t_comp)
                    if canon_comp.exists() and not comp_exr.exists() and spp != max_export_spp:
                        vdir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(canon_comp, comp_exr)
                    if comp_exr.exists():
                        save_preview_png(comp_exr, vdir / "composite.png")
                        mc = metrics_vs_gt(comp_exr, gt)
                        mc.time_s = t_comp
                        comp_metrics.append(mc)
                        if spp == max_export_spp:
                            gt_metrics_comp.append(mc)

                if do_export and stoch_exr.exists():
                    entries = {"stochastic": stoch_metrics[-1]}
                    comp_path = vdir / "composite.exr"
                    if comp_path.exists():
                        entries["composite"] = comp_metrics[-1] if comp_metrics else metrics_vs_gt(comp_path, gt)
                        stoch_vs_comp.setdefault(spp, []).append(metrics_vs_gt(stoch_exr, comp_path))
                    if (vdir / "rasterizer.png").exists():
                        entries["rasterizer"] = metrics_vs_gt(vdir / "rasterizer.png", gt)
                    write_metrics_txt(vdir / "metrics.txt", gt, entries)

            if stoch_times:
                scene_entry["timing_stochastic"][str(spp)] = {
                    "mean_time_s": float(np.mean(stoch_times)),
                    "per_view_s": stoch_times,
                }
            if stoch_metrics and spp == 1024:
                scene_entry["quality"]["stochastic_1024"] = asdict(aggregate_metrics(stoch_metrics))
            if comp_metrics and spp == max_export_spp:
                comp_agg = asdict(aggregate_metrics(comp_metrics))
                if comp_render_times:
                    comp_agg["time_s"] = float(np.mean(comp_render_times))
                scene_entry["quality"]["composite_1"] = comp_agg
            if spp in stoch_vs_comp:
                scene_entry["quality"][f"stochastic_vs_composite_{spp}"] = asdict(
                    aggregate_metrics(stoch_vs_comp[spp])
                )

        if gt_metrics_rast:
            scene_entry["quality"]["rasterizer"] = asdict(aggregate_metrics(gt_metrics_rast))

        # Stochastic vs composite at every SPP (composite rendered once per view at max_export_spp)
        for spp in scene_spp:
            cross: list[ViewMetrics] = []
            for frame in frames:
                stoch = view_dir(ds_name, scene, spp, frame) / "stochastic.exr"
                comp = view_dir(ds_name, scene, max_export_spp, frame) / "composite.exr"
                if stoch.exists() and comp.exists():
                    cross.append(metrics_vs_gt(stoch, comp))
            if cross:
                scene_entry["quality"][f"stochastic_vs_composite_{spp}"] = asdict(aggregate_metrics(cross))

        summary["scenes"][key] = scene_entry

        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
