#!/usr/bin/env python3
"""Shared asset layout + scene generation for 3dgs and tandt_db benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cameras_json_to_pbrt import (  # noqa: E402
    camera_to_lookat,
    format_gaussiancloud_camera_params,
    format_lookat_block,
)


@dataclass(frozen=True)
class Dataset:
    name: str
    root: Path
    gt_subpath: str  # under test/ours_30000/
    renders_subdir: str

    def scene_names(self) -> list[str]:
        if not self.root.exists():
            return []
        out: list[str] = []
        for p in sorted(self.root.iterdir()):
            if not p.is_dir():
                continue
            ply = p / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            cam = p / "cameras.json"
            if ply.exists() and cam.exists():
                out.append(p.name)
        return out

    def scene_dir(self, scene: str) -> Path:
        return self.root / scene

    def ply_rel(self, scene: str) -> str:
        return f"../assets/{self.name}/{scene}/point_cloud/iteration_30000/point_cloud.ply"

    def gt_path(self, scene: str, frame: int) -> Path:
        return self.scene_dir(scene) / "test" / "ours_30000" / self.gt_subpath / f"{frame:05d}.png"

    def white_background(self, scene: str) -> bool:
        metrics = self.scene_dir(scene) / "metrics_composited_30000.json"
        if metrics.exists():
            return bool(json.loads(metrics.read_text()).get("white_background", True))
        return False

    def training_psnr(self, scene: str) -> float | None:
        for name in ("metrics_composited_30000.json", "results.json"):
            m = self.scene_dir(scene) / name
            if not m.exists():
                continue
            data = json.loads(m.read_text())
            if name == "results.json":
                block = data.get("ours_30000", {})
                if "PSNR" in block:
                    return float(block["PSNR"])
            else:
                test = data.get("test", {})
                if "psnr_mean" in test:
                    return float(test["psnr_mean"])
        return None


DATASETS: dict[str, Dataset] = {
    "3dgs": Dataset(
        name="3dgs",
        root=ROOT / "scenes" / "assets" / "3dgs",
        gt_subpath="composited/gt",
        renders_subdir="renders/3dgs",
    ),
    "tandt_db": Dataset(
        name="tandt_db",
        root=ROOT / "scenes" / "assets" / "tandt_db",
        gt_subpath="gt",
        renders_subdir="renders/tandt_db",
    ),
}


def all_scene_keys(datasets: list[str] | None = None) -> list[str]:
    names = datasets or list(DATASETS)
    keys: list[str] = []
    for ds in names:
        if ds not in DATASETS:
            raise ValueError(f"unknown dataset {ds!r}")
        for scene in DATASETS[ds].scene_names():
            keys.append(f"{ds}/{scene}")
    return keys


def parse_scene_key(key: str) -> tuple[Dataset, str]:
    if "/" not in key:
        raise ValueError(f"expected dataset/scene, got {key!r}")
    ds_name, scene = key.split("/", 1)
    return DATASETS[ds_name], scene


@dataclass(frozen=True)
class GaussianFloaterFilter:
    """Render-time PLY filters (no retraining) to suppress edge floaters."""

    min_opacity: float
    max_scale_percentile: float
    prune_outlier_dc: float
    prune_outlier_distance_frac: float

    def pbrt_lines(self) -> str:
        return (
            f'        "float min_opacity" [{self.min_opacity}]\n'
            f'        "float max_scale_percentile" [{self.max_scale_percentile}]\n'
            f'        "float prune_outlier_dc" [{self.prune_outlier_dc}]\n'
            f'        "float prune_outlier_distance_frac" [{self.prune_outlier_distance_frac}]'
        )


# Synthetic NeRF / Blender 3DGS objects: white background, common edge floaters.
_3DGS_FLOATER_DEFAULT = GaussianFloaterFilter(
    min_opacity=0.008,
    max_scale_percentile=2.8,
    prune_outlier_dc=3.8,
    prune_outlier_distance_frac=0.84,
)

# Per-scene micro-tuning (still no retraining).
_3DGS_FLOATER_OVERRIDES: dict[str, GaussianFloaterFilter] = {
    # Lego: many bright distant SH outliers.
    "lego": GaussianFloaterFilter(0.008, 2.8, 3.2, 0.78),
    # Thin / reflective geometry.
    "materials": GaussianFloaterFilter(0.005, 2.8, 3.8, 0.82),
    "mic": GaussianFloaterFilter(0.005, 3.0, 4.0, 0.83),
    # Larger organic meshes — slightly looser scale clamp.
    "ficus": GaussianFloaterFilter(0.005, 3.5, 4.5, 0.88),
    "ship": GaussianFloaterFilter(0.005, 3.5, 4.5, 0.88),
}

# TandT: indoor scenes tolerate mild filters; outdoor train/truck need stronger pruning.
_TANDT_FLOATER_DEFAULT = GaussianFloaterFilter(
    min_opacity=1.0 / 255.0,
    max_scale_percentile=5.0,
    prune_outlier_dc=0.0,
    prune_outlier_distance_frac=0.85,
)

_TANDT_FLOATER_OVERRIDES: dict[str, GaussianFloaterFilter] = {
    # Outdoor TnT: PLY has many giant semi-opaque floaters (see train/truck stats).
    "train": GaussianFloaterFilter(0.005, 3.0, 3.5, 0.82),
    "truck": GaussianFloaterFilter(0.005, 3.0, 3.5, 0.82),
}


def gaussian_floater_filter(dataset_name: str, scene: str) -> GaussianFloaterFilter:
    if dataset_name == "3dgs":
        return _3DGS_FLOATER_OVERRIDES.get(scene, _3DGS_FLOATER_DEFAULT)
    return _TANDT_FLOATER_OVERRIDES.get(scene, _TANDT_FLOATER_DEFAULT)


def use_2d_alpha(dataset_name: str) -> bool:
    """Rasterizer-style 2D projection alpha works for NeRF Synthetic; T&T needs 3D alpha."""
    return dataset_name == "3dgs"


def stochastic_sample_config(spp: int, max_multi: int = 64) -> tuple[int, int]:
    """Map total spp to (pixelsamples, multi_samples) per paper §3.5 / Table 5.

    Prefer the largest multi_samples (fewer BVH traversals per pixel) that divides spp.
    """
    spp = max(1, spp)
    multi = 1
    for m in range(min(max_multi, spp), 0, -1):
        if spp % m == 0:
            multi = m
            break
    return spp // multi, multi


def load_camera(dataset: Dataset, scene: str, frame: int) -> dict:
    cams = json.loads((dataset.scene_dir(scene) / "cameras.json").read_text())
    if frame < 0 or frame >= len(cams):
        raise IndexError(f"{dataset.name}/{scene} frame {frame} out of range 0..{len(cams)-1}")
    return cams[frame]


def camera_for_evaluation(dataset: Dataset, scene: str, frame: int) -> dict:
    """Match film + gaussiancloud intrinsics to held-out GT resolution.

    Some T&T scenes store cameras.json at 2× the GT/rasterizer PNG size (train, truck).
    """
    cam = dict(load_camera(dataset, scene, frame))
    gt = dataset.gt_path(scene, frame)
    if not gt.exists():
        return cam
    from PIL import Image

    with Image.open(gt) as im:
        gt_w, gt_h = im.size
    cam_w = int(cam.get("width", gt_w))
    cam_h = int(cam.get("height", gt_h))
    if gt_w == cam_w and gt_h == cam_h:
        return cam
    sx, sy = gt_w / cam_w, gt_h / cam_h
    cam["width"] = gt_w
    cam["height"] = gt_h
    for key in ("fx", "fl_x"):
        if key in cam:
            cam[key] = float(cam[key]) * sx
    for key in ("fy", "fl_y"):
        if key in cam:
            cam[key] = float(cam[key]) * sy
    if "cx" in cam:
        cam["cx"] = float(cam["cx"]) * sx
    if "cy" in cam:
        cam["cy"] = float(cam["cy"]) * sy
    return cam


def render_scene_pbrt(
    dataset: Dataset,
    scene: str,
    *,
    frame: int,
    mode: str,
    spp: int,
    out_exr: str,
) -> str:
    cam = camera_for_evaluation(dataset, scene, frame)
    w = int(cam.get("width", 800))
    h = int(cam.get("height", 800))
    eye, target, up, fov = camera_to_lookat(cam)
    lookat = format_lookat_block(eye, target, up, fov, cam=cam)
    wb = dataset.white_background(scene)
    bg = "1 1 1" if wb else "0 0 0"
    ply_rel = dataset.ply_rel(scene)

    if mode == "reference":
        sampling = "composite"
        samples = 1
        multi_samples = 1
        use_center = True
    elif mode == "stochastic":
        sampling = "stochastic"
        samples, multi_samples = stochastic_sample_config(spp)
        # Secondary rays use OursMean; primary 2D rays composite inside IntersectBVH.
        use_center = False
    else:
        raise ValueError(mode)

    u2d = use_2d_alpha(dataset.name)
    # Camera params drive the 2D tile grid (candidate lookup) even when alpha stays 3D.
    cam_params_block = "\n" + format_gaussiancloud_camera_params(cam)
    floater = gaussian_floater_filter(dataset.name, scene)

    gt_rel = dataset.gt_path(scene, frame).relative_to(ROOT).as_posix()
    return f"""# Auto-generated {dataset.name}/{scene} frame {frame} ({mode})

Film "rgb"
    "string filename" ["{out_exr}"]
    "integer xresolution" [{w}] "integer yresolution" [{h}]

Sampler "halton"
    "integer pixelsamples" [{samples}]

Integrator "path"

Accelerator "bvh"

{lookat}

WorldBegin

Material "gaussian"
    "integer sh_degree" [3]
    "float emission_scale" [1.0]

AttributeBegin
    Material "gaussian"
        "integer sh_degree" [3]
        "float emission_scale" [1.0]
    Shape "gaussiancloud"
        "string filename" ["{ply_rel}"]
        "float sigma_cutoff" [2.828]
        "integer sh_degree" [3]
        "bool use_center_depth" [{"true" if use_center else "false"}]
        "string sampling_mode" ["{sampling}"]
        "string internal_accel" ["bvh"]
        "integer multi_samples" [{multi_samples}]
        "bool use_2d_alpha" [{"true" if u2d else "false"}]
{floater.pbrt_lines()}
        "rgb background" [{bg}]{cam_params_block}
AttributeEnd
"""
