#!/usr/bin/env python3
"""Render GI showcase demos: mic+mirror and drjohnson+reflective mesh."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT.parent
SCENES = ROOT / "scenes"
ASSETS = SCENES / "assets"
RENDER_OUT = ROOT / "renders" / "gi_showcase"
OUT = FINAL / "report_figures" / "gi_showcase"
PBRT = ROOT / "build" / "Release" / "pbrt.exe"
TEAPOT_SRC = FINAL.parent / "HW2" / "scenes" / "teapot.ply"
TEAPOT_ASSET = ASSETS / "props" / "teapot_clean.ply"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_assets import (  # noqa: E402
    DATASETS,
    camera_for_evaluation,
    gaussian_floater_filter,
    stochastic_sample_config,
)
from cameras_json_to_pbrt import camera_to_transform, format_lookat_block  # noqa: E402
from exr_to_png import read_exr  # noqa: E402

MIC_SCALE = (0.358001, 0.358001, 0.358001)
MIC_TRANSLATE_XZ = (0.014071, -0.078613)
MIC_PLY = ASSETS / "3dgs/mic/point_cloud/iteration_30000/point_cloud.ply"

# Utah teapot centre in teapot_clean.ply (converted from HW2/scenes/teapot.ply).
TEAPOT_CENTER = (0.044218, 0.000379, 1.739349)
TEAPOT_MESH = "assets/props/teapot_clean.ply"
TEAPOT_SCALE = 0.08
TEAPOT_VIEW_DIST = 0.85  # metres along view 17 look axis (centre of frame)


def ensure_teapot_asset() -> None:
    """Rewrite HW2 teapot to pbrt-friendly ASCII PLY (assets/props/)."""
    if TEAPOT_ASSET.exists():
        return
    if not TEAPOT_SRC.exists():
        return
    TEAPOT_ASSET.parent.mkdir(parents=True, exist_ok=True)
    lines = TEAPOT_SRC.read_text().splitlines()
    i = lines.index("end_header") + 1
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for line in lines[i:]:
        parts = line.split()
        if len(parts) == 3:
            verts.append((float(parts[0]), float(parts[1]), float(parts[2])))
        elif parts and parts[0].isdigit():
            n = int(parts[0])
            faces.append(tuple(int(x) for x in parts[1 : 1 + n]))
    out = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(verts)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for v in verts:
        out.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for f in faces:
        out.append(str(len(f)) + " " + " ".join(map(str, f)))
    TEAPOT_ASSET.write_text("\n".join(out) + "\n", encoding="utf-8")


def gs_bounds(ply: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    data = ply.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    n = int([line for line in header.splitlines() if line.startswith("element vertex")][0].split()[-1])
    props = [line.split()[2] for line in header.splitlines() if line.startswith("property float")]
    stride = len(props) * 4
    xs, ys, zs = [], [], []
    off = header_end
    for _ in range(n):
        vals = struct.unpack("<" + "f" * len(props), data[off : off + stride])
        xs.append(vals[0])
        ys.append(vals[1])
        zs.append(vals[2])
        off += stride
    pts = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=1)
    return pts.mean(0), pts.min(0), pts.max(0), n


def floater_lines(dataset: str, scene: str) -> str:
    return gaussian_floater_filter(dataset, scene).pbrt_lines()


def mic_translate_y_for_base(floor_y: float = 0.0) -> float:
    """Object-space translate Y so mic PLY base sits on floor_y (after Scale)."""
    _, mn, _, _ = gs_bounds(MIC_PLY)
    return floor_y / MIC_SCALE[1] - mn[1]


def showcase_sample_config(spp: int, min_pixelsamples: int = 4) -> tuple[int, int]:
    """Stochastic config with enough pixel samples for clean showcase stills."""
    spp = max(1, spp)
    for multi in range(min(16, spp), 0, -1):
        if spp % multi == 0 and spp // multi >= min_pixelsamples:
            return spp // multi, multi
    return stochastic_sample_config(spp)


def teapot_world_pos(frame: int = 17, dist: float = TEAPOT_VIEW_DIST) -> tuple[float, float, float]:
    """Place teapot on the view axis, dist metres in front of the camera."""
    cam = camera_for_evaluation(DATASETS["tandt_db"], "drjohnson", frame)
    mat_T, pos, _ = camera_to_transform(cam)
    wfc = np.linalg.inv(mat_T.T)
    look = -wfc[:3, 2]
    look /= np.linalg.norm(look)
    p = pos + look * dist
    return float(p[0]), float(p[1]), float(p[2])


def build_mic_mirror_scene(spp: int = 32, res: int = 512) -> Path:
    """Mic 3DGS beside a vertical conductor mirror — mic reflection visible in mirror."""
    out_exr = "renders/gi_showcase/mic_mirror_stochastic.exr"
    floor_y = 0.0
    mic_ty = mic_translate_y_for_base(floor_y)
    mic_tx = -0.12 + MIC_TRANSLATE_XZ[0]
    mic_tz = MIC_TRANSLATE_XZ[1]
    px_samples, multi = showcase_sample_config(spp)
    text = f"""# Mic 3DGS + vertical mirror wall (Cornell-style GI showcase)

Film "rgb"
    "string filename" ["{out_exr}"]
    "integer xresolution" [{res}] "integer yresolution" [{res}]

Sampler "halton"
    "integer pixelsamples" [{px_samples}]

Integrator "path"
    "integer maxdepth" [8]

Accelerator "bvh"

# Frontal view: mic left, mirror panel right; reflection visible in mirror.
LookAt 0.0 0.55 4.5  0.0 0.55 0.0  0 1 0
Camera "perspective"
    "float fov" [35]

WorldBegin

MakeNamedMaterial "backdrop"
    "string type" ["diffuse"]
    "rgb reflectance" [0.04 0.04 0.045]
MakeNamedMaterial "mirror"
    "string type" ["conductor"]
    "rgb reflectance" [0.82 0.82 0.85]
    "float roughness" [0.0005]
MakeNamedMaterial "gsmat"
    "string type" ["gaussian"]
    "integer sh_degree" [3]

AttributeBegin
    NamedMaterial "backdrop"
    Translate 0 -0.02 0
    Shape "bilinearmesh"
        "point3 P" [ -3 0 -3  3 0 -3  3 0 3  -3 0 3 ]
AttributeEnd

AttributeBegin
    NamedMaterial "mirror"
    Translate 0.42 0.35 0.0
    Rotate -90 0 1 0
    Shape "bilinearmesh"
        "point3 P" [ -1.2 0 -1.2  1.2 0 -1.2  1.2 0 1.2  -1.2 0 1.2 ]
AttributeEnd

AttributeBegin
    NamedMaterial "gsmat"
    Scale {MIC_SCALE[0]:.6f} {MIC_SCALE[1]:.6f} {MIC_SCALE[2]:.6f}
    Translate {mic_tx:.6f} {mic_ty:.6f} {mic_tz:.6f}
    Shape "gaussiancloud"
        "string filename" ["assets/3dgs/mic/point_cloud/iteration_30000/point_cloud.ply"]
        "float sigma_cutoff" [2.828]
        "integer sh_degree" [3]
        "bool use_center_depth" [false]
        "string sampling_mode" ["stochastic"]
        "string internal_accel" ["bvh"]
        "integer multi_samples" [{multi}]
        "bool use_2d_alpha" [false]
AttributeEnd

LightSource "distant"
    "rgb L" [1.0 0.98 0.94]
    "float scale" [5.5]
    "point3 from" [0.25 -0.45 -0.35]
    "point3 to" [0 0.2 0]
LightSource "infinite"
    "rgb L" [0.85 0.88 0.95]
    "float scale" [0.25]
"""
    path = SCENES / "gi_mic_mirror.pbrt"
    path.write_text(text, encoding="utf-8")
    return path


def build_drjohnson_teapot_scene(frame: int = 17, spp: int = 48, scale: float = 1.0) -> Path:
    """Drjohnson room (view frame) + ceramic teapot in centre (mixed geometry GI)."""
    dataset = DATASETS["tandt_db"]
    scene = "drjohnson"
    cam = camera_for_evaluation(dataset, scene, frame)
    w = int(int(cam.get("width", 1332)) * scale)
    h = int(int(cam.get("height", 876)) * scale)
    px_samples, multi = showcase_sample_config(spp)
    lookat = format_lookat_block(None, None, None, 0, cam=cam)
    floater = floater_lines("tandt_db", scene)
    tx, ty, tz = teapot_world_pos(frame)
    cx, cy, cz = TEAPOT_CENTER

    out_exr = "renders/gi_showcase/drjohnson_teapot_stochastic.exr"
    text = f"""# Drjohnson 3DGS + teapot (view {frame:05d}, mixed geometry GI)

Film "rgb"
    "string filename" ["{out_exr}"]
    "integer xresolution" [{w}] "integer yresolution" [{h}]

Sampler "halton"
    "integer pixelsamples" [{px_samples}]

Integrator "path"
    "integer maxdepth" [8]

Accelerator "bvh"

{lookat}

WorldBegin

MakeNamedMaterial "ceramic"
    "string type" ["diffuse"]
    "rgb reflectance" [0.88 0.82 0.74]
MakeNamedMaterial "gsmat"
    "string type" ["gaussian"]
    "integer sh_degree" [3]

AttributeBegin
    NamedMaterial "gsmat"
    Shape "gaussiancloud"
        "string filename" ["assets/tandt_db/drjohnson/point_cloud/iteration_30000/point_cloud.ply"]
        "float sigma_cutoff" [2.828]
        "integer sh_degree" [3]
        "bool use_center_depth" [false]
        "string sampling_mode" ["stochastic"]
        "string internal_accel" ["bvh"]
        "integer multi_samples" [{multi}]
        "bool use_2d_alpha" [false]
{floater}
        "rgb background" [0 0 0]
AttributeEnd

AttributeBegin
    NamedMaterial "ceramic"
    Translate {tx:.3f} {ty:.3f} {tz:.3f}
    Rotate -20 0 1 0
    Scale {TEAPOT_SCALE:.4f} {TEAPOT_SCALE:.4f} {TEAPOT_SCALE:.4f}
    Translate {-cx:.3f} {-cy:.3f} {-cz:.3f}
    Shape "plymesh"
        "string filename" ["{TEAPOT_MESH}"]
AttributeEnd
"""
    path = SCENES / "gi_drjohnson_teapot.pbrt"
    path.write_text(text, encoding="utf-8")
    return path


def run_pbrt(scene: Path, *, nthreads: int = 1, extra: list[str] | None = None) -> None:
    import os

    RENDER_OUT.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(scene.resolve(), ROOT.resolve())
    cmd = [str(PBRT), rel, "--nthreads", str(nthreads)]
    if extra:
        cmd.extend(extra)
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_pbrt_tiled(
    scene: Path,
    exr_out: Path,
    res: int,
    *,
    tile: int = 128,
    nthreads: int = 1,
    base_extra: list[str] | None = None,
) -> None:
    """Render large images as stable 128×128 tiles (avoids mixed-geometry crash on full frame)."""
    acc = np.zeros((res, res, 3), dtype=np.float32)
    rel = scene.relative_to(ROOT)
    for y0 in range(0, res, tile):
        y1 = min(y0 + tile, res)
        for x0 in range(0, res, tile):
            x1 = min(x0 + tile, res)
            cmd = [str(PBRT), str(rel), "--nthreads", str(nthreads)]
            if base_extra:
                cmd.extend(base_extra)
            cmd.extend(["--pixelbounds", f"{x0},{x1},{y0},{y1}"])
            print(f"  tile [{x0}:{x1}, {y0}:{y1}]", flush=True)
            subprocess.run(cmd, check=True, cwd=ROOT)
            partial = read_exr(exr_out)
            th, tw = partial.shape[0], partial.shape[1]
            acc[y0 : y0 + th, x0 : x0 + tw] = partial
    import OpenEXR
    import Imath

    header = OpenEXR.Header(res, res)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {c: Imath.Channel(pt) for c in "RGB"}
    exr = OpenEXR.OutputFile(str(exr_out), header)
    exr.writePixels({c: acc[:, :, i].tobytes() for i, c in enumerate("RGB")})
    exr.close()


def exr_to_png(exr: Path, png: Path, exposure: float = 1.0) -> None:
    from PIL import Image

    rgb = np.clip(read_exr(exr) * exposure, 0.0, None)
    hi = float(np.percentile(rgb, 99.5))
    scale = 0.95 / hi if hi > 1.0 else 1.0
    linear = np.clip(rgb * scale, 0.0, 1.0)
    Image.fromarray((linear * 255.0 + 0.5).astype(np.uint8)).save(png)


def validate_image(png: Path, name: str, *, min_std: float = 0.06) -> dict:
    from PIL import Image

    im = np.asarray(Image.open(png)).astype(np.float64) / 255.0
    mean = float(im.mean())
    std = float(im.std())
    mx = float(im.max())
    black_frac = float((im.max(axis=2) < 0.02).mean())
    ok = std > min_std and black_frac < 0.5 and mx > 0.08
    stats = {"name": name, "mean": mean, "std": std, "max": mx, "black_frac": black_frac, "ok": ok}
    print(f"  {name}: mean={mean:.3f} std={std:.3f} max={mx:.3f} black={black_frac:.2%} ok={ok}")
    return stats


def main() -> int:
    if not PBRT.exists():
        print(f"Missing {PBRT}", file=sys.stderr)
        return 1

    ensure_teapot_asset()
    OUT.mkdir(parents=True, exist_ok=True)
    previews = "--preview" in sys.argv
    extra = ["--disable-pixel-jitter"]
    dj_bounds: list[str] | None = None
    if previews:
        cam = camera_for_evaluation(DATASETS["tandt_db"], "drjohnson", 17)
        pw = int(int(cam.get("width", 1332)) * 0.5)
        ph = int(int(cam.get("height", 876)) * 0.5)
        x0 = max(0, pw // 2 - 64)
        y0 = max(0, ph // 2 - 64)
        dj_bounds = [str(x0), str(x0 + 128), str(y0), str(y0 + 128)]
        extra.extend(["--pixelbounds", "0,128,0,128"])

    print("=== mic on mirror (path-traced GI) ===")
    mic_res = 128 if previews else 384
    mic_spp = 16 if previews else 32
    mic_scene = build_mic_mirror_scene(spp=mic_spp, res=mic_res)
    mic_exr = RENDER_OUT / "mic_mirror_stochastic.exr"
    mic_extra = list(extra)
    if previews:
        run_pbrt(mic_scene, nthreads=1, extra=mic_extra)
    else:
        run_pbrt_tiled(mic_scene, mic_exr, mic_res, tile=128, nthreads=1, base_extra=mic_extra)

    print("=== drjohnson view 17 + teapot ===")
    dj_exr = RENDER_OUT / "drjohnson_teapot_stochastic.exr"
    run_pbrt(
        build_drjohnson_teapot_scene(
            frame=17,
            spp=16 if previews else 64,
            scale=0.5 if previews else 0.67,
        ),
        nthreads=1,
        extra=extra if not previews else ["--disable-pixel-jitter", "--pixelbounds", ",".join(dj_bounds)],
    )

    if previews:
        print("Preview done.")
        return 0

    pairs = [
        (RENDER_OUT / "mic_mirror_stochastic.exr", OUT / "mic_mirror_stochastic.png", 1.0),
        (RENDER_OUT / "drjohnson_teapot_stochastic.exr", OUT / "drjohnson_teapot_stochastic.png", 1.0),
    ]
    print("=== tonemap PNG ===")
    for exr, png, exp in pairs:
        if not exr.exists():
            print(f"Missing {exr}", file=sys.stderr)
            return 1
        shutil.copy2(exr, OUT / exr.name)
        exr_to_png(exr, png, exposure=exp)

    print("=== validation ===")
    stats = [validate_image(png, png.stem, min_std=0.035) for _, png, _ in pairs]
    (OUT / "manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if not all(s["ok"] for s in stats):
        return 2
    print(f"Saved to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
