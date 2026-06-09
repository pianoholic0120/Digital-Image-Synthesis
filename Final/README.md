# Final Project — Stochastic 3D Gaussian Ray Tracing in PBRT v4

Course final implementation extending **PBRT v4** with **3D Gaussian Splatting (3DGS)** as a first-class shape/material, supporting CPU path tracing and optional **OptiX GPU** rendering, plus experiment scripts for ablation studies.

Implementation plan and paper alignment: [`plan.md`](plan.md).

---

## Directory layout

```
Final/
├── README.md                 ← this file
├── plan.md                   ← design doc, experiments, task list
└── pbrt-v4/                  ← modified PBRT v4 + 3DGS extension
    ├── src/pbrt/             ← C++/CUDA core (gaussian, materials, OptiX)
    ├── scenes/               ← .pbrt scenes + assets/
    │   └── assets/lego/point_cloud.ply
    ├── gt/                   ← optional reference images for PSNR
    ├── scripts/              ← ablation, SPP sweep, plotting, asset setup
    ├── tools/                ← GPU build scripts
    ├── build/                ← CPU Release build (local)
    └── build-gpu/            ← OptiX build (local)
```

---

## What this project does

1. **Loads** standard 3DGS `point_cloud.ply` (binary, SH degree 3).
2. **Builds** an internal BVH/Kd-tree over Gaussian AABBs.
3. **Intersects** rays stochastically (TrigHash accept/reject) along 1D Gaussian depth.
4. **Shades** via spherical harmonics (`Material "gaussian"`).
5. **Integrates** with full PBRT lighting (path/volpath, mirrors, glass, DOF, etc.).

Unlike rasterization-based 3DGS viewers, this renderer supports **global illumination**, **non-pinhole cameras**, and **arbitrary PBRT materials** mixed with Gaussians.

---

## Quick start

### 1. Build CPU renderer

```powershell
cd Final/pbrt-v4
cmake -B build -DPBRT_OPTIX_PATH=""
cmake --build build --config Release --target pbrt_exe
```

### 2. Install lego 3DGS asset

If `scenes/assets/lego/point_cloud.ply` is missing:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_lego_assets.ps1
```

This downloads only `lego.ply` from the [KAIST CS479 data.zip mirror](https://drive.google.com/file/d/14YVFRR-8L8UVR_UXOe_W-ogNs0IM0572/view?usp=sharing) and installs it as `point_cloud.ply`. **No assignment source code or git history is added.**

### 3. Smoke tests

```powershell
# No external assets
.\build\Release\pbrt.exe scenes/test_unit_sphere.pbrt

# 3DGS scenes (need lego PLY) — use crop for fast check
.\build\Release\pbrt.exe --spp 4 --pixelbounds 0,64,0,64 scenes/lego_3dgs.pbrt
.\build\Release\pbrt.exe --spp 4 --pixelbounds 0,64,0,64 scenes/cornell_3dgs.pbrt
.\build\Release\pbrt.exe --spp 4 scenes/dof_scene.pbrt
.\build\Release\pbrt.exe --spp 4 scenes/occlusion.pbrt
```

Full HD renders (800×800, 64+ spp) with ~322k Gaussians can take **many minutes** on CPU.

### 4. GPU build (optional)

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_gpu.ps1
.\build-gpu\Release\pbrt.exe --gpu --spp 4 scenes/lego_3dgs.pbrt
```

Requires CUDA 12.x + OptiX 7.7 (see `tools/` and HW2 scripts).

---

## Scenes

### Ready without extra assets (20 scenes)

All `test_*.pbrt`, `broken-normals*.pbrt`, `test_sphere.pbrt` — procedural geometry, for build/regression.

### 3DGS scenes (require `assets/lego/point_cloud.ply`)

| Scene | Purpose |
|-------|---------|
| `lego_3dgs.pbrt` | Minimal 3DGS + distant light; ablation baseline |
| `cornell_3dgs.pbrt` | Cornell box + scaled lego + glass sphere + mirror wall |
| `dof_scene.pbrt` | Depth of field + background spheres |
| `occlusion.pbrt` | Glass panel in front of 3DGS |

### Still needs separate training (not bundled)

| Scene | Asset |
|-------|-------|
| `soft_shadow.pbrt` | `assets/room/point_cloud.ply` (Mip-NeRF 360 **room**) |

---

## Scene syntax (3DGS)

```pbrt
Shape "gaussiancloud"
    "string filename"   ["assets/lego/point_cloud.ply"]
    "float sigma_cutoff" [2.828]
    "integer sh_degree"  [3]
    "bool use_center_depth" [true]
    "string internal_accel" ["bvh"]   # or "kdtree"

Material "gaussian"
    "integer sh_degree" [3]
```

Named materials (Cornell / DOF scenes):

```pbrt
MakeNamedMaterial "gsmat"
    "string type" ["gaussian"]
    "integer sh_degree" [3]
NamedMaterial "gsmat"
```

---

## Experiments & ablation

Scripts under `pbrt-v4/scripts/`:

| Script | Purpose |
|--------|---------|
| `setup_lego_assets.ps1` | Install lego PLY |
| `ablation_runner.py` + `ablations.yaml` | Batch ablation variants |
| `sigma_sweep.py` | σ cutoff sweep |
| `render_spp_sweep.ps1` | SPP convergence |
| `eval_psnr.py` | PSNR/SSIM vs reference |
| `plot_ablations.py`, `plot_convergence.py` | Figures |

Example:

```powershell
pip install pyyaml numpy imageio
python scripts/ablation_runner.py --config scripts/ablations.yaml --pbrt build/Release/pbrt.exe
```

Ablation knobs: `use_center_depth`, `sigma_cutoff`, `internal_accel`, `sh_degree`, `--spp`, scene-level `Integrator` / `Accelerator`.

---

## Training your own 3DGS for this codebase

Use the **official Inria trainer** — not the KAIST rasterization homework (that repo is only a convenient **pre-trained lego PLY** source).

### Step 1 — Clone official 3DGS

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
pip install -r requirements.txt
```

### Step 2 — Prepare input data

**Option A — NeRF Synthetic** (object-centric, like lego):

1. Download a scene from [NeRF Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) (e.g. `lego/`).
2. Folder must contain `transforms_train.json`, `transforms_test.json`, `train/`, `test/` images.

**Option B — COLMAP / Mip-NeRF 360** (room, garden, bicycle, …):

1. Download images from [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) or [3DGS project page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
2. Run COLMAP / use the project's `convert.py` as documented in the gaussian-splatting README until you have a `source_path` with `sparse/` and `images/`.

### Step 3 — Train

```bash
python train.py -s /path/to/dataset -m output/my_scene --iterations 30000
```

Default SH degree is **3** — matches this PBRT loader.

### Step 4 — Copy output PLY

Official output:

```
output/my_scene/point_cloud/iteration_30000/point_cloud.ply
```

Install into PBRT:

```
Final/pbrt-v4/scenes/assets/<scene_name>/point_cloud.ply
```

Example for a new `room` scene:

```
scenes/assets/room/point_cloud.ply
```

Update `soft_shadow.pbrt` (or duplicate `lego_3dgs.pbrt` and change the `filename` parameter).

### Step 5 — Adapt the `.pbrt` scene

- **Scale / translate** the `Shape "gaussiancloud"` block — NeRF objects are ~unit-sized; Cornell uses `Scale 0.4`.
- **Camera** — adjust `LookAt` / `fov` so Gaussians are in frame.
- **Coordinate system** — same as COLMAP/NeRF training; if the model appears upside-down, add `Rotate` in the shape's `AttributeBegin` block.

### PLY compatibility checklist

| Requirement | Official 3DGS `point_cloud.ply` |
|-------------|----------------------------------|
| Binary little-endian | Yes |
| `f_rest_0` … `f_rest_44` | Yes (degree 3) |
| Log-scale + logit opacity in file | Yes (loader applies `exp` / `sigmoid`) |

If you train with `--sh_degree 0` or `1`, the PLY will have fewer `f_rest_*` properties and **will not load** until you retrain at degree 3 or extend `plyloader_3dgs.cpp`.

---

## Key implementation files

| Component | Path |
|-----------|------|
| PLY loader | `src/pbrt/util/plyloader_3dgs.cpp` |
| Gaussian shape + BVH | `src/pbrt/gaussian.cpp` |
| Gaussian material | `src/pbrt/materials.h`, `gaussian_eval.h` |
| TrigHash RNG | `src/pbrt/util/trighash.h` |
| SH evaluation | `src/pbrt/util/sphericalharmonics.h` |
| OptiX GPU | `src/pbrt/gpu/optix/gaussian_optix.cu` |

More detail: [`pbrt-v4/README_GAUSSIAN.md`](pbrt-v4/README_GAUSSIAN.md).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `unable to open 3DGS PLY` | Run `scripts/setup_lego_assets.ps1` or copy PLY manually |
| `only binary_little_endian` | Re-export from official 3DGS; don't use ASCII PLY |
| `missing f_rest_*` | Train with SH degree 3 |
| No progress bar | Remove `--quiet` |
| Render very slow | Use `--spp 4 --pixelbounds …` or GPU `--gpu` |
| Empty / wrong framing | Add `Scale` / `Translate` / adjust camera |

---

## References

### Papers

1. **Kerbl, Kopanas, Leimkühler, Drettakis** — *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.  
   https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

2. **Mildenhall et al.** — *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, ECCV 2020.  
   https://www.matthewtancik.com/nerf

3. **Barron et al.** — *Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields*, CVPR 2022.  
   https://jonbarron.info/mipnerf360/

4. **Pharr, Jakob, Humphreys** — *Physically Based Rendering: From Theory to Implementation* (PBRT v4).  
   https://pbrt.org/

### Software & data

5. **graphdeco-inria/gaussian-splatting** — official training code.  
   https://github.com/graphdeco-inria/gaussian-splatting

6. **PBRT v4 source** — base renderer.  
   https://github.com/mmp/pbrt-v4

7. **NeRF Synthetic dataset** (CC BY 4.0).  
   https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1

8. **KAIST CS479 Assignment 3DGS** — course assignment; **lego.ply** and NeRF test images used as pre-trained asset only (not vendored as code).  
   https://github.com/KAIST-Visual-AI-Group/CS479-Assignment-3DGS  
   Data mirror: https://drive.google.com/file/d/14YVFRR-8L8UVR_UXOe_W-ogNs0IM0572/view?usp=sharing

9. **torch-splatting** (referenced by KAIST assignment).  
   https://github.com/hbb1/torch-splatting

### Course / repo context

10. **Digital Image Synthesis** — HW2 uniform grid / HW0–HW2 assignments in parent repo.

---

## License

PBRT v4 is Apache 2.0 (`pbrt-v4/LICENSE.txt`). Project extensions follow the same license. Third-party datasets and the KAIST assignment data are subject to their respective terms — cite them in your report.
