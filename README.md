# Digital Image Synthesis

Coursework for **Digital Image Synthesis**: PBRT v4 assignments (HW0–HW2) and a final project implementing [*Stochastic Ray Tracing of Transparent 3D Gaussians*](https://arxiv.org/abs/2504.06598) (Sun, Georgiev, Fei, Hašan; EGSR 2025).

## Repository map

| Path | Project | Summary |
|------|---------|---------|
| [`pbrt-v4/`](pbrt-v4/) | **HW0** | Vanilla PBRT v4 — install, build, and run official example scenes. |
| [`HW1/`](HW1/) | **Project 1** | Metropolis image copying + intentional `Normal3f` transform bug in PBRT. |
| [`HW2/`](HW2/) | **Project 2** | Uniform Grid acceleration experiments in a modified PBRT v4 tree. |
| [`Final/`](Final/) | **Final** | 3DGS as a PBRT shape/material; **CPU-only** benchmarks and report figures. |

Each subdirectory has its own README with build steps and artifacts.

## HW0 — PBRT v4 setup

Goal: download, build, and render with stock PBRT v4.

```powershell
cd pbrt-v4
cmake -B build -DPBRT_OPTIX_PATH=""
cmake --build build --config Release --target pbrt_exe
.\build\Release\pbrt.exe --help
```

Use scenes from the [PBRT v4 scenes repo](https://github.com/mmp/pbrt-v4-scenes) or write your own `.pbrt` files. No course-specific code changes.

## HW1 — Metropolis sampling & normal transforms

**Part I** — Two Metropolis samplers copy grayscale images at multiple SPP levels; metrics and plots are in `HW1/part_1/analysis/`.

**Part II** — A patch to `Transform::operator()(Normal3f)` produces visibly wrong shading on `part_2/scene/broken-normals.pbrt`. Reference and bugged renders are in `part_2/results/`.

See [`HW1/README.md`](HW1/README.md).

## HW2 — Uniform Grid

Modified PBRT v4 with a **CPU-only** workflow: build a uniform grid for a scene (`build_uniform_grid.exe`), then render with `pbrt.exe`. Test scenes and outputs live under `HW2/scenes/` and `HW2/results/`.

See [`HW2/README.md`](HW2/README.md).

## Final — Stochastic 3D Gaussian Ray Tracing

Extends PBRT v4 to ray trace trained **3D Gaussian Splatting** models on the **CPU** using Sun et al.'s stochastic estimator. Benchmarked on **12 scenes** (8 NeRF Synthetic + 4 Tanks & Temples): timing, PSNR vs. composite reference, and comparison to the 3DGS rasterizer.

Report data and figure scripts are under `Final/`; renderer code is in `Final/pbrt-v4/`.

See [`Final/README.md`](Final/README.md).

## Prerequisites (typical)

- Windows 10/11, Visual Studio 2022, CMake ≥ 3.20
- Python 3 + `numpy`, `Pillow`, `OpenEXR`, `Imath` (Final benchmarks/figures; `python-docx` for Word updates)
- **Final only:** trained `point_cloud.ply` per scene (large; kept local — see Final README)
- **3DGS training (optional):** CUDA + Conda per `Final/gaussian-splatting/` (separate from the CPU ray tracer)

## References

- **Sun, X., Georgiev, I., Fei, Y., and Hašan, M.** — *Stochastic Ray Tracing of Transparent 3D Gaussians* (EGSR 2025). [arXiv:2504.06598](https://arxiv.org/abs/2504.06598)
- **Kerbl, B., et al.** — *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023). https://github.com/graphdeco-inria/gaussian-splatting
- **Pharr, M., Jakob, W., and Humphreys, G.** — *Physically Based Rendering* (PBRT v4). https://pbrt.org/

## License

PBRT v4 is Apache 2.0. Course extensions follow the same license where applicable. Third-party datasets (NeRF Synthetic, Tanks & Temples, etc.) retain their original terms.
