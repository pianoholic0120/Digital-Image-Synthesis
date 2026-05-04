# HW2 - Uniform Grid Acceleration in PBRT-v4 (CPU Build)

This project contains a modified `pbrt-v4` codebase for HW2, focusing on **Uniform Grid** acceleration experiments and reproducible CPU-only testing.

## Project Scope

- Implement and test Uniform Grid related changes in `pbrt-v4`.
- Build and run the project on CPU only (no OptiX/GPU dependency required).
- Evaluate behavior on included test scenes.

## Repository Layout

- `src/`: PBRT source code (including your HW2 modifications).
- `scenes/`: scene files for acceleration experiments.
- `tools/`: utility programs.
- `results/`: output images and experiment artifacts.
- `benchmark_accelerators.py`: helper script for accelerator benchmarking.

## Prerequisites

Recommended environment (Windows):

- Windows 10/11
- Visual Studio 2022 (MSVC toolchain, C++ workload)
- CMake >= 3.20
- Git

Notes:

- This setup intentionally uses **CPU-only** build flow.
- CUDA may still be detected by CMake, but GPU compilation remains disabled when `PBRT_OPTIX_PATH` is empty.

## Reproducible Build (CPU-only)

From the project root:

```powershell
cmake -S . -B build-cpu-only -DPBRT_OPTIX_PATH=""
cmake --build build-cpu-only --config Release -j 12
```

Expected binaries are generated in:

- `build-cpu-only/Release/pbrt.exe`
- `build-cpu-only/Release/build_uniform_grid.exe`
- `build-cpu-only/Release/imgtool.exe`

## Reproducible Test Workflow

### 1) Build Uniform Grid for a scene

```powershell
& ".\build-cpu-only\Release\build_uniform_grid.exe" ".\scenes\accel-compare-stadium.pbrt"
```

### 2) Render with CPU PBRT

```powershell
& ".\build-cpu-only\Release\pbrt.exe" --quick --spp 4 --outfile ".\results\stadium_cpu_uniformgrid.exr" ".\scenes\accel-compare-stadium.pbrt"
```

### 3) Optional: compare another scene

```powershell
& ".\build-cpu-only\Release\build_uniform_grid.exe" ".\scenes\accel-compare-dense.pbrt"
& ".\build-cpu-only\Release\pbrt.exe" --quick --spp 4 --outfile ".\results\dense_cpu_uniformgrid.exr" ".\scenes\accel-compare-dense.pbrt"
```

## What To Verify

- Build completes without errors in `Release`.
- `build_uniform_grid.exe` runs successfully and reports scene processing.
- `pbrt.exe` renders and writes `.exr` output files into `results/`.
- Runtime and output differences across scenes are reasonable for Uniform Grid behavior.

## Brief Technical Explanation

`PBRT-v4` supports multiple acceleration structures for ray-scene intersection.  
In this HW2 project, Uniform Grid related modifications are tested through a dedicated tool (`build_uniform_grid`) and then validated via CPU rendering runs.  
This separates **acceleration structure construction** from **final rendering**, making debugging and performance checks easier.

## References

1. Teapot mesh: [https://github.com/gnomeby/canvas3D/blob/master/teapot.ply](https://github.com/gnomeby/canvas3D/blob/master/teapot.ply)
2. PBRT-v2 accelerators: [https://github.com/mmp/pbrt-v2/tree/master/src/accelerators](https://github.com/mmp/pbrt-v2/tree/master/src/accelerators)
3. PBRT-v4 scenes (Barcelona Pavilion section): [https://github.com/mmp/pbrt-v4-scenes?tab=readme-ov-file#barcelona-pavilion](https://github.com/mmp/pbrt-v4-scenes?tab=readme-ov-file#barcelona-pavilion)
4. PBRT-v4 repository: [https://github.com/mmp/pbrt-v4?tab=readme-ov-file](https://github.com/mmp/pbrt-v4?tab=readme-ov-file)
