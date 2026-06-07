# README — PBRT+ 3D Gaussian Stochastic Ray Tracing

## Build (CPU-only, Windows)

```powershell
cd Final/pbrt-v4
cmake -B build -DPBRT_OPTIX_PATH=""
cmake --build build --config Release --target pbrt_exe
```

GPU build requires CUDA + OptiX; use HW2 `tools/build_gpu_cuda124.ps1` as reference.

## Usage

```powershell
.\build\Release\pbrt.exe scenes\lego_3dgs.pbr --pixelsamples=64
```

Place 3DGS `.ply` files under `scenes/assets/<name>/point_cloud.ply`.

## Scene syntax

```
Shape "gaussiancloud"
    "string filename"   ["assets/lego/point_cloud.ply"]
    "float sigma_cutoff" [2.828]
    "integer sh_degree"  [3]
    "bool use_center_depth" [true]
    "string internal_accel" ["bvh"]   # or "kdtree"
```

Material:

```
Material "gaussian"
    "integer sh_degree" [3]
```

Scene-level accelerator supports PBRT `bvh` and `kdtree` (from HW2/pbrt-v4).

## Core implementation

| Component | File |
|-----------|------|
| PLY loader | `src/pbrt/util/plyloader_3dgs.cpp` |
| TrigHash RNG | `src/pbrt/util/trighash.h` |
| SH evaluation | `src/pbrt/util/sphericalharmonics.h` |
| Gaussian shape | `src/pbrt/gaussian.cpp` |
| Gaussian material | `src/pbrt/materials.h` + `gaussian_eval.h` |
