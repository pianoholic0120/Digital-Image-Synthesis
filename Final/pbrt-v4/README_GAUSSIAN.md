# PBRT 3DGS extension (technical quick reference)

Full project overview, asset setup, training guide, and references: **[../README.md](../README.md)**

## Build

```powershell
cd Final/pbrt-v4
cmake -B build -DPBRT_OPTIX_PATH=""
cmake --build build --config Release --target pbrt_exe
```

## Assets

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_lego_assets.ps1
```

→ `scenes/assets/lego/point_cloud.ply`

## Render

```powershell
.\build\Release\pbrt.exe --spp 4 scenes/lego_3dgs.pbrt
```

## Shape / material

See [../README.md](../README.md#scene-syntax-3dgs) and [scenes/assets/README.md](scenes/assets/README.md).
