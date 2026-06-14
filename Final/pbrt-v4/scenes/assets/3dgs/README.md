# Self-trained 3DGS assets (NeRF Synthetic)

Each subdirectory is one official 3DGS training output copied from `output/<scene>/`.

## Layout (per scene)

```
<scene>/
  point_cloud/iteration_30000/point_cloud.ply   # PBRT input
  cameras.json                                 # train+test cameras (300 entries)
  metrics_composited_30000.json                # 3DGS raster composited PSNR
  test/ours_30000/composited/gt/00000.png      # GT (200 test views)
  test/ours_30000/composited/renders/00000.png # 3DGS composited reference
```

## Scenes

| Scene | 3DGS composited PSNR | white_background |
|-------|---------------------|------------------|
| chair | 32.57 dB | yes |
| drums | 23.77 dB | yes |
| ficus | 24.85 dB | yes |
| hotdog | 36.13 dB | yes |
| lego | 33.02 dB | yes |
| materials | 25.92 dB | yes |
| mic | 30.14 dB | yes |
| ship | 30.46 dB | yes |

## PBRT scenes

Auto-generated under `scenes/3dgs/`:

```powershell
python scripts/generate_3dgs_scenes.py --frame 0 --spp 64
```

Benchmark vs composited GT:

```powershell
python scripts/benchmark_3dgs_pipeline.py --reference-only --frames 1
python scripts/benchmark_3dgs_pipeline.py --frames 10 --spp 64   # slower
```

Camera index `N` in `cameras.json` matches GT file `{N:05d}.png` for `N = 0..199`.
