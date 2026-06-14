# 3DGS scene assets

This directory holds **trained 3D Gaussian Splatting PLY files** used by pbrt scenes.
Only extracted data files live here — not the KAIST assignment repository.

## Layout

```
assets/
└── lego/
    └── point_cloud.ply    # NeRF Synthetic "lego" (~322k Gaussians, ~76 MB)
```

Scenes that use `assets/lego/point_cloud.ply`:

- `scenes/lego_3dgs.pbrt`
- `scenes/cornell_3dgs.pbrt`
- `scenes/dof_scene.pbrt`
- `scenes/occlusion.pbrt`

## Provenance

| File | Source | License / terms |
|------|--------|-----------------|
| `lego/point_cloud.ply` | Pre-trained 3DGS model bundled with [KAIST CS479 Assignment 3DGS](https://github.com/KAIST-Visual-AI-Group/CS479-Assignment-3DGS) (`data/lego.ply` in their `data.zip`) | Course assignment data; NeRF Synthetic scenes are from [NeRF authors](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Use with attribution in reports. |

## Re-download (if missing)

From `Final/pbrt-v4`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_lego_assets.ps1
```

Or manually: download [data.zip](https://drive.google.com/file/d/14YVFRR-8L8UVR_UXOe_W-ogNs0IM0572/view?usp=sharing), extract `data/lego.ply`, copy to `scenes/assets/lego/point_cloud.ply`.

## PLY requirements (pbrt loader)

- Format: `binary_little_endian`
- Properties: standard 3DGS export (`x,y,z`, `opacity`, `scale_*`, `rot_*`, `f_dc_*`, `f_rest_0`…`f_rest_44`)
- SH degree: **3** (48 coefficients)

ASCII PLY or mesh PLY (e.g. teapot) will **not** work.
