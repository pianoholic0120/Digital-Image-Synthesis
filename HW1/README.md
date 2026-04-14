# HW1 Reproducibility

This repository contains the deliverables for HW1 Part 1 and Part 2.

## Part 1: Metropolis image copying

### Goal

Compare two Metropolis sampling strategies for image copying:

- `metropolis_1.cpp`: uniform random initialization
- `metropolis_2.cpp`: rejection-based initialization

Both programs take an input image, convert it to grayscale, and generate a copied image at several sample-per-pixel settings.

### Included files

- `part_1/metropolis_1.cpp`: first implementation
- `part_1/metropolis_2.cpp`: second implementation
- `part_1/stb_image.h`, `part_1/stb_image_write.h`: image I/O helpers
- `part_1/source_images/`: original color inputs (`jaguar`, `lion`, `tiger`, `zebra`)
- `part_1/source_images_gray/`: grayscale reference images used for evaluation
- `part_1/copied_images_1/`: outputs from `metropolis_1.cpp`
- `part_1/copied_images_2/`: outputs from `metropolis_2.cpp`
- `part_1/analysis.py`: metric and figure generation script
- `part_1/analysis/`: generated metrics, plots, and markdown summary

### Experiment setup

- Inputs: `jaguar`, `lion`, `tiger`, `zebra` (from [ImageNet](https://www.kaggle.com/datasets/asaniczka/mammals-image-classification-dataset-45-animals/data))
- Samples per pixel: `1`, `4`, `8`, `64`, `256`, `1024`
- Metrics: MSE, MAE, PSNR, SSIM

### Reproducing Part 1

1. Compile `metropolis_1.cpp` and `metropolis_2.cpp` as standalone executables.
2. Run each executable on the four source images for every SPP value.
3. Save the results into `part_1/copied_images_1/` and `part_1/copied_images_2/` using the existing naming scheme.
4. Run `python part_1/analysis.py` to regenerate the CSV files, plots, and markdown summary in `part_1/analysis/`.

### Notes

- The analysis outputs show that quality improves steadily as SPP increases.
- The two initialization strategies produce nearly identical aggregate results in this dataset, with only small differences at low SPP.

## Part 2: PBRT normal transform reproducibility

### Goal

Demonstrate the effect of changing `Transform::operator()(Normal3<T>)` in PBRT v4.

The provided patch intentionally replaces the correct inverse-transpose normal transform with the wrong forward transform, which produces visibly incorrect shading on the test scene.

### Included files

- `part_2/scene/broken-normals.pbrt`: test scene designed to expose the bug
- `part_2/code/normal_transform_original.txt`: original `Normal3` transform
- `part_2/code/normal_transform_modified.txt`: intentionally broken `Normal3` transform
- `part_2/code/normal_transform.patch`: patch against `src/pbrt/util/transform.h`
- `part_2/results/correct-old.png`: reference output from the unmodified PBRT build
- `part_2/results/correct-new.png`: output from the modified PBRT build
- `part_2/scripts/render_compare.ps1`: helper script to render both versions

### Required environment

You need:

- Windows with PowerShell 7 (`pwsh`) or Windows PowerShell
- Visual Studio 2022 Build Tools / Community
- CMake
- CUDA / OptiX setup compatible with your PBRT configuration
- A local checkout of PBRT v4 source code

### Recommended workflow

1. Clone PBRT v4 somewhere on disk, for example:

	`C:\Users\Arthur\Desktop\pbrt-v4`

2. Keep one clean PBRT checkout for the original build.
3. Apply `part_2/code/normal_transform.patch` to a second checkout, worktree, or branch.
4. Build two out-of-source directories in the PBRT repo:

	- original build: `build-old`
	- modified build: `build-new`

5. Render `part_2/scene/broken-normals.pbrt` with both binaries and compare the outputs.

### Exact commands

#### 1) Build and render the original version

Use the unchanged PBRT checkout and build it in a separate directory, then render:

```powershell
.\build-old\Release\pbrt.exe --seed 1 --outfile .\part_2\results\correct-old.exr .\part_2\scene\broken-normals.pbrt
```

#### 2) Build and render the modified version

Apply `part_2/code/normal_transform.patch` to the PBRT source tree, configure a second build directory, and build the modified executable.

Then render:

```powershell
.\build-new\Release\pbrt.exe --seed 1 --outfile .\part_2\results\correct-new.exr .\part_2\scene\broken-normals.pbrt
```

### If you want a clean branch/worktree setup

This is also a good option:

- keep one PBRT checkout as the original version
- create a second branch or a second worktree for the modified version
- apply the patch only in the modified worktree

This avoids mixing the two versions and makes the comparison easier to explain.

### Notes

- The difference is intentionally limited to `Transform::operator()(Normal3<T>)` in `src/pbrt/util/transform.h`.
- The scene is designed to expose the error with a non-uniformly scaled glossy object on a plane.
- The provided PNGs are the expected outputs for quick verification.
