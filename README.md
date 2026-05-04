# Digital Image Synthesis

Coursework repository for Digital Image Synthesis assignments using PBRT v4.

## Projects

### Project 0 — Install PBRT v4
- Download and build PBRT v4.
- Run example scenes from the official PBRT v4 scenes repository or Benedikt Bitterli's rendering resources.
- Optional: create and render custom scenes.
- References: PBRT v4 input file format and official PBRT exporters/resources.

### Project 1 — Metropolis Sampling and Transforming Normals

**Part I: Metropolis Sampling**
- Implement `ImageCopy` with Metropolis Sampling.
- Evaluate image quality/performance under different samples-per-pixel settings.
- Use any `256x256` image for experiments.

**Part II: Normals (Exercise 3.4)**
- Modify PBRT so `Normal3f` is transformed like `Vector3f` (intentionally incorrect).
- Design a scene that clearly exposes the shading error.
- Compare bugged vs. correct rendering results.

**Submission (PDF via COOL)**
- Part I: implementation details and experimental comparison across sampling rates.
- Part II: code changes, test-scene design, rendered result with bug, rendered result without bug.

See `HW1/README.md` for full reproducibility details and artifacts.

### Project 2 — Uniform Grid
- Implement and evaluate Uniform Grid acceleration in PBRT.

See `HW2/README.md` for build, run, and benchmarking workflow.

## Repository Structure
- `HW1/`: Project 1 implementation, scenes, scripts, and results.
- `HW2/`: Project 2 PBRT codebase, scenes, tools, and outputs.
