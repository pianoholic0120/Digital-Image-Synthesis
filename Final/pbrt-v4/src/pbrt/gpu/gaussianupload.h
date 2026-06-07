// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_GAUSSIANUPLOAD_H
#define PBRT_GPU_GAUSSIANUPLOAD_H

#include <pbrt/pbrt.h>
#include <pbrt/util/plyloader_3dgs.h>

#include <cuda_runtime.h>
#include <vector>

namespace pbrt {

Gaussian3D *UploadGaussiansToDevice(pstd::span<const Gaussian3D> hostGaussians);

// Precompute Sobol offsets for TrigHash (dims 0–2) up to maxSPP samples.
void UploadGaussianSobolOffsets(int maxSPP);

}  // namespace pbrt

#endif  // PBRT_GPU_GAUSSIANUPLOAD_H
