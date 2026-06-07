// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/gpu/gaussianupload.h>

#include <pbrt/gpu/util.h>
#include <pbrt/util/lowdiscrepancy.h>

namespace pbrt {

Gaussian3D *UploadGaussiansToDevice(pstd::span<const Gaussian3D> hostGaussians) {
    if (hostGaussians.empty())
        return nullptr;

    Gaussian3D *devicePtr = nullptr;
    size_t bytes = hostGaussians.size() * sizeof(Gaussian3D);
    CUDA_CHECK(cudaMalloc(&devicePtr, bytes));
    CUDA_CHECK(cudaMemcpy(devicePtr, hostGaussians.data(), bytes, cudaMemcpyHostToDevice));
    return devicePtr;
}

void UploadGaussianSobolOffsets(int maxSPP) {
    // Sobol offsets are evaluated on-device via SobolSample in TrigHash.
    // This hook pre-warms the low-discrepancy tables on the host so the first
    // render sample does not pay one-time initialization cost.
    for (int i = 0; i < maxSPP; ++i) {
        (void)SobolSample(i, 0, NoRandomizer());
        (void)SobolSample(i, 1, NoRandomizer());
        (void)SobolSample(i, 2, NoRandomizer());
    }
}

}  // namespace pbrt
