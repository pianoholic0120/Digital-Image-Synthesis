// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GAUSSIAN_EVAL_H
#define PBRT_GAUSSIAN_EVAL_H

#include <pbrt/pbrt.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

class GaussianCloud;

SampledSpectrum EvaluateGaussianSH(const GaussianCloud *cloud, int gaussianIndex,
                                   const Vector3f &viewDir, SampledWavelengths &wl);

}  // namespace pbrt

#endif  // PBRT_GAUSSIAN_EVAL_H
