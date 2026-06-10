// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GAUSSIAN_EVAL_H
#define PBRT_GAUSSIAN_EVAL_H

#include <pbrt/pbrt.h>
#include <pbrt/util/color.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

#include <optional>

namespace pbrt {

class GaussianCloud;

inline constexpr int kGaussianCompositeFaceIndex = -2;
inline constexpr int kGaussianMultiSampleFaceIndex = -3;
inline constexpr int kGaussianBackgroundFaceIndex = -4;
inline constexpr Float kGaussianAlphaMinThreshold = 1.f / 255.f;
inline constexpr Float kGaussianAlphaCap = 0.99f;
// Kerbl 3DGS rasterizer frustum cull: p_view.z must exceed this.
inline constexpr Float kGaussianNearPlaneZ = 0.2f;

// When set, RayIntegrator writes this sRGB-linear display RGB directly to RGBFilm.
// Storage lives in gaussian.cpp (one TU) so MSVC does not duplicate thread_local.
void ResetGaussianDirectRGBSample();
void SetGaussianDirectRGBSample(RGB rgb);
pstd::optional<RGB> GetGaussianDirectRGBSample();

// Kerbl 3DGS training uses normalize(mu - camera). Stoch3DGS 3DGRUT uses ray direction.
enum class GaussianSHViewDir { CAM_TO_GAUSSIAN, RAY };

PBRT_CPU_GPU inline Vector3f GaussianSHViewDirection(GaussianSHViewDir mode, const Point3f &mu,
                                                       const Ray &ray) {
    if (mode == GaussianSHViewDir::RAY)
        return Normalize(ray.d);
    return Normalize(Vector3f(mu - ray.o));
}

SampledSpectrum EvaluateGaussianSH(const GaussianCloud *cloud, int gaussianIndex,
                                   const Vector3f &viewDir, SampledWavelengths &wl);

RGB EvaluateGaussianRGB(const GaussianCloud *cloud, int gaussianIndex,
                        const Vector3f &viewDir);

}  // namespace pbrt

#endif  // PBRT_GAUSSIAN_EVAL_H
