// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_SPHERICALHARMONICS_H
#define PBRT_UTIL_SPHERICALHARMONICS_H

#include <pbrt/pbrt.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/math.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

PBRT_CPU_GPU inline Float Sigmoid(Float x) { return 1.f / (1.f + std::exp(-x)); }

// Real SH basis through degree 3 (3DGS / gsplat convention).
PBRT_CPU_GPU inline void EvalSH3(const Vector3f &dir, Float basis[16]) {
    Float x = dir.x, y = dir.y, z = dir.z;
    basis[0] = 0.28209479f;
    basis[1] = -0.48860251f * y;
    basis[2] = 0.48860251f * z;
    basis[3] = -0.48860251f * x;
    basis[4] = 1.09254843f * x * y;
    basis[5] = -1.09254843f * y * z;
    basis[6] = 0.31539157f * (2.f * z * z - x * x - y * y);
    basis[7] = -1.09254843f * x * z;
    basis[8] = 0.54627421f * (x * x - y * y);
    basis[9] = -0.59004358f * y * (3.f * x * x - y * y);
    basis[10] = 2.89061144f * x * y * z;
    basis[11] = -0.45704580f * y * (4.f * z * z - x * x - y * y);
    basis[12] = 0.37317633f * z * (2.f * z * z - 3.f * x * x - 3.f * y * y);
    basis[13] = -0.45704580f * x * (4.f * z * z - x * x - y * y);
    basis[14] = 1.44530572f * z * (x * x - y * y);
    basis[15] = -0.59004358f * x * (x * x - 3.f * y * y);
}

PBRT_CPU_GPU inline int SHNumCoeffs(int degree) { return (degree + 1) * (degree + 1); }

PBRT_CPU_GPU inline SampledSpectrum EvaluateSHColor(const Vector3f &viewDir,
                                                    const Float *sh, int degree,
                                                    SampledWavelengths &lambda) {
    Float basis[16] = {};
    EvalSH3(Normalize(viewDir), basis);
    int nCoeffs = SHNumCoeffs(degree);

    Float r = 0, g = 0, b = 0;
    for (int i = 0; i < nCoeffs; ++i) {
        r += sh[i * 3 + 0] * basis[i];
        g += sh[i * 3 + 1] * basis[i];
        b += sh[i * 3 + 2] * basis[i];
    }

    RGB rgb(Sigmoid(r + 0.5f), Sigmoid(g + 0.5f), Sigmoid(b + 0.5f));
    RGBAlbedoSpectrum spec(*RGBColorSpace::sRGB, rgb);
    return spec.Sample(lambda);
}

}  // namespace pbrt

#endif  // PBRT_UTIL_SPHERICALHARMONICS_H
