// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_PLYLOADER_3DGS_H
#define PBRT_UTIL_PLYLOADER_3DGS_H

#include <pbrt/pbrt.h>
#include <pbrt/util/vecmath.h>

#include <string>
#include <vector>

namespace pbrt {

struct Gaussian3D {
    Point3f mu;
    Float scale[3];
    Float quat[4];
    Float opacity;
    Float sh[48];

    SquareMatrix<3> sigmaInv;
    SquareMatrix<3> sigma;   // 3D covariance Σ = R diag(s²) Rᵀ (needed for 2D projection)
    Bounds3f aabb;
};

std::vector<Gaussian3D> Load3DGSPly(const std::string &filename, Float sigmaCutoff);

void PrecomputeGaussian(Gaussian3D *g, Float sigmaCutoff);

}  // namespace pbrt

#endif  // PBRT_UTIL_PLYLOADER_3DGS_H
