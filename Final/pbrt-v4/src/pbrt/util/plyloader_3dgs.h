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

// Load-time filters mirroring 3DGS training / rasterizer behaviour.
struct Load3DGSPlyOptions {
    Float sigmaCutoff = 2.828f;
    // Drop Gaussians below this opacity (0 = keep all).  1/255 matches rasterizer alpha cutoff.
    Float minOpacity = 0.f;
    // Clamp per-axis scale to percentile × this factor (default 5, same as before).
    Float maxScalePercentileFactor = 5.f;
    // Remove distant Gaussians with extreme SH DC (0 = off).  Targets edge floaters without retraining.
    Float pruneOutlierDcThreshold = 0.f;
    // Distance percentile (0–1) paired with pruneOutlierDcThreshold; default 0.85.
    Float pruneOutlierDistanceFrac = 0.85f;
};

std::vector<Gaussian3D> Load3DGSPly(const std::string &filename, Float sigmaCutoff);
std::vector<Gaussian3D> Load3DGSPly(const std::string &filename,
                                    const Load3DGSPlyOptions &options);

void PrecomputeGaussian(Gaussian3D *g, Float sigmaCutoff);

}  // namespace pbrt

#endif  // PBRT_UTIL_PLYLOADER_3DGS_H
