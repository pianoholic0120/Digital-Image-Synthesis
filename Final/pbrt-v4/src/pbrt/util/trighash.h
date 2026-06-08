// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_TRIGHASH_H
#define PBRT_UTIL_TRIGHASH_H

#include <pbrt/pbrt.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

// Storage in trighash.cpp (single TU) — avoids MSVC duplicating thread_local in headers.
void SetGaussianFrameNumber(int frame);
int GetGaussianFrameNumber();

PBRT_CPU_GPU inline Float Frac(Float x) { return x - std::floor(x); }

// Stateless trigonometric hash (Sun et al., EGSR 2025, Eq. 8–10).
PBRT_CPU_GPU inline Float TrigHash1(Float q, Float a, Float b) {
    return Frac(b * std::sin(a * q));
}

PBRT_CPU_GPU inline Float TrigHash2(Float qx, Float qy, Float ax, Float ay, Float b) {
    return Frac(b * std::sin(ax * qx + ay * qy));
}

PBRT_CPU_GPU inline Float TrigHash(Point3f p, int frameNumber) {
    constexpr Float a1 = 91.3458f;
    constexpr Float b1 = 47453.5453f;
    constexpr Float a2x = 12.9898f;
    constexpr Float a2y = 78.233f;
    constexpr Float b2 = 43758.5453f;
    constexpr Float positionScale = 1e-4f;

    Float sobolX = SobolSample(frameNumber, 0, NoRandomizer());
    Float sobolY = SobolSample(frameNumber, 1, NoRandomizer());
    Float sobolZ = SobolSample(frameNumber, 2, NoRandomizer());
    Point3f offset(positionScale * sobolX, positionScale * sobolY, positionScale * sobolZ);
    p += offset;

    Float r1 = TrigHash1(p.z, a1, b1);
    Float xi = TrigHash2(p.x + r1, p.y, a2x, a2y, b2);
    return xi;
}

}  // namespace pbrt

#endif  // PBRT_UTIL_TRIGHASH_H
