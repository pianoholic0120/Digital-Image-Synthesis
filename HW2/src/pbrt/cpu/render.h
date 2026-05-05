// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_RENDER_H
#define PBRT_CPU_RENDER_H

#include <pbrt/pbrt.h>

#include <cstdint>

namespace pbrt {

class BasicScene;

void RenderCPU(BasicScene &scene);

// Optional timings (milliseconds): top-level aggregate/accelerator build, then integrator
// rendering only. Pass nullptr to omit.
void RenderCPU(BasicScene &scene, int64_t *outAcceleratorBuildMs,
               int64_t *outIntegratorRenderMs);

}  // namespace pbrt

#endif  // PBRT_CPU_RENDER_H
