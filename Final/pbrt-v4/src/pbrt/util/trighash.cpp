// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/trighash.h>

namespace pbrt {

thread_local int gGaussianFrameNumber = 0;

void SetGaussianFrameNumber(int frame) { gGaussianFrameNumber = frame; }

int GetGaussianFrameNumber() { return gGaussianFrameNumber; }

}  // namespace pbrt
