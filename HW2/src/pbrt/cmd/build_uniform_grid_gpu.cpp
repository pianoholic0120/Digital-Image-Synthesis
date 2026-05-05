// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/options.h>
#include <pbrt/parser.h>
#include <pbrt/scene.h>
#include <pbrt/util/print.h>
#include <pbrt/wavefront/wavefront.h>

#include <chrono>
#include <string>
#include <vector>

using namespace pbrt;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        Printf("usage: build_uniform_grid_gpu <scenefile>\n");
        return 1;
    }

    std::vector<std::string> filenames;
    for (int i = 1; i < argc; ++i)
        filenames.push_back(argv[i]);

    PBRTOptions options;
    options.useGPU = true;
    options.wavefront = false;
    options.quiet = false;

    InitPBRT(options);

    Printf("Building scene on GPU path: %s\n", filenames[0].c_str());
    Printf("(GPU path uses OptiX/wavefront; CPU Accelerator lines are not used for tracing.)\n");

    auto parseStart = std::chrono::high_resolution_clock::now();
    BasicScene scene;
    BasicSceneBuilder builder(&scene);
    ParseFiles(&builder, filenames);
    auto parseEnd = std::chrono::high_resolution_clock::now();

    auto wavefrontStart = std::chrono::high_resolution_clock::now();
    RenderWavefront(scene);
    auto wavefrontEnd = std::chrono::high_resolution_clock::now();

    auto parseMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(parseEnd - parseStart);
    auto wavefrontMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(wavefrontEnd - wavefrontStart);
    auto totalWorkMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(wavefrontEnd - parseStart);

    Printf("Scene processed (GPU path)\n");
    Printf("Scene parse time: %d ms\n", (int)parseMs.count());
    Printf("Wavefront pipeline time: %d ms\n", (int)wavefrontMs.count());
    Printf("Total work time: %d ms\n", (int)totalWorkMs.count());

    CleanupPBRT();

    return 0;
}
