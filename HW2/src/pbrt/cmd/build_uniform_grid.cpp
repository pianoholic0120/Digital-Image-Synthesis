// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/cpu/render.h>
#include <pbrt/options.h>
#include <pbrt/parser.h>
#include <pbrt/scene.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>

#include <chrono>
#include <string>
#include <vector>

using namespace pbrt;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        Printf("usage: build_uniform_grid <scenefile>\n");
        return 1;
    }

    std::vector<std::string> filenames;
    for (int i = 1; i < argc; ++i)
        filenames.push_back(argv[i]);

    PBRTOptions options;
    options.useGPU = false;
    options.wavefront = false;
    options.quiet = false;

    InitPBRT(options);

    Printf("Building scene: %s\n", filenames[0].c_str());

    auto buildStart = std::chrono::high_resolution_clock::now();
    BasicScene scene;
    BasicSceneBuilder builder(&scene);
    ParseFiles(&builder, filenames);
    auto buildEnd = std::chrono::high_resolution_clock::now();

    auto renderStart = std::chrono::high_resolution_clock::now();
    RenderCPU(scene);
    auto renderEnd = std::chrono::high_resolution_clock::now();

    auto buildMs = std::chrono::duration_cast<std::chrono::milliseconds>(buildEnd - buildStart);
    auto renderMs = std::chrono::duration_cast<std::chrono::milliseconds>(renderEnd - renderStart);
    auto totalWorkMs = std::chrono::duration_cast<std::chrono::milliseconds>(renderEnd - buildStart);

    Printf("Scene processed\n");
    Printf("Construction time: %d ms\n", (int)buildMs.count());
    Printf("Rendering time: %d ms\n", (int)renderMs.count());
    Printf("Total work time: %d ms\n", (int)totalWorkMs.count());

    CleanupPBRT();

    return 0;
}
