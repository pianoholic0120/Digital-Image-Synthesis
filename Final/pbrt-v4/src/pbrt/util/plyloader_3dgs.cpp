// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/plyloader_3dgs.h>

#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sphericalharmonics.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace pbrt {

static Bounds3f ComputeGaussianAABB(const Gaussian3D &g, Float sigmaCutoff) {
    Float w = g.quat[0], x = g.quat[1], y = g.quat[2], z = g.quat[3];
    SquareMatrix<3> R(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y));

    Vector3f halfLocal(sigmaCutoff * g.scale[0], sigmaCutoff * g.scale[1],
                       sigmaCutoff * g.scale[2]);
    Vector3f extent(0, 0, 0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            extent[i] += std::abs(R[i][j]) * halfLocal[j];
    return Bounds3f(g.mu - extent, g.mu + extent);
}

void PrecomputeGaussian(Gaussian3D *g, Float sigmaCutoff) {
    Float w = g->quat[0], x = g->quat[1], y = g->quat[2], z = g->quat[3];
    SquareMatrix<3> R(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y));

    SquareMatrix<3> S = SquareMatrix<3>::Zero();
    S[0][0] = g->scale[0] * g->scale[0];
    S[1][1] = g->scale[1] * g->scale[1];
    S[2][2] = g->scale[2] * g->scale[2];

    SquareMatrix<3> sigma = R * S * Transpose(R);
    pstd::optional<SquareMatrix<3>> sigmaInv = Inverse(sigma);
    if (!sigmaInv)
        ErrorExit("PrecomputeGaussian: singular covariance matrix.");
    g->sigma    = sigma;
    g->sigmaInv = *sigmaInv;
    g->aabb = ComputeGaussianAABB(*g, sigmaCutoff);
}

std::vector<Gaussian3D> Load3DGSPly(const std::string &filename, Float sigmaCutoff) {
    std::string path = ResolveFilename(filename);
    std::ifstream in(path, std::ios::binary);
    if (!in)
        ErrorExit("%s: unable to open 3DGS PLY file.", path);

    std::string line;
    if (!std::getline(in, line) || line != "ply")
        ErrorExit("%s: not a PLY file.", path);

    int vertexCount = 0;
    bool binaryLittle = false;
    std::vector<std::string> properties;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line == "end_header")
            break;
        if (line.rfind("element vertex", 0) == 0) {
            std::istringstream iss(line);
            std::string element, vertex;
            iss >> element >> vertex >> vertexCount;
        } else if (line.rfind("property", 0) == 0) {
            std::istringstream iss(line);
            std::string prop, type, name;
            iss >> prop >> type >> name;
            properties.push_back(name);
        } else if (line.find("format binary_little_endian") != std::string::npos) {
            binaryLittle = true;
        }
    }

    if (!binaryLittle)
        ErrorExit("%s: only binary_little_endian 3DGS PLY is supported.", path);

    std::unordered_map<std::string, int> propIndex;
    for (size_t i = 0; i < properties.size(); ++i)
        propIndex[properties[i]] = int(i);

    auto requireProp = [&](const char *name) -> int {
        auto it = propIndex.find(name);
        if (it == propIndex.end())
            ErrorExit("%s: missing PLY property \"%s\".", path, name);
        return it->second;
    };

    int xIdx = requireProp("x");
    int yIdx = requireProp("y");
    int zIdx = requireProp("z");
    int opacityIdx = requireProp("opacity");
    int s0 = requireProp("scale_0");
    int s1 = requireProp("scale_1");
    int s2 = requireProp("scale_2");
    int r0 = requireProp("rot_0");
    int r1 = requireProp("rot_1");
    int r2 = requireProp("rot_2");
    int r3 = requireProp("rot_3");
    int dc0 = requireProp("f_dc_0");
    int dc1 = requireProp("f_dc_1");
    int dc2 = requireProp("f_dc_2");

    std::vector<int> restIdx;
    for (int i = 0; i < 45; ++i) {
        std::string name = StringPrintf("f_rest_%d", i);
        auto it = propIndex.find(name);
        if (it == propIndex.end())
            ErrorExit("%s: missing PLY property \"%s\".", path, name);
        restIdx.push_back(it->second);
    }

    int propCount = int(properties.size());
    std::vector<Gaussian3D> gaussians(vertexCount);
    std::vector<float> row(propCount);

    // First pass: read all raw data into gaussians (without precompute).
    for (int v = 0; v < vertexCount; ++v) {
        in.read((char *)row.data(), propCount * sizeof(float));
        if (!in)
            ErrorExit("%s: unexpected end of file reading vertex %d.", path, v);

        Gaussian3D &g = gaussians[v];
        g.mu = Point3f(row[xIdx], row[yIdx], row[zIdx]);
        g.scale[0] = std::exp(row[s0]);
        g.scale[1] = std::exp(row[s1]);
        g.scale[2] = std::exp(row[s2]);
        g.quat[0] = row[r0];
        g.quat[1] = row[r1];
        g.quat[2] = row[r2];
        g.quat[3] = row[r3];
        Float qlen = std::sqrt(g.quat[0] * g.quat[0] + g.quat[1] * g.quat[1] +
                               g.quat[2] * g.quat[2] + g.quat[3] * g.quat[3]);
        if (qlen > 0)
            for (int i = 0; i < 4; ++i)
                g.quat[i] /= qlen;
        g.opacity = Sigmoid(row[opacityIdx]);

        g.sh[0] = row[dc0];
        g.sh[1] = row[dc1];
        g.sh[2] = row[dc2];
        // PLY stores rest coefficients channel-first: all 15 R, then 15 G, then 15 B.
        // EvaluateSHRGB expects interleaved RGB triples (sh[i*3+c]).
        // Reorder here so indices match: sh[3 + k*3 + c] = rest coeff k for channel c.
        for (int k = 0; k < 15; ++k) {
            g.sh[3 + k * 3 + 0] = row[restIdx[k]];       // R channel, band index k
            g.sh[3 + k * 3 + 1] = row[restIdx[15 + k]];  // G channel, band index k
            g.sh[3 + k * 3 + 2] = row[restIdx[30 + k]];  // B channel, band index k
        }
    }

    // ---------- Scale sanitisation ----------
    // Two distinct failure modes require scale clamping:
    //
    // (1) DEGENERATE (near-zero) scales: a scale as small as 5e-7 makes
    //     sigmaInv have entries of order 1e12.  Float32 loses precision and
    //     the resulting sigmaInv ceases to be positive definite, yielding
    //     mhd2 < 0.  Those Gaussians wrongly receive alpha=0.99 and produce
    //     fireworks artifacts even on training views.
    //     Fix: clamp each axis to kMinScale = 1e-4.
    //
    // (2) GIANT (floater) scales: a scale of 6.87 world-units makes the 3-D
    //     Mahalanobis sphere enormous.  A background ray 5 units away in the
    //     elongated direction has mhd2=0.53 (< threshold=8), so the Gaussian
    //     contaminates most background pixels.
    //     Fix: clamp each axis to 5× the 99th-percentile maximum scale.
    static constexpr Float kMinScale = 1e-4f;

    // Compute per-Gaussian maximum scale for percentile.
    std::vector<Float> maxScales(vertexCount);
    for (int v = 0; v < vertexCount; ++v)
        maxScales[v] = std::max({gaussians[v].scale[0], gaussians[v].scale[1], gaussians[v].scale[2]});
    std::vector<Float> sortedScales = maxScales;
    std::sort(sortedScales.begin(), sortedScales.end());
    Float p99MaxScale = sortedScales[std::max(0, int(sortedScales.size() * 0.99f) - 1)];
    Float maxScaleLimit = std::max(p99MaxScale * 5.0f, kMinScale);

    int nClampedMin = 0, nClampedMax = 0;
    for (int v = 0; v < vertexCount; ++v) {
        Gaussian3D &g = gaussians[v];
        bool clampedMin = false, clampedMax = false;
        for (int i = 0; i < 3; ++i) {
            if (g.scale[i] < kMinScale) { g.scale[i] = kMinScale; clampedMin = true; }
            if (g.scale[i] > maxScaleLimit) { g.scale[i] = maxScaleLimit; clampedMax = true; }
        }
        if (clampedMin) ++nClampedMin;
        if (clampedMax) ++nClampedMax;
        PrecomputeGaussian(&g, sigmaCutoff);
    }

    if (nClampedMin > 0)
        LOG_VERBOSE("Min-clamped %d near-degenerate Gaussians (scale < %.1e) in %s",
                    nClampedMin, kMinScale, path);
    if (nClampedMax > 0)
        LOG_VERBOSE("Max-clamped %d floater Gaussians (scale > %.4f, p99=%.4f) in %s",
                    nClampedMax, maxScaleLimit, p99MaxScale, path);
    LOG_VERBOSE("Loaded %d 3D Gaussians from %s", vertexCount, path);
    return gaussians;
}

}  // namespace pbrt
