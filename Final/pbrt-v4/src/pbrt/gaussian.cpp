// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/shapes.h>
#include <pbrt/gaussian.h>
#include <pbrt/gaussian_eval.h>

#include <pbrt/base/material.h>
#include <pbrt/bsdf.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/sphericalharmonics.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/trighash.h>

#include <algorithm>

namespace pbrt {

thread_local pstd::optional<RGB> gGaussianDirectRGBSample;

void ResetGaussianDirectRGBSample() { gGaussianDirectRGBSample.reset(); }

void SetGaussianDirectRGBSample(RGB rgb) { gGaussianDirectRGBSample = rgb; }

pstd::optional<RGB> GetGaussianDirectRGBSample() { return gGaussianDirectRGBSample; }

STAT_COUNTER("Geometry/GaussianClouds", nGaussianClouds);
STAT_COUNTER("Geometry/GaussianPrimitives", nGaussianPrimitives);

GaussianCloud *GaussianCloud::Create(const Transform *renderFromObject,
                                     const Transform *objectFromRender,
                                     bool reverseOrientation,
                                     const ParameterDictionary &parameters,
                                     const FileLoc *loc, Allocator alloc) {
    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));
    if (filename.empty())
        ErrorExit(loc, "gaussiancloud: \"filename\" parameter not specified.");

    Float sigmaCutoff = parameters.GetOneFloat("sigma_cutoff", 2.828f);
    int shDegree = parameters.GetOneInt("sh_degree", 3);
    bool useCenterDepth = parameters.GetOneBool("use_center_depth", false);
    std::string accelName = parameters.GetOneString("internal_accel", "bvh");
    InternalAccel accel = InternalAccel::BVH;
    if (accelName == "kdtree")
        accel = InternalAccel::KDTREE;
    else if (accelName == "brute")
        accel = InternalAccel::BRUTE;
    std::string samplingName = parameters.GetOneString("sampling_mode", "stochastic");
    SamplingMode samplingMode = SamplingMode::STOCHASTIC;
    if (samplingName == "composite")
        samplingMode = SamplingMode::COMPOSITE;
    else if (samplingName != "stochastic")
        ErrorExit(loc, "gaussiancloud: unknown sampling_mode \"%s\".", samplingName);
    int multiSamples = std::max(1, parameters.GetOneInt("multi_samples", 1));
    std::string shViewName = parameters.GetOneString("sh_viewdir", "cam_to_gaussian");
    GaussianSHViewDir shViewDir = GaussianSHViewDir::CAM_TO_GAUSSIAN;
    if (shViewName == "ray")
        shViewDir = GaussianSHViewDir::RAY;
    else if (shViewName != "cam_to_gaussian")
        ErrorExit(loc, "gaussiancloud: unknown sh_viewdir \"%s\".", shViewName);
    std::vector<RGB> background = parameters.GetRGBArray("background");
    RGB backgroundColor =
        background.empty() ? RGB(0.f, 0.f, 0.f) : background[0];

    std::vector<Gaussian3D> gaussians = Load3DGSPly(filename, sigmaCutoff);
    if (gaussians.empty())
        ErrorExit(loc, "gaussiancloud: no Gaussians loaded from %s.", filename);

    auto *cloud = alloc.new_object<GaussianCloud>(
        renderFromObject, objectFromRender, reverseOrientation, std::move(gaussians),
        sigmaCutoff, shDegree, useCenterDepth, accel, samplingMode, multiSamples,
        backgroundColor, shViewDir);
    ++nGaussianClouds;
    nGaussianPrimitives += cloud->NumGaussians();
    return cloud;
}

GaussianCloud::GaussianCloud(const Transform *renderFromObject,
                             const Transform *objectFromRender, bool reverseOrientation,
                             std::vector<Gaussian3D> gaussians, Float sigmaCutoff,
                             int shDegree, bool useCenterDepth, InternalAccel internalAccel,
                             SamplingMode samplingMode, int multiSamples,
                             RGB backgroundColor, GaussianSHViewDir shViewDir)
    : renderFromObject(renderFromObject),
      objectFromRender(objectFromRender),
      reverseOrientation(reverseOrientation),
      sigmaCutoff(sigmaCutoff),
      shDegree(shDegree),
      useCenterDepth(useCenterDepth),
      internalAccel(internalAccel),
      samplingMode(samplingMode),
      multiSamples(std::max(1, multiSamples)),
      backgroundColor(backgroundColor),
      shViewDir(shViewDir),
      gaussians(std::move(gaussians)) {
    bounds = Bounds3f();
    for (const auto &g : this->gaussians)
        bounds = Union(bounds, g.aabb);

    orderedIndices.resize(this->gaussians.size());
    for (size_t i = 0; i < this->gaussians.size(); ++i)
        orderedIndices[i] = int(i);

    if (internalAccel == InternalAccel::KDTREE)
        BuildKdTree();
    else
        BuildBVH();
}

SampledSpectrum EvaluateGaussianSH(const GaussianCloud *cloud, int gaussianIndex,
                                   const Vector3f &viewDir, SampledWavelengths &wl) {
    if (!cloud)
        return SampledSpectrum(0.f);
    return cloud->EvaluateGaussianColor(gaussianIndex, viewDir, wl);
}

RGB EvaluateGaussianRGB(const GaussianCloud *cloud, int gaussianIndex,
                        const Vector3f &viewDir) {
    if (!cloud)
        return RGB(0.f, 0.f, 0.f);
    return EvaluateSHRGB(viewDir, cloud->GetGaussian(gaussianIndex).sh, cloud->SHDegree());
}

static void FinishGaussianInteraction(SurfaceInteraction &si, Normal3f n,
                                      Vector3f viewDirObject) {
    Vector3f dpdu, dpdv;
    CoordinateSystem(Vector3f(n), &dpdu, &dpdv);
    si.dpdu = dpdu;
    si.dpdv = dpdv;
    si.shading.dpdv = dpdv;
    si.shading.dpdu = viewDirObject;
}

static bool HasVisibleBackground(RGB backgroundColor) {
    return backgroundColor.r > 0.f || backgroundColor.g > 0.f || backgroundColor.b > 0.f;
}

static ShapeIntersection CreateDisplayRGBHit(const Ray &objectRay, Float tMax, RGB rgb,
                                             int faceIndex, Material material) {
    SurfaceInteraction si;
    si.pi = Point3fi(objectRay.o + tMax * objectRay.d);
    si.n = Normal3f(-objectRay.d);
    si.shading.n = si.n;
    si.wo = -objectRay.d;
    si.uv = Point2f(0, 0);
    si.time = objectRay.time;
    si.faceIndex = faceIndex;
    si.dpdu = Vector3f(1, 0, 0);
    si.dpdv = Vector3f(0, 1, 0);
    si.shading.dpdu = Vector3f(rgb.r, rgb.g, rgb.b);
    si.shading.dpdv = si.dpdv;
    si.SetIntersectionProperties(material, nullptr, nullptr, objectRay.medium);
    SetGaussianDirectRGBSample(rgb);
    return ShapeIntersection{si, 1e-4f};
}

static pstd::optional<ShapeIntersection> StochasticMissResult(const Ray &objectRay, Float tMax,
                                                              RGB backgroundColor,
                                                              Material material) {
    if (!HasVisibleBackground(backgroundColor))
        return {};
    return CreateDisplayRGBHit(objectRay, tMax, backgroundColor, kGaussianBackgroundFaceIndex,
                               material);
}

void GaussianCloud::BindMaterial(Material material) { boundMaterial = material; }

SampledSpectrum GaussianCloud::EvaluateGaussianColor(int index, const Vector3f &viewDir,
                                                     SampledWavelengths &wl) const {
    return EvaluateSHColor(viewDir, gaussians[index].sh, shDegree, wl);
}

Float GaussianCloud::EvalIntersectionT(const Gaussian3D &g, const Ray &objectRay) const {
    if (useCenterDepth) {
        Vector3f oc = Vector3f(g.mu - objectRay.o);
        return Dot(oc, objectRay.d);
    }

    Vector3f diff = Vector3f(g.mu - objectRay.o);
    Vector3f sinvd = g.sigmaInv * objectRay.d;
    Float denom = Dot(objectRay.d, sinvd);
    if (std::abs(denom) < 1e-8f)
        return Infinity;
    return Dot(diff, sinvd) / denom;
}

void GaussianCloud::BuildBVH() {
    struct PrimInfo {
        Bounds3f bounds;
        Point3f centroid;
        int index;
    };

    std::vector<PrimInfo> rootPrims(gaussians.size());
    for (size_t i = 0; i < gaussians.size(); ++i) {
        rootPrims[i].bounds = gaussians[i].aabb;
        rootPrims[i].centroid = (rootPrims[i].bounds.pMin + rootPrims[i].bounds.pMax) * 0.5f;
        rootPrims[i].index = int(i);
    }

    bvhNodes.clear();
    orderedIndices.clear();
    bvhNodes.reserve(gaussians.size() * 2);
    orderedIndices.reserve(gaussians.size() * 2);

    auto build = [&](auto &&self, const std::vector<PrimInfo> &items) -> int {
        int nodeIndex = int(bvhNodes.size());
        bvhNodes.push_back({});

        Bounds3f nodeBounds;
        for (const PrimInfo &p : items)
            nodeBounds = Union(nodeBounds, p.bounds);

        bvhNodes[nodeIndex].bounds = nodeBounds;

        if (items.size() <= 4) {
            bvhNodes[nodeIndex].start = int(orderedIndices.size());
            bvhNodes[nodeIndex].n = int(items.size());
            for (const PrimInfo &p : items)
                orderedIndices.push_back(p.index);
            return nodeIndex;
        }

        Bounds3f centroidBounds;
        for (const PrimInfo &p : items)
            centroidBounds = Union(centroidBounds, p.centroid);

        int axis = centroidBounds.MaxDimension();
        Float splitPos = (nodeBounds.pMin[axis] + nodeBounds.pMax[axis]) * 0.5f;

        std::vector<PrimInfo> left, right, overlap;
        left.reserve(items.size());
        right.reserve(items.size());
        overlap.reserve(items.size() / 8 + 1);

        for (const PrimInfo &p : items) {
            if (p.bounds.pMax[axis] <= splitPos)
                left.push_back(p);
            else if (p.bounds.pMin[axis] >= splitPos)
                right.push_back(p);
            else
                overlap.push_back(p);
        }

        if (left.empty() && right.empty()) {
            int mid = std::max(1, int(items.size()) / 2);
            std::vector<PrimInfo> sorted = items;
            std::nth_element(sorted.begin(), sorted.begin() + mid, sorted.end(),
                             [axis](const PrimInfo &a, const PrimInfo &b) {
                                 return a.centroid[axis] < b.centroid[axis];
                             });
            left.assign(sorted.begin(), sorted.begin() + mid);
            right.assign(sorted.begin() + mid, sorted.end());
            overlap.clear();
        }

        std::vector<PrimInfo> leftChild = left;
        std::vector<PrimInfo> rightChild = right;
        if (!overlap.empty()) {
            leftChild.insert(leftChild.end(), overlap.begin(), overlap.end());
            rightChild.insert(rightChild.end(), overlap.begin(), overlap.end());
        }
        if (leftChild.size() >= items.size() || rightChild.size() >= items.size()) {
            int mid = std::max(1, int(items.size()) / 2);
            leftChild.assign(items.begin(), items.begin() + mid);
            rightChild.assign(items.begin() + mid, items.end());
        }

        bvhNodes[nodeIndex].n = 0;
        bvhNodes[nodeIndex].axis = axis;
        bvhNodes[nodeIndex].left = self(self, leftChild);
        bvhNodes[nodeIndex].right = self(self, rightChild);
        return nodeIndex;
    };

    if (!rootPrims.empty())
        build(build, rootPrims);
}

void GaussianCloud::BuildKdTree() {
    struct PrimInfo {
        Bounds3f bounds;
        int index;
    };

    std::vector<PrimInfo> prims(gaussians.size());
    for (size_t i = 0; i < gaussians.size(); ++i) {
        prims[i].bounds = gaussians[i].aabb;
        prims[i].index = int(i);
    }

    kdNodes.clear();
    kdNodes.reserve(gaussians.size() * 2);

    auto build = [&](auto &&self, int start, int end, const Bounds3f &nodeBounds,
                     int depth) -> int {
        int nodeIndex = int(kdNodes.size());
        kdNodes.push_back({});

        kdNodes[nodeIndex].bounds = nodeBounds;
        kdNodes[nodeIndex].start = start;
        kdNodes[nodeIndex].n = end - start;

        if (end - start <= 8 || depth > 12) {
            kdNodes[nodeIndex].isLeaf = true;
            for (int i = start; i < end; ++i)
                orderedIndices[i] = prims[i].index;
            return nodeIndex;
        }

        int bestAxis = nodeBounds.MaxDimension();
        int mid = (start + end) / 2;
        std::nth_element(
            prims.begin() + start, prims.begin() + mid, prims.begin() + end,
            [bestAxis](const PrimInfo &a, const PrimInfo &b) {
                Float ca = (a.bounds.pMin[bestAxis] + a.bounds.pMax[bestAxis]) * 0.5f;
                Float cb = (b.bounds.pMin[bestAxis] + b.bounds.pMax[bestAxis]) * 0.5f;
                return ca < cb;
            });

        Float splitPos =
            ((prims[mid - 1].bounds.pMin[bestAxis] + prims[mid - 1].bounds.pMax[bestAxis]) *
                 0.5f +
             (prims[mid].bounds.pMin[bestAxis] + prims[mid].bounds.pMax[bestAxis]) * 0.5f) *
            0.5f;

        kdNodes[nodeIndex].isLeaf = false;
        kdNodes[nodeIndex].axis = bestAxis;
        kdNodes[nodeIndex].splitPos = splitPos;

        Bounds3f bounds0 = nodeBounds, bounds1 = nodeBounds;
        bounds0.pMax[bestAxis] = splitPos;
        bounds1.pMin[bestAxis] = splitPos;

        kdNodes[nodeIndex].left = self(self, start, mid, bounds0, depth + 1);
        kdNodes[nodeIndex].right = self(self, mid, end, bounds1, depth + 1);
        kdNodes[nodeIndex].n = 0;
        return nodeIndex;
    };

    if (!prims.empty())
        build(build, 0, int(prims.size()), bounds, 0);
}

Bounds3f GaussianCloud::Bounds() const {
    if (samplingMode == SamplingMode::COMPOSITE && HasVisibleBackground(backgroundColor)) {
        Bounds3f expanded = Union(
            bounds, Bounds3f(Point3f(-1e4f, -1e4f, -1e4f), Point3f(1e4f, 1e4f, 1e4f)));
        return (*renderFromObject)(expanded);
    }
    return (*renderFromObject)(bounds);
}

pstd::optional<ShapeIntersection> GaussianCloud::Intersect(const Ray &ray,
                                                           Float tMax) const {
    Ray objectRay = (*objectFromRender)(ray);
    auto isect = IntersectStochastic(objectRay, tMax);
    if (!isect)
        return {};

    SurfaceInteraction &si = isect->intr;
    Vector3f viewDir = si.shading.dpdu;
    si.pi = (*renderFromObject)(si.pi);
    si.n = (*renderFromObject)(si.n);
    si.shading.n = (*renderFromObject)(si.shading.n);
    Vector3f dpdu, dpdv;
    CoordinateSystem(Vector3f(si.shading.n), &dpdu, &dpdv);
    si.dpdu = dpdu;
    si.dpdv = dpdv;
    si.shading.dpdv = dpdv;
    if (si.faceIndex == kGaussianCompositeFaceIndex ||
        si.faceIndex == kGaussianMultiSampleFaceIndex ||
        si.faceIndex == kGaussianBackgroundFaceIndex)
        si.shading.dpdu = viewDir;
    else
        si.shading.dpdu = Normalize((*renderFromObject)(viewDir));
    si.wo = -ray.d;
    return isect;
}

bool GaussianCloud::IntersectP(const Ray &ray, Float tMax) const {
    return Intersect(ray, tMax).has_value();
}

pstd::optional<ShapeIntersection> GaussianCloud::IntersectStochastic(const Ray &objectRay,
                                                                   Float tMax) const {
    if (samplingMode == SamplingMode::COMPOSITE)
        return IntersectComposite(objectRay, tMax);
    if (internalAccel == InternalAccel::KDTREE)
        return IntersectKdTree(objectRay, tMax);
    return IntersectBVH(objectRay, tMax);
}

pstd::optional<ShapeIntersection> GaussianCloud::IntersectComposite(const Ray &objectRay,
                                                                    Float tMax) const {
    struct Candidate {
        int index;
        Float t;
        Float alpha;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(256);

    auto testGaussian = [&](int gIndex) {
        const Gaussian3D &g = gaussians[gIndex];
        // Per-Gaussian AABB slab test (OursCenter can pass mhd² at projected t even when
        // the ray misses the conservative ellipsoid bounds along the ray).
        if (!g.aabb.IntersectP(objectRay.o, objectRay.d, tMax))
            return;

        Float t = EvalIntersectionT(g, objectRay);
        if (t <= 0.f || t >= tMax)
            return;

        Point3f p = objectRay(t);
        Vector3f d2mu = p - g.mu;
        Float mhd2 = Dot(d2mu, g.sigmaInv * d2mu);
        if (mhd2 > sigmaCutoff * sigmaCutoff)
            return;

        Float alpha = std::min(g.opacity * std::exp(-0.5f * mhd2), kGaussianAlphaCap);
        if (alpha < kGaussianAlphaMinThreshold)
            return;

        candidates.push_back({gIndex, t, alpha});
    };

    if (internalAccel == InternalAccel::BRUTE) {
        for (int gIndex = 0; gIndex < (int)gaussians.size(); ++gIndex)
            testGaussian(gIndex);
    } else if (internalAccel == InternalAccel::KDTREE) {
        if (kdNodes.empty())
            return {};

        struct StackEntry {
            int node;
            Float tMin;
            Float tMax;
        };
        std::vector<StackEntry> stack;
        stack.push_back({0, 0.f, tMax});

        while (!stack.empty()) {
            int nodeIndex = stack.back().node;
            Float tMin = stack.back().tMin;
            Float nodeTMax = stack.back().tMax;
            stack.pop_back();

            const KdTreeNode &node = kdNodes[nodeIndex];
            if (!node.bounds.IntersectP(objectRay.o, objectRay.d, nodeTMax, &tMin))
                continue;

            if (node.isLeaf) {
                for (int i = 0; i < node.n; ++i)
                    testGaussian(orderedIndices[node.start + i]);
                continue;
            }

            int axis = node.axis;
            Float tSplit = node.splitPos;
            Float tPlane = (tSplit - objectRay.o[axis]) / objectRay.d[axis];
            int nearChild = objectRay.o[axis] < tSplit ||
                                    (objectRay.o[axis] == tSplit && objectRay.d[axis] <= 0)
                                ? node.left
                                : node.right;
            int farChild = nearChild == node.left ? node.right : node.left;

            if (tPlane > tMin && tPlane < nodeTMax)
                stack.push_back({farChild, tPlane, nodeTMax});
            stack.push_back({nearChild, tMin, std::min(nodeTMax, tPlane)});
        }
    } else {
        if (bvhNodes.empty())
            return {};

        struct StackEntry {
            int node;
            Float tMax;
        };
        std::vector<StackEntry> stack;
        stack.push_back({0, tMax});

        while (!stack.empty()) {
            int nodeIndex = stack.back().node;
            Float nodeTMax = stack.back().tMax;
            stack.pop_back();

            const BVHNode &node = bvhNodes[nodeIndex];
            if (!node.bounds.IntersectP(objectRay.o, objectRay.d, nodeTMax))
                continue;

            if (node.n > 0) {
                for (int i = 0; i < node.n; ++i)
                    testGaussian(orderedIndices[node.start + i]);
            } else {
                Float tSplit = (node.bounds.pMin[node.axis] + node.bounds.pMax[node.axis]) * 0.5f;
                bool belowFirst = objectRay.o[node.axis] < tSplit ||
                                  (objectRay.o[node.axis] == tSplit && objectRay.d[node.axis] <= 0);
                if (belowFirst) {
                    stack.push_back({node.right, nodeTMax});
                    stack.push_back({node.left, nodeTMax});
                } else {
                    stack.push_back({node.left, nodeTMax});
                    stack.push_back({node.right, nodeTMax});
                }
            }
        }
    }

    if (candidates.empty())
        return CreateDisplayRGBHit(objectRay, tMax, backgroundColor,
                                   kGaussianCompositeFaceIndex, boundMaterial);

    // Straddling primitives live in both BVH children; drop duplicate indices.
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) {
                  return a.index < b.index || (a.index == b.index && a.t < b.t);
              });
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                   [](const Candidate &a, const Candidate &b) {
                                       return a.index == b.index;
                                   }),
                     candidates.end());

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) { return a.t < b.t; });

    RGB rgb(0.f, 0.f, 0.f);
    Float T = 1.f;
    for (const Candidate &c : candidates) {
        const Gaussian3D &g = gaussians[c.index];
        RGB ci = EvaluateGaussianRGB(this, c.index,
                                     GaussianSHViewDirection(shViewDir, g.mu, objectRay));
        rgb += T * c.alpha * ci;
        T *= (1.f - c.alpha);
        if (T < 1e-4f)
            break;
    }
    rgb += T * backgroundColor;

    return CreateDisplayRGBHit(objectRay, tMax, rgb, kGaussianCompositeFaceIndex, boundMaterial);
}

pstd::optional<ShapeIntersection> GaussianCloud::IntersectBVH(const Ray &objectRay,
                                                              Float tMax) const {
    if (bvhNodes.empty())
        return {};

    const int N = std::max(1, multiSamples);
    struct Slot {
        Float currentTMax;
        pstd::optional<ShapeIntersection> best;
    };
    std::vector<Slot> slots(N, {tMax, {}});
    int frameNum = GetGaussianFrameNumber();

    auto minSlotTMax = [&]() {
        Float m = tMax;
        for (const Slot &s : slots)
            m = std::min(m, s.currentTMax);
        return m;
    };

    auto testGaussian = [&](int gIndex, int sampleIndex) {
        Slot &slot = slots[sampleIndex];
        const Gaussian3D &g = gaussians[gIndex];
        if (!g.aabb.IntersectP(objectRay.o, objectRay.d, slot.currentTMax))
            return;

        Float t = EvalIntersectionT(g, objectRay);
        if (t <= 0.f || t >= slot.currentTMax)
            return;

        Point3f p = objectRay(t);
        Vector3f d2mu = p - g.mu;
        Float mhd2 = Dot(d2mu, g.sigmaInv * d2mu);
        if (mhd2 > sigmaCutoff * sigmaCutoff)
            return;

        Float alpha = std::min(g.opacity * std::exp(-0.5f * mhd2), kGaussianAlphaCap);
        if (alpha < kGaussianAlphaMinThreshold)
            return;

        if (TrigHash(p, frameNum + sampleIndex) >= alpha)
            return;

        slot.currentTMax = t;
        Normal3f n(Normalize(d2mu));
        if (reverseOrientation)
            n = -n;

        SurfaceInteraction si;
        si.pi = Point3fi(p);
        si.n = n;
        si.shading.n = n;
        si.wo = -objectRay.d;
        si.uv = Point2f(0, 0);
        si.time = objectRay.time;
        si.faceIndex = gIndex;
        FinishGaussianInteraction(si, n, GaussianSHViewDirection(shViewDir, g.mu, objectRay));
        si.SetIntersectionProperties(boundMaterial, nullptr, nullptr, objectRay.medium);
        slot.best = ShapeIntersection{si, t};
    };

    struct StackEntry {
        int node;
        Float tMax;
    };
    std::vector<StackEntry> stack;
    stack.push_back({0, tMax});

    while (!stack.empty()) {
        int nodeIndex = stack.back().node;
        Float nodeTMax = std::min(stack.back().tMax, minSlotTMax());
        stack.pop_back();

        const BVHNode &node = bvhNodes[nodeIndex];
        if (!node.bounds.IntersectP(objectRay.o, objectRay.d, nodeTMax))
            continue;

        if (node.n > 0) {
            for (int i = 0; i < node.n; ++i) {
                int gIndex = orderedIndices[node.start + i];
                for (int s = 0; s < N; ++s)
                    testGaussian(gIndex, s);
            }
        } else {
            Float tSplit = (node.bounds.pMin[node.axis] + node.bounds.pMax[node.axis]) * 0.5f;
            bool belowFirst = objectRay.o[node.axis] < tSplit ||
                              (objectRay.o[node.axis] == tSplit && objectRay.d[node.axis] <= 0);
            if (belowFirst) {
                stack.push_back({node.right, nodeTMax});
                stack.push_back({node.left, nodeTMax});
            } else {
                stack.push_back({node.left, nodeTMax});
                stack.push_back({node.right, nodeTMax});
            }
        }
    }
    if (N == 1) {
        if (slots[0].best)
            return slots[0].best;
        return StochasticMissResult(objectRay, tMax, backgroundColor, boundMaterial);
    }

    RGB sum(0.f, 0.f, 0.f);
    int count = 0;
    for (const Slot &slot : slots) {
        if (!slot.best)
            continue;
        const SurfaceInteraction &intr = slot.best->intr;
        RGB ci = EvaluateGaussianRGB(this, intr.faceIndex, Normalize(intr.shading.dpdu));
        sum += ci;
        ++count;
    }
    if (count == 0)
        return StochasticMissResult(objectRay, tMax, backgroundColor, boundMaterial);

    RGB avg = sum / Float(N);
    return CreateDisplayRGBHit(objectRay, tMax, avg, kGaussianMultiSampleFaceIndex,
                               boundMaterial);
}

pstd::optional<ShapeIntersection> GaussianCloud::IntersectKdTree(const Ray &objectRay,
                                                                 Float tMax) const {
    if (kdNodes.empty())
        return {};

    const int N = std::max(1, multiSamples);
    struct Slot {
        Float currentTMax;
        pstd::optional<ShapeIntersection> best;
    };
    std::vector<Slot> slots(N, {tMax, {}});
    int frameNum = GetGaussianFrameNumber();

    auto minSlotTMax = [&]() {
        Float m = tMax;
        for (const Slot &s : slots)
            m = std::min(m, s.currentTMax);
        return m;
    };

    auto testGaussian = [&](int gIndex, int sampleIndex) {
        Slot &slot = slots[sampleIndex];
        const Gaussian3D &g = gaussians[gIndex];
        if (!g.aabb.IntersectP(objectRay.o, objectRay.d, slot.currentTMax))
            return;

        Float t = EvalIntersectionT(g, objectRay);
        if (t <= 0.f || t >= slot.currentTMax)
            return;

        Point3f p = objectRay(t);
        Vector3f d2mu = p - g.mu;
        Float mhd2 = Dot(d2mu, g.sigmaInv * d2mu);
        if (mhd2 > sigmaCutoff * sigmaCutoff)
            return;

        Float alpha = std::min(g.opacity * std::exp(-0.5f * mhd2), kGaussianAlphaCap);
        if (alpha < kGaussianAlphaMinThreshold)
            return;

        if (TrigHash(p, frameNum + sampleIndex) >= alpha)
            return;

        slot.currentTMax = t;
        Normal3f n(Normalize(d2mu));
        if (reverseOrientation)
            n = -n;

        SurfaceInteraction si;
        si.pi = Point3fi(p);
        si.n = n;
        si.shading.n = n;
        si.wo = -objectRay.d;
        si.uv = Point2f(0, 0);
        si.time = objectRay.time;
        si.faceIndex = gIndex;
        FinishGaussianInteraction(si, n, GaussianSHViewDirection(shViewDir, g.mu, objectRay));
        si.SetIntersectionProperties(boundMaterial, nullptr, nullptr, objectRay.medium);
        slot.best = ShapeIntersection{si, t};
    };

    struct StackEntry {
        int node;
        Float tMin;
        Float tMax;
    };
    std::vector<StackEntry> stack;
    stack.push_back({0, 0.f, tMax});

    while (!stack.empty()) {
        int nodeIndex = stack.back().node;
        Float tMin = stack.back().tMin;
        Float nodeTMax = std::min(stack.back().tMax, minSlotTMax());
        stack.pop_back();

        const KdTreeNode &node = kdNodes[nodeIndex];
        if (!node.bounds.IntersectP(objectRay.o, objectRay.d, nodeTMax, &tMin))
            continue;

        if (node.isLeaf) {
            for (int i = 0; i < node.n; ++i) {
                int gIndex = orderedIndices[node.start + i];
                for (int s = 0; s < N; ++s)
                    testGaussian(gIndex, s);
            }
            continue;
        }

        int axis = node.axis;
        Float tSplit = node.splitPos;
        Float tPlane = (tSplit - objectRay.o[axis]) / objectRay.d[axis];

        int nearChild = objectRay.o[axis] < tSplit ||
                                (objectRay.o[axis] == tSplit && objectRay.d[axis] <= 0)
                            ? node.left
                            : node.right;
        int farChild = nearChild == node.left ? node.right : node.left;

        if (tPlane > tMin && tPlane < nodeTMax)
            stack.push_back({farChild, tPlane, nodeTMax});
        stack.push_back({nearChild, tMin, std::min(nodeTMax, tPlane)});
    }
    if (N == 1) {
        if (slots[0].best)
            return slots[0].best;
        return StochasticMissResult(objectRay, tMax, backgroundColor, boundMaterial);
    }

    RGB sum(0.f, 0.f, 0.f);
    int count = 0;
    for (const Slot &slot : slots) {
        if (!slot.best)
            continue;
        const SurfaceInteraction &intr = slot.best->intr;
        RGB ci = EvaluateGaussianRGB(this, intr.faceIndex, Normalize(intr.shading.dpdu));
        sum += ci;
        ++count;
    }
    if (count == 0)
        return StochasticMissResult(objectRay, tMax, backgroundColor, boundMaterial);

    RGB avg = sum / Float(N);
    return CreateDisplayRGBHit(objectRay, tMax, avg, kGaussianMultiSampleFaceIndex,
                               boundMaterial);
}

std::string GaussianCloud::ToString() const {
    return StringPrintf("[ GaussianCloud count: %d sigmaCutoff: %f shDegree: %d ]",
                        NumGaussians(), sigmaCutoff, shDegree);
}

}  // namespace pbrt
