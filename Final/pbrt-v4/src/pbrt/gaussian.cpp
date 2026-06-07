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
    bool useCenterDepth = parameters.GetOneBool("use_center_depth", true);
    std::string accelName = parameters.GetOneString("internal_accel", "bvh");
    InternalAccel accel = InternalAccel::BVH;
    if (accelName == "kdtree")
        accel = InternalAccel::KDTREE;

    std::vector<Gaussian3D> gaussians = Load3DGSPly(filename, sigmaCutoff);
    if (gaussians.empty())
        ErrorExit(loc, "gaussiancloud: no Gaussians loaded from %s.", filename);

    auto *cloud = alloc.new_object<GaussianCloud>(
        renderFromObject, objectFromRender, reverseOrientation, std::move(gaussians),
        sigmaCutoff, shDegree, useCenterDepth, accel);
    ++nGaussianClouds;
    nGaussianPrimitives += cloud->NumGaussians();
    return cloud;
}

GaussianCloud::GaussianCloud(const Transform *renderFromObject,
                             const Transform *objectFromRender, bool reverseOrientation,
                             std::vector<Gaussian3D> gaussians, Float sigmaCutoff,
                             int shDegree, bool useCenterDepth, InternalAccel internalAccel)
    : renderFromObject(renderFromObject),
      objectFromRender(objectFromRender),
      reverseOrientation(reverseOrientation),
      sigmaCutoff(sigmaCutoff),
      shDegree(shDegree),
      useCenterDepth(useCenterDepth),
      internalAccel(internalAccel),
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
        return SampledSpectrum(0.5f);
    return cloud->EvaluateGaussianColor(gaussianIndex, viewDir, wl);
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

    std::vector<PrimInfo> prims(gaussians.size());
    for (size_t i = 0; i < gaussians.size(); ++i) {
        prims[i].bounds = gaussians[i].aabb;
        prims[i].centroid = (prims[i].bounds.pMin + prims[i].bounds.pMax) * 0.5f;
        prims[i].index = int(i);
    }

    bvhNodes.clear();
    bvhNodes.reserve(gaussians.size() * 2);

    auto build = [&](auto &&self, int start, int end) -> int {
        int nodeIndex = int(bvhNodes.size());
        bvhNodes.push_back({});

        Bounds3f nodeBounds;
        for (int i = start; i < end; ++i)
            nodeBounds = Union(nodeBounds, prims[i].bounds);

        int count = end - start;
        bvhNodes[nodeIndex].bounds = nodeBounds;
        bvhNodes[nodeIndex].start = start;
        bvhNodes[nodeIndex].n = count;

        if (count <= 4) {
            for (int i = start; i < end; ++i)
                orderedIndices[i] = prims[i].index;
            return nodeIndex;
        }

        Bounds3f centroidBounds;
        for (int i = start; i < end; ++i)
            centroidBounds = Union(centroidBounds, prims[i].centroid);

        int axis = centroidBounds.MaxDimension();
        int mid = (start + end) / 2;
        std::nth_element(prims.begin() + start, prims.begin() + mid, prims.begin() + end,
                         [axis](const PrimInfo &a, const PrimInfo &b) {
                             return a.centroid[axis] < b.centroid[axis];
                         });

        bvhNodes[nodeIndex].n = 0;
        bvhNodes[nodeIndex].axis = axis;
        bvhNodes[nodeIndex].left = self(self, start, mid);
        bvhNodes[nodeIndex].right = self(self, mid, end);
        return nodeIndex;
    };

    if (!prims.empty())
        build(build, 0, int(prims.size()));
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

Bounds3f GaussianCloud::Bounds() const { return (*renderFromObject)(bounds); }

pstd::optional<ShapeIntersection> GaussianCloud::Intersect(const Ray &ray,
                                                           Float tMax) const {
    Ray objectRay = (*objectFromRender)(ray);
    auto isect = IntersectStochastic(objectRay, tMax);
    if (!isect)
        return {};

    SurfaceInteraction &si = isect->intr;
    si.pi = (*renderFromObject)(si.pi);
    si.n = (*renderFromObject)(si.n);
    si.shading.n = (*renderFromObject)(si.shading.n);
    si.wo = -ray.d;
    return isect;
}

bool GaussianCloud::IntersectP(const Ray &ray, Float tMax) const {
    return Intersect(ray, tMax).has_value();
}

pstd::optional<ShapeIntersection> GaussianCloud::IntersectStochastic(const Ray &objectRay,
                                                                   Float tMax) const {
    if (internalAccel == InternalAccel::KDTREE)
        return IntersectKdTree(objectRay, tMax);
    return IntersectBVH(objectRay, tMax);
}

pstd::optional<ShapeIntersection> GaussianCloud::IntersectBVH(const Ray &objectRay,
                                                              Float tMax) const {
    if (bvhNodes.empty())
        return {};

    Float currentTMax = tMax;
    pstd::optional<ShapeIntersection> best;
    int frameNum = GaussianFrameNumber;

    auto testGaussian = [&](int gIndex) {
        const Gaussian3D &g = gaussians[gIndex];
        Float t = EvalIntersectionT(g, objectRay);
        if (t <= 0 || t >= currentTMax)
            return;

        Point3f p = objectRay(t);
        Vector3f d2mu = p - g.mu;
        Float mhd2 = Dot(d2mu, g.sigmaInv * d2mu);
        if (mhd2 > sigmaCutoff * sigmaCutoff)
            return;

        Float alpha = g.opacity * std::exp(-0.5f * mhd2);
        if (alpha < 1e-4f)
            return;

        if (TrigHash(p, frameNum) >= alpha)
            return;

        currentTMax = t;
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
        si.SetIntersectionProperties(boundMaterial, nullptr, nullptr, objectRay.medium);
        best = ShapeIntersection{si, t};
    };

    struct StackEntry {
        int node;
        Float tMax;
    };
    std::vector<StackEntry> stack;
    stack.push_back({0, currentTMax});

    while (!stack.empty()) {
        int nodeIndex = stack.back().node;
        Float nodeTMax = std::min(stack.back().tMax, currentTMax);
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

    return best;
}

pstd::optional<ShapeIntersection> GaussianCloud::IntersectKdTree(const Ray &objectRay,
                                                                 Float tMax) const {
    if (kdNodes.empty())
        return {};

    Float currentTMax = tMax;
    pstd::optional<ShapeIntersection> best;
    int frameNum = GaussianFrameNumber;

    auto testGaussian = [&](int gIndex) {
        const Gaussian3D &g = gaussians[gIndex];
        Float t = EvalIntersectionT(g, objectRay);
        if (t <= 0 || t >= currentTMax)
            return;

        Point3f p = objectRay(t);
        Vector3f d2mu = p - g.mu;
        Float mhd2 = Dot(d2mu, g.sigmaInv * d2mu);
        if (mhd2 > sigmaCutoff * sigmaCutoff)
            return;

        Float alpha = g.opacity * std::exp(-0.5f * mhd2);
        if (alpha < 1e-4f)
            return;

        if (TrigHash(p, frameNum) >= alpha)
            return;

        currentTMax = t;
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
        si.SetIntersectionProperties(boundMaterial, nullptr, nullptr, objectRay.medium);
        best = ShapeIntersection{si, t};
    };

    struct StackEntry {
        int node;
        Float tMin;
        Float tMax;
    };
    std::vector<StackEntry> stack;
    stack.push_back({0, 0.f, currentTMax});

    while (!stack.empty()) {
        int nodeIndex = stack.back().node;
        Float tMin = stack.back().tMin;
        Float nodeTMax = std::min(stack.back().tMax, currentTMax);
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

    return best;
}

std::string GaussianCloud::ToString() const {
    return StringPrintf("[ GaussianCloud count: %d sigmaCutoff: %f shDegree: %d ]",
                        NumGaussians(), sigmaCutoff, shDegree);
}

}  // namespace pbrt
