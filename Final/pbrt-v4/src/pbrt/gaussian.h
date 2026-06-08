// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GAUSSIAN_H
#define PBRT_GAUSSIAN_H

#include <pbrt/pbrt.h>

#include <pbrt/util/color.h>
#include <pbrt/util/plyloader_3dgs.h>
#include <pbrt/gaussian_eval.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <vector>

namespace pbrt {

class Material;

class GaussianCloud {
  public:
    enum class InternalAccel { BVH, KDTREE, BRUTE };
    enum class SamplingMode { STOCHASTIC, COMPOSITE };

    static GaussianCloud *Create(const Transform *renderFromObject,
                                 const Transform *objectFromRender, bool reverseOrientation,
                                 const ParameterDictionary &parameters, const FileLoc *loc,
                                 Allocator alloc);

    std::string ToString() const;

    GaussianCloud(const Transform *renderFromObject, const Transform *objectFromRender,
                  bool reverseOrientation, std::vector<Gaussian3D> gaussians,
                  Float sigmaCutoff, int shDegree, bool useCenterDepth,
                  InternalAccel internalAccel, SamplingMode samplingMode,
                  int multiSamples, RGB backgroundColor, GaussianSHViewDir shViewDir);

    PBRT_CPU_GPU Bounds3f Bounds() const;
    PBRT_CPU_GPU DirectionCone NormalBounds() const { return DirectionCone::EntireSphere(); }
    PBRT_CPU_GPU pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                             Float tMax) const;
    PBRT_CPU_GPU bool IntersectP(const Ray &ray, Float tMax) const;
    PBRT_CPU_GPU Float Area() const { return 0; }
    PBRT_CPU_GPU pstd::optional<ShapeSample> Sample(Point2f u) const { return {}; }
    PBRT_CPU_GPU Float PDF(const Interaction &) const { return 0; }
    PBRT_CPU_GPU pstd::optional<ShapeSample> Sample(const ShapeSampleContext &, Point2f) const {
        return {};
    }
    PBRT_CPU_GPU Float PDF(const ShapeSampleContext &, Vector3f) const { return 0; }

    PBRT_CPU_GPU const Gaussian3D &GetGaussian(int index) const { return gaussians[index]; }
    PBRT_CPU_GPU int NumGaussians() const { return int(gaussians.size()); }
    PBRT_CPU_GPU int SHDegree() const { return shDegree; }

    void BindMaterial(Material material);

    PBRT_CPU_GPU SampledSpectrum EvaluateGaussianColor(int index, const Vector3f &viewDir,
                                                       SampledWavelengths &wl) const;

  private:
    struct BVHNode {
        Bounds3f bounds;
        int start = 0;
        int n = 0;
        int left = -1;
        int right = -1;
        int axis = 0;
    };

    struct KdTreeNode {
        Bounds3f bounds;
        int start = 0;
        int n = 0;
        int left = -1;
        int right = -1;
        int axis = 0;
        Float splitPos = 0;
        bool isLeaf = true;
    };

    void BuildBVH();
    void BuildKdTree();
    pstd::optional<ShapeIntersection> IntersectStochastic(const Ray &objectRay,
                                                          Float tMax) const;
    pstd::optional<ShapeIntersection> IntersectComposite(const Ray &objectRay,
                                                         Float tMax) const;
    pstd::optional<ShapeIntersection> IntersectBVH(const Ray &objectRay, Float tMax) const;
    pstd::optional<ShapeIntersection> IntersectKdTree(const Ray &objectRay, Float tMax) const;

    PBRT_CPU_GPU Float EvalIntersectionT(const Gaussian3D &g, const Ray &objectRay) const;

    const Transform *renderFromObject, *objectFromRender;
    bool reverseOrientation;
    Float sigmaCutoff;
    int shDegree;
    bool useCenterDepth;
    InternalAccel internalAccel;
    SamplingMode samplingMode = SamplingMode::STOCHASTIC;
    int multiSamples = 1;
    RGB backgroundColor;
    GaussianSHViewDir shViewDir = GaussianSHViewDir::CAM_TO_GAUSSIAN;

    std::vector<Gaussian3D> gaussians;
    Bounds3f bounds;

    std::vector<int> orderedIndices;
    std::vector<BVHNode> bvhNodes;
    std::vector<KdTreeNode> kdNodes;

    Material boundMaterial;
};

}  // namespace pbrt

#endif  // PBRT_GAUSSIAN_H
