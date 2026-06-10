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

// Camera parameters for 2D Gaussian projection (rasterizer-compatible alpha).
// rx/ry/rz are camera right / down (3DGS Y) / forward axes in world space
// (i.e. columns 0, 1, 2 of the 3DGS C2W rotation matrix R).
struct GaussianCameraParams {
    Float fx = -1.f, fy = -1.f, cx = 0.f, cy = 0.f;
    Vector3f rx, ry, rz;
    Point3f pos;          // camera origin in world space
    int width = 800, height = 800;
    bool valid = false;
};

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
                  int multiSamples, RGB backgroundColor, GaussianSHViewDir shViewDir,
                  GaussianCameraParams cameraParams, bool use2DAlpha);

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
    // 2D tile grid for fast candidate lookup when cameraParams.valid.
    // Tile cell (cx,cy) stores the indices of all Gaussians whose 2D projected
    // bounding box overlaps the cell.
    void Build2DGrid();
    pstd::optional<ShapeIntersection> IntersectStochastic(const Ray &objectRay,
                                                          Float tMax) const;
    pstd::optional<ShapeIntersection> IntersectComposite(const Ray &objectRay,
                                                         Float tMax) const;
    pstd::optional<ShapeIntersection> IntersectBVH(const Ray &objectRay, Float tMax) const;
    pstd::optional<ShapeIntersection> IntersectKdTree(const Ray &objectRay, Float tMax) const;

    PBRT_CPU_GPU Float EvalIntersectionT(const Gaussian3D &g, const Ray &objectRay) const;
    // OursMean depth: argmin_t mhd² along the ray (always used for alpha / cutoff).
    PBRT_CPU_GPU static Float EvalMeanDepthT(const Gaussian3D &g, const Ray &objectRay);
    PBRT_CPU_GPU Float EvalAlphaDepthT(const Gaussian3D &g, const Ray &objectRay) const;

    struct GaussianAlphaEval {
        Float tAlpha;
        Float mhd2;
        Float alpha;
    };
    PBRT_CPU_GPU GaussianAlphaEval EvalGaussianAlpha(const Gaussian3D &g,
                                                     const Ray &objectRay) const;
    // 2D-projection alpha (matches rasterizer) when cameraParams.valid == true.
    PBRT_CPU_GPU GaussianAlphaEval EvalGaussianAlpha2D(const Gaussian3D &g,
                                                       const Ray &objectRay) const;
    // Rasterizer-aligned 2D alpha when enabled and camera params are present.
    PBRT_CPU_GPU GaussianAlphaEval EvalAlphaForRay(const Gaussian3D &g,
                                                   const Ray &objectRay) const;

    const Transform *renderFromObject, *objectFromRender;
    bool reverseOrientation;
    Float sigmaCutoff;
    int shDegree;
    bool useCenterDepth;
    bool use2DAlpha = true;
    InternalAccel internalAccel;
    SamplingMode samplingMode = SamplingMode::STOCHASTIC;
    int multiSamples = 1;
    RGB backgroundColor;
    GaussianSHViewDir shViewDir = GaussianSHViewDir::CAM_TO_GAUSSIAN;
    GaussianCameraParams cameraParams;

    std::vector<Gaussian3D> gaussians;
    Bounds3f bounds;

    std::vector<int> orderedIndices;
    std::vector<BVHNode> bvhNodes;
    std::vector<KdTreeNode> kdNodes;

    // 2D tile grid (built when cameraParams.valid).
    std::vector<std::vector<int>> grid2D;
    int grid2DW = 0, grid2DH = 0;
    static constexpr int kGrid2DCellSize = 16;

    Material boundMaterial;
};

}  // namespace pbrt

#endif  // PBRT_GAUSSIAN_H
