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
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>

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

    // Optional 2D-projection camera parameters.  When present (cam_fx > 0) the
    // composite integrator uses the rasterizer-equivalent 2D Mahalanobis distance
    // instead of the 3D one, eliminating 3D/2D mismatch artefacts.
    GaussianCameraParams camParams;
    camParams.fx = parameters.GetOneFloat("cam_fx", -1.f);
    camParams.fy = parameters.GetOneFloat("cam_fy", -1.f);
    if (camParams.fx > 0.f && camParams.fy > 0.f) {
        camParams.valid = true;
        camParams.cx = parameters.GetOneFloat("cam_cx", 0.f);
        camParams.cy = parameters.GetOneFloat("cam_cy", 0.f);
        std::vector<Float> rx_arr = parameters.GetFloatArray("cam_rx");
        std::vector<Float> ry_arr = parameters.GetFloatArray("cam_ry");
        std::vector<Float> rz_arr = parameters.GetFloatArray("cam_rz");
        if (rx_arr.size() >= 3)
            camParams.rx = Vector3f(rx_arr[0], rx_arr[1], rx_arr[2]);
        if (ry_arr.size() >= 3)
            camParams.ry = Vector3f(ry_arr[0], ry_arr[1], ry_arr[2]);
        if (rz_arr.size() >= 3)
            camParams.rz = Vector3f(rz_arr[0], rz_arr[1], rz_arr[2]);
        std::vector<Float> pos_arr = parameters.GetFloatArray("cam_pos");
        if (pos_arr.size() >= 3)
            camParams.pos = Point3f(pos_arr[0], pos_arr[1], pos_arr[2]);
        camParams.width  = parameters.GetOneInt("cam_width",  800);
        camParams.height = parameters.GetOneInt("cam_height", 800);
    }

    std::vector<Gaussian3D> gaussians = Load3DGSPly(filename, sigmaCutoff);
    if (gaussians.empty())
        ErrorExit(loc, "gaussiancloud: no Gaussians loaded from %s.", filename);

    auto *cloud = alloc.new_object<GaussianCloud>(
        renderFromObject, objectFromRender, reverseOrientation, std::move(gaussians),
        sigmaCutoff, shDegree, useCenterDepth, accel, samplingMode, multiSamples,
        backgroundColor, shViewDir, camParams);
    ++nGaussianClouds;
    nGaussianPrimitives += cloud->NumGaussians();
    return cloud;
}

GaussianCloud::GaussianCloud(const Transform *renderFromObject,
                             const Transform *objectFromRender, bool reverseOrientation,
                             std::vector<Gaussian3D> gaussians, Float sigmaCutoff,
                             int shDegree, bool useCenterDepth, InternalAccel internalAccel,
                             SamplingMode samplingMode, int multiSamples,
                             RGB backgroundColor, GaussianSHViewDir shViewDir,
                             GaussianCameraParams cameraParams)
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
      cameraParams(cameraParams),
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

    if (cameraParams.valid)
        Build2DGrid();
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

struct CompositeTraceConfig {
    bool enabled = false;
    int maxRays = 1;
    int maxEntries = 256;
    std::string file = "gaussian_composite_trace.log";
};

static CompositeTraceConfig GetCompositeTraceConfig() {
    CompositeTraceConfig cfg;
    const char *enabled = std::getenv("PBRT_GAUSS_TRACE");
    if (!enabled || enabled[0] == '\0' || enabled[0] == '0')
        return cfg;

    cfg.enabled = true;

    if (const char *maxRays = std::getenv("PBRT_GAUSS_TRACE_RAYS")) {
        int v = std::atoi(maxRays);
        if (v > 0)
            cfg.maxRays = v;
    }
    if (const char *maxEntries = std::getenv("PBRT_GAUSS_TRACE_MAX")) {
        int v = std::atoi(maxEntries);
        if (v > 0)
            cfg.maxEntries = v;
    }
    if (const char *file = std::getenv("PBRT_GAUSS_TRACE_FILE")) {
        if (file[0] != '\0')
            cfg.file = file;
    }
    return cfg;
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
    return EvalMeanDepthT(g, objectRay);
}

Float GaussianCloud::EvalMeanDepthT(const Gaussian3D &g, const Ray &objectRay) {
    Vector3f diff = Vector3f(g.mu - objectRay.o);
    Vector3f sinvd = g.sigmaInv * objectRay.d;
    Float denom = Dot(objectRay.d, sinvd);
    if (std::abs(denom) < 1e-8f)
        return Infinity;
    return Dot(diff, sinvd) / denom;
}

Float GaussianCloud::EvalAlphaDepthT(const Gaussian3D &g, const Ray &objectRay) const {
    // The alpha is evaluated at the 1D Gaussian mean (maximum contribution point) along
    // the ray.  This is true for both OursMean and OursCenter modes.
    // In OursCenter mode, EvalIntersectionT (center projection) is used for DEPTH SORTING,
    // but the alpha magnitude is still computed at the 1D mean for maximum accuracy.
    // When tMean <= 0 (behind camera), fall back to the sorting depth.
    Float tMean = EvalMeanDepthT(g, objectRay);
    if (tMean <= 0.f)
        return EvalIntersectionT(g, objectRay);
    return tMean;
}

GaussianCloud::GaussianAlphaEval GaussianCloud::EvalGaussianAlpha(
    const Gaussian3D &g, const Ray &objectRay) const {
    Float tAlpha = EvalAlphaDepthT(g, objectRay);
    Point3f p = objectRay(tAlpha);
    Vector3f d2mu = p - g.mu;
    Float mhd2 = Dot(d2mu, g.sigmaInv * d2mu);
    // Guard: mhd2 must be >= 0 for a valid positive-definite sigmaInv.
    // Very small or degenerate Gaussian scales can cause float32 precision loss
    // in sigmaInv, yielding negative mhd2.  Treat those as no contribution.
    if (mhd2 < 0.f) mhd2 = sigmaCutoff * sigmaCutoff + 1.f;
    Float alpha = std::min(g.opacity * std::exp(-0.5f * mhd2), kGaussianAlphaCap);
    return {tAlpha, mhd2, alpha};
}

GaussianCloud::GaussianAlphaEval GaussianCloud::EvalGaussianAlpha2D(
    const Gaussian3D &g, const Ray &objectRay) const {
    // Project the Gaussian center into the camera's 2D image plane using the
    // stored 3DGS camera parameters and compute the rasterizer-equivalent 2D
    // Mahalanobis distance.  Falls back to the 3D version if the Gaussian is
    // behind the camera or the camera parameters are not available.
    //
    // Sign convention:
    //   cameraParams.rx = camera-right   in world space  (R[:,0] of 3DGS C2W)
    //   cameraParams.ry = camera-down    in world space  (R[:,1] of 3DGS C2W)
    //   cameraParams.rz = camera-forward in world space  (R[:,2] of 3DGS C2W)

    // Gaussian centre in camera space
    Vector3f mu_c = Vector3f(g.mu - objectRay.o);
    Float z = Dot(cameraParams.rz, mu_c);
    if (z <= 0.f)
        return {-1.f, sigmaCutoff * sigmaCutoff + 1.f, 0.f};

    Float inv_z  = 1.f / z;
    Float x_cam  = Dot(cameraParams.rx, mu_c);
    Float y_cam  = Dot(cameraParams.ry, mu_c);

    // Projected pixel centre of this Gaussian
    Float u = cameraParams.fx * x_cam * inv_z + cameraParams.cx;
    Float v = cameraParams.fy * y_cam * inv_z + cameraParams.cy;

    // Pixel addressed by the current ray.
    // Ray direction in 3DGS camera space (rz = forward, ry = down):
    //   x_dir / z_dir = (px - cx) / fx  →  px = fx*(x_dir/z_dir) + cx
    // PBRT's camera Y is up = -ry, so the dot products already handle the flip:
    //   Dot(ry, d) gives the "downward" component.
    Vector3f d = Normalize(objectRay.d);
    Float dz = Dot(cameraParams.rz, d);
    if (std::abs(dz) < 1e-8f)
        return EvalGaussianAlpha(g, objectRay);
    Float inv_dz = 1.f / dz;
    Float px = cameraParams.fx * Dot(cameraParams.rx, d) * inv_dz + cameraParams.cx;
    Float py = cameraParams.fy * Dot(cameraParams.ry, d) * inv_dz + cameraParams.cy;

    Float du = px - u;
    Float dv = py - v;

    // 2D covariance  Σ_2D = J Σ_3D Jᵀ + 0.3 I
    // where J (2×3 in world coords) is the Jacobian of (u,v) w.r.t. world position:
    //   du/dp = (fx/z) * (rx  -  (x_cam/z) * rz)
    //   dv/dp = (fy/z) * (ry  -  (y_cam/z) * rz)
    Float fx_z = cameraParams.fx * inv_z;
    Float fy_z = cameraParams.fy * inv_z;
    Vector3f J0 = fx_z * (cameraParams.rx - (x_cam * inv_z) * cameraParams.rz);
    Vector3f J1 = fy_z * (cameraParams.ry - (y_cam * inv_z) * cameraParams.rz);

    // Σ_3D * J0  and  Σ_3D * J1
    Vector3f SJ0 = g.sigma * J0;
    Vector3f SJ1 = g.sigma * J1;

    Float s00 = Dot(J0, SJ0) + 0.3f;
    Float s01 = Dot(J0, SJ1);
    Float s11 = Dot(J1, SJ1) + 0.3f;

    Float det = s00 * s11 - s01 * s01;
    if (det < 1e-10f)
        return EvalGaussianAlpha(g, objectRay);

    Float mhd2_2D = (s11 * du * du - 2.f * s01 * du * dv + s00 * dv * dv) / det;
    if (mhd2_2D < 0.f) mhd2_2D = 0.f;

    // Sort by projected centre depth (= t_center in OursCenter convention)
    Float t = Dot(mu_c, d);
    if (t <= 0.f)
        return {-1.f, sigmaCutoff * sigmaCutoff + 1.f, 0.f};

    Float alpha = std::min(g.opacity * std::exp(-0.5f * mhd2_2D), kGaussianAlphaCap);
    return {t, mhd2_2D, alpha};
}

void GaussianCloud::Build2DGrid() {
    // Build a tile grid over image space so IntersectComposite can find Gaussians
    // whose 2D projected footprint covers a pixel in O(cell_size) instead of O(N).
    const int W = cameraParams.width;
    const int H = cameraParams.height;
    grid2DW = (W + kGrid2DCellSize - 1) / kGrid2DCellSize;
    grid2DH = (H + kGrid2DCellSize - 1) / kGrid2DCellSize;
    grid2D.assign(grid2DW * grid2DH, {});

    const Float fx = cameraParams.fx, fy = cameraParams.fy;
    const Float cx = cameraParams.cx, cy = cameraParams.cy;
    const Float sigmaCut = sigmaCutoff;

    for (int gi = 0; gi < (int)gaussians.size(); ++gi) {
        const Gaussian3D &g = gaussians[gi];
        Vector3f mu_c = Vector3f(g.mu - cameraParams.pos);
        Float z = Dot(cameraParams.rz, mu_c);
        if (z <= 0.f) continue;

        Float inv_z = 1.f / z;
        Float x_cam = Dot(cameraParams.rx, mu_c);
        Float y_cam = Dot(cameraParams.ry, mu_c);
        Float u = fx * x_cam * inv_z + cx;
        Float v = fy * y_cam * inv_z + cy;

        // Conservative 2D radius: sigmaCut * max_2D_scale.
        // max_2D_scale ≤ max(fx,fy) * max_3D_scale / z
        Float max3DScale = std::max({g.scale[0], g.scale[1], g.scale[2]});
        Float r2D = sigmaCut * std::max(fx, fy) * max3DScale * inv_z;

        // Pixel bounding box with 1-cell margin
        int xMin = std::max(0, int((u - r2D) / kGrid2DCellSize));
        int xMax = std::min(grid2DW - 1, int((u + r2D) / kGrid2DCellSize));
        int yMin = std::max(0, int((v - r2D) / kGrid2DCellSize));
        int yMax = std::min(grid2DH - 1, int((v + r2D) / kGrid2DCellSize));

        for (int cy2 = yMin; cy2 <= yMax; ++cy2)
            for (int cx2 = xMin; cx2 <= xMax; ++cx2)
                grid2D[cy2 * grid2DW + cx2].push_back(gi);
    }

    size_t totalEntries = 0;
    for (const auto &cell : grid2D) totalEntries += cell.size();
    LOG_VERBOSE("Build2DGrid: %dx%d cells, %.1f avg Gaussians/cell, %zu total entries",
                grid2DW, grid2DH, double(totalEntries) / double(grid2D.size()), totalEntries);
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
        Float splitPos = (centroidBounds.pMin[axis] + centroidBounds.pMax[axis]) * 0.5f;

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
        Float tAlpha;
        Float mhd2;
        Float alpha;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(256);

    const bool use2D = cameraParams.valid;

    auto testGaussian = [&](int gIndex) {
        const Gaussian3D &g = gaussians[gIndex];
        // Per-Gaussian AABB slab test (OursCenter can pass mhd² at projected t even when
        // the ray misses the conservative ellipsoid bounds along the ray).
        // When 2D camera params are available we also accept Gaussians whose AABB is
        // missed (their 3D body may not intersect the ray but their 2D projection does),
        // so we skip the AABB guard in that case.
        if (!use2D && !g.aabb.IntersectP(objectRay.o, objectRay.d, tMax))
            return;

        GaussianAlphaEval eval = use2D ? EvalGaussianAlpha2D(g, objectRay)
                                       : EvalGaussianAlpha(g, objectRay);
        Float t = eval.tAlpha;
        if (t <= 0.f || t >= tMax)
            return;

        if (eval.mhd2 > sigmaCutoff * sigmaCutoff)
            return;
        if (eval.alpha < kGaussianAlphaMinThreshold)
            return;

        candidates.push_back({gIndex, t, eval.tAlpha, eval.mhd2, eval.alpha});
    };

    if (use2D) {
        // 2D projection: look up the tile grid instead of iterating all Gaussians.
        // Recover the pixel addressed by this ray.
        Vector3f d = Normalize(objectRay.d);
        Float dz = Dot(cameraParams.rz, d);
        if (std::abs(dz) > 1e-8f) {
            Float inv_dz = 1.f / dz;
            Float px = cameraParams.fx * Dot(cameraParams.rx, d) * inv_dz + cameraParams.cx;
            Float py = cameraParams.fy * Dot(cameraParams.ry, d) * inv_dz + cameraParams.cy;
            int cx_cell = std::max(0, std::min(grid2DW - 1, int(px / kGrid2DCellSize)));
            int cy_cell = std::max(0, std::min(grid2DH - 1, int(py / kGrid2DCellSize)));
            const auto &cell = grid2D[cy_cell * grid2DW + cx_cell];
            for (int gIndex : cell)
                testGaussian(gIndex);
        }
    } else if (internalAccel == InternalAccel::BRUTE) {
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
        static constexpr int kStackSize = 64;
        StackEntry stack[kStackSize];
        int stackTop = 0;
        stack[stackTop++] = {0, 0.f, tMax};

        while (stackTop > 0) {
            --stackTop;
            int nodeIndex = stack[stackTop].node;
            Float tMin = stack[stackTop].tMin;
            Float nodeTMax = stack[stackTop].tMax;

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

            if (tPlane > tMin && tPlane < nodeTMax) {
                if (stackTop < kStackSize)
                    stack[stackTop++] = {farChild, tPlane, nodeTMax};
            }
            if (stackTop < kStackSize)
                stack[stackTop++] = {nearChild, tMin, std::min(nodeTMax, tPlane)};
        }
    } else {
        if (bvhNodes.empty())
            return {};

        struct StackEntry {
            int node;
            Float tMax;
        };
        static constexpr int kStackSize = 64;
        StackEntry stack[kStackSize];
        int stackTop = 0;
        stack[stackTop++] = {0, tMax};

        while (stackTop > 0) {
            int nodeIndex = stack[--stackTop].node;
            Float nodeTMax = stack[stackTop].tMax;

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
                    if (stackTop < kStackSize) stack[stackTop++] = {node.right, nodeTMax};
                    if (stackTop < kStackSize) stack[stackTop++] = {node.left, nodeTMax};
                } else {
                    if (stackTop < kStackSize) stack[stackTop++] = {node.left, nodeTMax};
                    if (stackTop < kStackSize) stack[stackTop++] = {node.right, nodeTMax};
                }
            }
        }
    }

    if (candidates.empty())
        return CreateDisplayRGBHit(objectRay, tMax, backgroundColor,
                                   kGaussianCompositeFaceIndex, boundMaterial);

    int candidateCountPreDedup = int(candidates.size());
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

    static const CompositeTraceConfig traceCfg = GetCompositeTraceConfig();
    static std::mutex traceMutex;
    static std::atomic<int> traceRayCount(0);
    bool emitTrace = false;
    if (traceCfg.enabled) {
        int rayIndex = traceRayCount.fetch_add(1);
        emitTrace = rayIndex < traceCfg.maxRays;
    }

    std::ostringstream traceOut;
    bool hasTraceOutput = false;
    if (emitTrace) {
        hasTraceOutput = true;
        traceOut << "TRACE_BEGIN"
                 << " ray_o=(" << objectRay.o.x << "," << objectRay.o.y << "," << objectRay.o.z << ")"
                 << " ray_d=(" << objectRay.d.x << "," << objectRay.d.y << "," << objectRay.d.z << ")"
                 << " tMax=" << tMax
                 << " useCenterDepth=" << (useCenterDepth ? 1 : 0)
                 << " preDedup=" << candidateCountPreDedup
                 << " postDedup=" << candidates.size()
                 << "\n";
    }

    RGB rgb(0.f, 0.f, 0.f);
    Float T = 1.f;
    int traceIndex = 0;
    for (const Candidate &c : candidates) {
        const Gaussian3D &g = gaussians[c.index];
        RGB ci = EvaluateGaussianRGB(this, c.index,
                                     GaussianSHViewDirection(shViewDir, g.mu, objectRay));
        Float tBefore = T;
        RGB contrib = tBefore * c.alpha * ci;
        rgb += contrib;
        T *= (1.f - c.alpha);
        if (hasTraceOutput && traceIndex < traceCfg.maxEntries) {
            traceOut << "CAND i=" << traceIndex
                     << " idx=" << c.index
                     << " t=" << c.t
                     << " tAlpha=" << c.tAlpha
                     << " mhd2=" << c.mhd2
                     << " alpha=" << c.alpha
                     << " T_before=" << tBefore
                     << " T_after=" << T
                     << " ci=(" << ci.r << "," << ci.g << "," << ci.b << ")"
                     << " contrib=(" << contrib.r << "," << contrib.g << "," << contrib.b << ")"
                     << "\n";
        }
        ++traceIndex;
        if (T < 1e-4f)
            break;
    }
    rgb += T * backgroundColor;
    if (hasTraceOutput) {
        traceOut << "TRACE_END"
                 << " candidatesUsed=" << traceIndex
                 << " T_final=" << T
                 << " rgb=(" << rgb.r << "," << rgb.g << "," << rgb.b << ")"
                 << " bg=(" << backgroundColor.r << "," << backgroundColor.g << "," << backgroundColor.b
                 << ")"
                 << "\n";
        std::lock_guard<std::mutex> lock(traceMutex);
        std::ofstream traceFile(traceCfg.file, std::ios::app);
        if (traceFile)
            traceFile << traceOut.str();
    }

    return CreateDisplayRGBHit(objectRay, tMax, rgb, kGaussianCompositeFaceIndex, boundMaterial);
}

pstd::optional<ShapeIntersection> GaussianCloud::IntersectBVH(const Ray &objectRay,
                                                              Float tMax) const {
    const bool canUse2DGrid = cameraParams.valid && !grid2D.empty();
    if (bvhNodes.empty() && !canUse2DGrid)
        return {};

    const int N = std::max(1, multiSamples);
    // Paper §3.5 / Table 5: up to 64 independent RR slots per BVH traversal.
    static constexpr int kMaxMultiSamples = 64;
    struct Slot {
        Float currentTMax;
        pstd::optional<ShapeIntersection> best;
    };
    // Use stack storage for up to kMaxMultiSamples; fall back to heap only for exotic configs.
    Slot slotsSmall[kMaxMultiSamples];
    std::vector<Slot> slotsLarge;
    Slot *slots;
    if (N <= kMaxMultiSamples) {
        for (int i = 0; i < N; ++i) slotsSmall[i] = {tMax, {}};
        slots = slotsSmall;
    } else {
        slotsLarge.assign(N, {tMax, {}});
        slots = slotsLarge.data();
    }
    const int frameNum = GetGaussianFrameNumber();

    auto minSlotTMax = [&]() {
        Float m = tMax;
        for (int i = 0; i < N; ++i)
            m = std::min(m, slots[i].currentTMax);
        return m;
    };

    // One alpha eval per primitive; N independent RR decisions (paper §3.5).
    auto testGaussian = [&](int gIndex) {
        const Gaussian3D &g = gaussians[gIndex];
        Float t = EvalIntersectionT(g, objectRay);
        if (t <= 0.f)
            return;

        GaussianAlphaEval eval = EvalGaussianAlpha(g, objectRay);
        if (eval.mhd2 > sigmaCutoff * sigmaCutoff)
            return;
        if (eval.alpha < kGaussianAlphaMinThreshold)
            return;

        Point3f p = objectRay(t);
        Vector3f viewDir = GaussianSHViewDirection(shViewDir, g.mu, objectRay);
        for (int s = 0; s < N; ++s) {
            Slot &slot = slots[s];
            if (t >= slot.currentTMax)
                continue;
            if (!g.aabb.IntersectP(objectRay.o, objectRay.d, slot.currentTMax))
                continue;
            if (TrigHash(p, frameNum * N + s) >= eval.alpha)
                continue;

            slot.currentTMax = t;
            Vector3f d2mu = p - g.mu;
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
            FinishGaussianInteraction(si, n, viewDir);
            si.SetIntersectionProperties(boundMaterial, nullptr, nullptr, objectRay.medium);
            slot.best = ShapeIntersection{si, t};
        }
    };

    auto finalizeStochastic = [&]() -> pstd::optional<ShapeIntersection> {
        if (N == 1) {
            if (slots[0].best)
                return slots[0].best;
            return StochasticMissResult(objectRay, tMax, backgroundColor, boundMaterial);
        }

        const bool visibleBg = HasVisibleBackground(backgroundColor);
        RGB sum(0.f, 0.f, 0.f);
        int hits = 0;
        for (int i = 0; i < N; ++i) {
            const Slot &slot = slots[i];
            if (slot.best) {
                const SurfaceInteraction &intr = slot.best->intr;
                sum += EvaluateGaussianRGB(this, intr.faceIndex, Normalize(intr.shading.dpdu));
                ++hits;
            } else if (visibleBg) {
                sum += backgroundColor;
            }
        }
        if (hits == 0 && !visibleBg)
            return StochasticMissResult(objectRay, tMax, backgroundColor, boundMaterial);

        RGB avg = sum / Float(N);
        return CreateDisplayRGBHit(objectRay, tMax, avg, kGaussianMultiSampleFaceIndex,
                                   boundMaterial);
    };

    // Fast path: 2D tile grid for primary camera rays (same conservative footprint as composite).
    if (canUse2DGrid) {
        Vector3f d = Normalize(objectRay.d);
        Float dz = Dot(cameraParams.rz, d);
        if (std::abs(dz) > 1e-8f) {
            Float inv_dz = 1.f / dz;
            Float px = cameraParams.fx * Dot(cameraParams.rx, d) * inv_dz + cameraParams.cx;
            Float py = cameraParams.fy * Dot(cameraParams.ry, d) * inv_dz + cameraParams.cy;
            int cx_cell = std::max(0, std::min(grid2DW - 1, int(px / kGrid2DCellSize)));
            int cy_cell = std::max(0, std::min(grid2DH - 1, int(py / kGrid2DCellSize)));
            const auto &cell = grid2D[cy_cell * grid2DW + cx_cell];

            std::vector<std::pair<Float, int>> ordered;
            ordered.reserve(cell.size());
            for (int gIndex : cell) {
                Float t = EvalIntersectionT(gaussians[gIndex], objectRay);
                if (t > 0.f && t < tMax)
                    ordered.emplace_back(t, gIndex);
            }
            std::sort(ordered.begin(), ordered.end());
            for (const auto &entry : ordered)
                testGaussian(entry.second);
            return finalizeStochastic();
        }
    }

    // Fixed-size stack avoids heap allocation in the hot path. Depth 64 is sufficient for
    // any BVH built over scenes with up to tens of millions of Gaussians.
    struct StackEntry {
        int node;
        Float tMax;
    };
    static constexpr int kStackSize = 64;
    StackEntry stack[kStackSize];
    int stackTop = 0;
    stack[stackTop++] = {0, tMax};

    while (stackTop > 0) {
        int nodeIndex = stack[--stackTop].node;
        Float nodeTMax = std::min(stack[stackTop].tMax, minSlotTMax());

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
            // Push far child first so near child is on top.
            if (belowFirst) {
                if (stackTop < kStackSize) stack[stackTop++] = {node.right, nodeTMax};
                if (stackTop < kStackSize) stack[stackTop++] = {node.left, nodeTMax};
            } else {
                if (stackTop < kStackSize) stack[stackTop++] = {node.left, nodeTMax};
                if (stackTop < kStackSize) stack[stackTop++] = {node.right, nodeTMax};
            }
        }
    }
    return finalizeStochastic();
}

pstd::optional<ShapeIntersection> GaussianCloud::IntersectKdTree(const Ray &objectRay,
                                                                 Float tMax) const {
    if (kdNodes.empty())
        return {};

    const int N = std::max(1, multiSamples);
    static constexpr int kMaxMultiSamples = 64;
    struct Slot {
        Float currentTMax;
        pstd::optional<ShapeIntersection> best;
    };
    Slot slotsSmall[kMaxMultiSamples];
    std::vector<Slot> slotsLarge;
    Slot *slots;
    if (N <= kMaxMultiSamples) {
        for (int i = 0; i < N; ++i) slotsSmall[i] = {tMax, {}};
        slots = slotsSmall;
    } else {
        slotsLarge.assign(N, {tMax, {}});
        slots = slotsLarge.data();
    }
    const int frameNum = GetGaussianFrameNumber();

    auto minSlotTMax = [&]() {
        Float m = tMax;
        for (int i = 0; i < N; ++i)
            m = std::min(m, slots[i].currentTMax);
        return m;
    };

    auto testGaussian = [&](int gIndex) {
        const Gaussian3D &g = gaussians[gIndex];
        Float t = EvalIntersectionT(g, objectRay);
        if (t <= 0.f)
            return;

        GaussianAlphaEval eval = EvalGaussianAlpha(g, objectRay);
        if (eval.mhd2 > sigmaCutoff * sigmaCutoff)
            return;
        if (eval.alpha < kGaussianAlphaMinThreshold)
            return;

        Point3f p = objectRay(t);
        Vector3f viewDir = GaussianSHViewDirection(shViewDir, g.mu, objectRay);
        for (int s = 0; s < N; ++s) {
            Slot &slot = slots[s];
            if (t >= slot.currentTMax)
                continue;
            if (!g.aabb.IntersectP(objectRay.o, objectRay.d, slot.currentTMax))
                continue;
            if (TrigHash(p, frameNum * N + s) >= eval.alpha)
                continue;

            slot.currentTMax = t;
            Vector3f d2mu = p - g.mu;
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
            FinishGaussianInteraction(si, n, viewDir);
            si.SetIntersectionProperties(boundMaterial, nullptr, nullptr, objectRay.medium);
            slot.best = ShapeIntersection{si, t};
        }
    };

    struct StackEntry {
        int node;
        Float tMin;
        Float tMax;
    };
    static constexpr int kStackSize = 64;
    StackEntry stack[kStackSize];
    int stackTop = 0;
    stack[stackTop++] = {0, 0.f, tMax};

    while (stackTop > 0) {
        --stackTop;
        int nodeIndex = stack[stackTop].node;
        Float tMin = stack[stackTop].tMin;
        Float nodeTMax = std::min(stack[stackTop].tMax, minSlotTMax());

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

        if (tPlane > tMin && tPlane < nodeTMax) {
            if (stackTop < kStackSize)
                stack[stackTop++] = {farChild, tPlane, nodeTMax};
        }
        if (stackTop < kStackSize)
            stack[stackTop++] = {nearChild, tMin, std::min(nodeTMax, tPlane)};
    }
    if (N == 1) {
        if (slots[0].best)
            return slots[0].best;
        return StochasticMissResult(objectRay, tMax, backgroundColor, boundMaterial);
    }

    const bool visibleBg = HasVisibleBackground(backgroundColor);
    RGB sum(0.f, 0.f, 0.f);
    int hits = 0;
    for (int i = 0; i < N; ++i) {
        const Slot &slot = slots[i];
        if (slot.best) {
            const SurfaceInteraction &intr = slot.best->intr;
            sum += EvaluateGaussianRGB(this, intr.faceIndex, Normalize(intr.shading.dpdu));
            ++hits;
        } else if (visibleBg) {
            sum += backgroundColor;
        }
    }
    if (hits == 0 && !visibleBg)
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
