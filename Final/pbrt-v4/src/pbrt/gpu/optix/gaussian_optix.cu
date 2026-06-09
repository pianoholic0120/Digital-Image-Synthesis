// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

// Included from optix.cu — device programs for 3D Gaussian stochastic intersection.

#include <pbrt/gpu/optix/optix.h>
#include <pbrt/gaussian_eval.h>
#include <pbrt/util/plyloader_3dgs.h>
#include <pbrt/util/trighash.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/wavefront/intersect.h>

using namespace pbrt;

PBRT_CPU_GPU inline Float EvalGaussianMeanDepthT(const Gaussian3D &g, const Ray &ray) {
    Vector3f diff = Vector3f(g.mu - ray.o);
    Vector3f sinvd = g.sigmaInv * ray.d;
    Float denom = Dot(ray.d, sinvd);
    if (std::abs(denom) < 1e-8f)
        return Infinity;
    return Dot(diff, sinvd) / denom;
}

PBRT_CPU_GPU inline Float EvalGaussianIntersectionT(const Gaussian3D &g, const Ray &ray,
                                                    bool useCenterDepth) {
    if (useCenterDepth) {
        Vector3f oc = Vector3f(g.mu - ray.o);
        return Dot(oc, ray.d);
    }
    return EvalGaussianMeanDepthT(g, ray);
}

PBRT_CPU_GPU inline Float EvalGaussianAlphaDepthT(const Gaussian3D &g, const Ray &ray,
                                                  bool useCenterDepth, Float tDepth) {
    if (!useCenterDepth)
        return tDepth;
    Float tMean = EvalGaussianMeanDepthT(g, ray);
    if (tMean <= 0.f)
        return tDepth;
    return tMean;
}

extern "C" __global__ void __intersection__gaussian() {
    GaussianRecord &rec = *((GaussianRecord *)optixGetSbtDataPointer());
    int idx = optixGetPrimitiveIndex();
    const Gaussian3D &g = rec.d_gaussians[idx];

    float3 org = optixGetObjectRayOrigin();
    float3 dir = optixGetObjectRayDirection();
    Float tMax = optixGetRayTmax();
    Ray ray(Point3f(org.x, org.y, org.z), Vector3f(dir.x, dir.y, dir.z));

    Float t = EvalGaussianIntersectionT(g, ray, rec.useCenterDepth);
    if (t <= 0 || t >= tMax)
        return;

    Float tAlpha = EvalGaussianAlphaDepthT(g, ray, rec.useCenterDepth, t);
    Point3f pAlpha = ray(tAlpha);
    Vector3f d2mu = pAlpha - g.mu;
    Float mhd2 = Dot(d2mu, g.sigmaInv * d2mu);
    if (mhd2 > rec.sigmaCutoff * rec.sigmaCutoff)
        return;

    Float alpha = std::min(g.opacity * std::exp(-0.5f * mhd2), kGaussianAlphaCap);
    if (alpha < kGaussianAlphaMinThreshold)
        return;

    Point3f p = ray(t);
    if (TrigHash(p, params.gaussianFrameNumber) >= alpha)
        return;

    optixReportIntersection(t, 0 /* hit kind */, FloatToBits(float(idx)));
}

extern "C" __global__ void __closesthit__gaussian() {
    GaussianRecord &rec = *((GaussianRecord *)optixGetSbtDataPointer());
    int gIndex = int(BitsToFloat(optixGetAttribute_0()));
    const Gaussian3D &g = rec.d_gaussians[gIndex];

    float3 worg = optixGetWorldRayOrigin();
    float3 wdir = optixGetWorldRayDirection();
    Transform worldFromInstance = getWorldFromInstance();
    Point3f org = worldFromInstance.ApplyInverse(Point3f(worg.x, worg.y, worg.z));
    Vector3f dir = worldFromInstance.ApplyInverse(Vector3f(wdir.x, wdir.y, wdir.z));
    Ray ray(org, dir, optixGetRayTime());
    Float t = optixGetRayTmax();
    Point3f p = ray(t);
    Vector3f d2mu = p - g.mu;
    Normal3f n(Normalize(d2mu));
    if (rec.reverseOrientation)
        n = -n;

    SurfaceInteraction intr;
    intr.pi = Point3fi(p);
    intr.n = n;
    intr.shading.n = n;
    intr.wo = -Vector3f(wdir.x, wdir.y, wdir.z);
    intr.uv = Point2f(0, 0);
    intr.time = optixGetRayTime();
    intr.faceIndex = gIndex;
    if (rec.mediumInterface && rec.mediumInterface->IsMediumTransition())
        intr.mediumInterface = rec.mediumInterface;
    intr.material = rec.material;

    intr = worldFromInstance(intr);

    ProcessClosestIntersection(intr);
}

extern "C" __global__ void __anyhit__shadowGaussian() {}

extern "C" __global__ void __closesthit__randomHitGaussian() {
    GaussianRecord &rec = *((GaussianRecord *)optixGetSbtDataPointer());
    RandomHitPayload *p = getPayload<RandomHitPayload>();

    int gIndex = int(BitsToFloat(optixGetAttribute_0()));
    const Gaussian3D &g = rec.d_gaussians[gIndex];

    float3 worg = optixGetWorldRayOrigin();
    float3 wdir = optixGetWorldRayDirection();
    Transform worldFromInstance = getWorldFromInstance();
    Point3f org = worldFromInstance.ApplyInverse(Point3f(worg.x, worg.y, worg.z));
    Vector3f dir = worldFromInstance.ApplyInverse(Vector3f(wdir.x, wdir.y, wdir.z));
    Ray ray(org, dir, optixGetRayTime());
    Float t = optixGetRayTmax();
    Point3f pObj = ray(t);
    Vector3f d2mu = pObj - g.mu;
    Normal3f n(Normalize(d2mu));
    if (rec.reverseOrientation)
        n = -n;

    SurfaceInteraction intr;
    intr.pi = Point3fi(pObj);
    intr.n = n;
    intr.shading.n = n;
    intr.wo = -Vector3f(wdir.x, wdir.y, wdir.z);
    intr.uv = Point2f(0, 0);
    intr.time = optixGetRayTime();
    intr.faceIndex = gIndex;
    p->intr = intr;

    if (rec.material == p->material)
        p->wrs.Add([&] __device__ () { return intr; }, 1.f);
}
