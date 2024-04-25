#pragma once

#include "../math_utils.h"

#ifdef __CUDACC__

using uint32_t = unsigned int;
template <int num>
using SphCoefficients = float3[num];

static constexpr float SHRadMinBound = 0.001f;
// Spherical harmonics coefficients
static constexpr float SH_C0 = 0.28209479177387814f;
static constexpr float SH_C1 = 0.4886025119029199f;
static constexpr float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                                  -1.0925484305920792f, 0.5462742152960396f};
static constexpr float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                                  -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

// TODO : rewrite and optimize

inline __device__ float3 getSphCoeff(const float3 *sphCoefficients, int idx)
{
    return sphCoefficients[idx];
}

static __device__ float3
computeColorFromSH(int deg, const float3 *sphCoefficients, const float3 &rdir, const float3 &rori, bool clamped = true)
{
    // The implementation is loosely based on code for
    // "Differentiable Point-Based Radiance Fields for
    // Efficient View Synthesis" by Zhang et al. (2022)
    float3 rad = SH_C0 * getSphCoeff(sphCoefficients, 0);
    if (deg > 0)
    {
        const float3 &dir = rdir; // safe_normalize(gpos - rori);

        const float x = dir.x;
        const float y = dir.y;
        const float z = dir.z;
        rad = rad - SH_C1 * y * getSphCoeff(sphCoefficients, 1) + SH_C1 * z * getSphCoeff(sphCoefficients, 2) -
              SH_C1 * x * getSphCoeff(sphCoefficients, 3);

        if (deg > 1)
        {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            rad = rad + SH_C2[0] * xy * getSphCoeff(sphCoefficients, 4) + SH_C2[1] * yz * getSphCoeff(sphCoefficients, 5) +
                  SH_C2[2] * (2.0f * zz - xx - yy) * getSphCoeff(sphCoefficients, 6) +
                  SH_C2[3] * xz * getSphCoeff(sphCoefficients, 7) + SH_C2[4] * (xx - yy) * getSphCoeff(sphCoefficients, 8);

            if (deg > 2)
            {
                rad = rad + SH_C3[0] * y * (3.0f * xx - yy) * getSphCoeff(sphCoefficients, 9) +
                      SH_C3[1] * xy * z * getSphCoeff(sphCoefficients, 10) +
                      SH_C3[2] * y * (4.0f * zz - xx - yy) * getSphCoeff(sphCoefficients, 11) +
                      SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * getSphCoeff(sphCoefficients, 12) +
                      SH_C3[4] * x * (4.0f * zz - xx - yy) * getSphCoeff(sphCoefficients, 13) +
                      SH_C3[5] * z * (xx - yy) * getSphCoeff(sphCoefficients, 14) +
                      SH_C3[6] * x * (xx - 3.0f * yy) * getSphCoeff(sphCoefficients, 15);
            }
        }
    }
    rad += 0.5f;
    // TODO : redefine explu as
    // explu_alpha(x) = x if x >= alpha
    //                = exp(x/alpha + log(alpha) -1) otherwise
    // return clamped ? make_float3(rad.x > SHRadMinBound ? rad.x : expf(rad.x - SHRadMinBound) * SHRadMinBound,
    //                              rad.y > SHRadMinBound ? rad.y : expf(rad.y - SHRadMinBound) * SHRadMinBound,
    //                              rad.z > SHRadMinBound ? rad.z : expf(rad.z - SHRadMinBound) * SHRadMinBound)
    //                : rad;
    return clamped ? maxf3(rad, make_float3(0.0f)) : rad;
}

template <typename TParams>
inline __device__ void addSphCoeffGrd(TParams &params, int gId, int idx, const float3 &val)
{
    const int off = idx * 3;
    atomicAdd(&params.mogSphGrd[gId][off + 0], val.x);
    atomicAdd(&params.mogSphGrd[gId][off + 1], val.y);
    atomicAdd(&params.mogSphGrd[gId][off + 2], val.z);
}

template <typename TParams>
__device__ float3 computeColorFromSHBwd(
    int deg, const float3 *sphCoefficients, const float3 &rori, uint32_t gId, const float3 &rdir, float weight, const float3 &rayRadGrd, TParams &params)
{
    // radiance unclamped
    const float3 gradu = computeColorFromSH(deg, sphCoefficients, rdir, rori, false);

    // clamped radiance
    // float3 grad = make_float3(gradu.x > SHRadMinBound ? gradu.x : expf(gradu.x - SHRadMinBound) * SHRadMinBound,
    //                           gradu.y > SHRadMinBound ? gradu.y : expf(gradu.y - SHRadMinBound) * SHRadMinBound,
    //                           gradu.z > SHRadMinBound ? gradu.z : expf(gradu.z - SHRadMinBound) * SHRadMinBound);
    float3 grad = make_float3(gradu.x > 0.0f ? gradu.x : 0.0f,
                              gradu.y > 0.0f ? gradu.y : 0.0f,
                              gradu.z > 0.0f ? gradu.z : 0.0f);

    //
    float3 dL_dRGB = rayRadGrd * weight;
    // dL_dRGB.x *= (gradu.x > SHRadMinBound ? 1 : grad.x);
    // dL_dRGB.y *= (gradu.y > SHRadMinBound ? 1 : grad.y);
    // dL_dRGB.z *= (gradu.z > SHRadMinBound ? 1 : grad.z);
    dL_dRGB.x *= (gradu.x > 0.0f ? 1 : 0);
    dL_dRGB.y *= (gradu.y > 0.0f ? 1 : 0);
    dL_dRGB.z *= (gradu.z > 0.0f ? 1 : 0);

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // ---> rayRad = weight * grad = weight * explu(gsph0 * SH_C0 +
    // 0.5,SHRadMinBound) with explu(x,a) = x if x > a else a*e(x-a)
    // ===> d_rayRad / d_gsph0 =   weight * SH_C0
    addSphCoeffGrd(params, gId, 0, SH_C0 * dL_dRGB);

    if (deg > 0)
    {
        // const float3 sphdiru = gpos - rori;
        // const float3 sphdir = safe_normalize(sphdiru);
        const float3 &sphdir = rdir;

        // float3 dRGBdx = make_float3(0);
        // float3 dRGBdy = make_float3(0);
        // float3 dRGBdz = make_float3(0);

        float x = sphdir.x;
        float y = sphdir.y;
        float z = sphdir.z;

        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;

        addSphCoeffGrd(params, gId, 1, dRGBdsh1 * dL_dRGB);
        addSphCoeffGrd(params, gId, 2, dRGBdsh2 * dL_dRGB);
        addSphCoeffGrd(params, gId, 3, dRGBdsh3 * dL_dRGB);

        // dRGBdx = -SH_C1 * getSphCoeff(sphCoefficients, 3);
        // dRGBdy = -SH_C1 * getSphCoeff(sphCoefficients, 1);
        // dRGBdz = SH_C1 * getSphCoeff(sphCoefficients, 2);

        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);

            addSphCoeffGrd(params, gId, 4, dRGBdsh4 * dL_dRGB);
            addSphCoeffGrd(params, gId, 5, dRGBdsh5 * dL_dRGB);
            addSphCoeffGrd(params, gId, 6, dRGBdsh6 * dL_dRGB);
            addSphCoeffGrd(params, gId, 7, dRGBdsh7 * dL_dRGB);
            addSphCoeffGrd(params, gId, 8, dRGBdsh8 * dL_dRGB);

            // dRGBdx += SH_C2[0] * y * getSphCoeff(sphCoefficients, 4) + SH_C2[2] * 2.f * -x * getSphCoeff(sphCoefficients, 6) +
            //           SH_C2[3] * z * getSphCoeff(sphCoefficients, 7) + SH_C2[4] * 2.f * x * getSphCoeff(sphCoefficients, 8);
            // dRGBdy += SH_C2[0] * x * getSphCoeff(sphCoefficients, 4) + SH_C2[1] * z * getSphCoeff(sphCoefficients, 5) +
            //           SH_C2[2] * 2.f * -y * getSphCoeff(sphCoefficients, 6) +
            //           SH_C2[4] * 2.f * -y * getSphCoeff(sphCoefficients, 8);
            // dRGBdz += SH_C2[1] * y * getSphCoeff(sphCoefficients, 5) +
            //           SH_C2[2] * 2.f * 2.f * z * getSphCoeff(sphCoefficients, 6) + SH_C2[3] * x * getSphCoeff(sphCoefficients, 7);

            if (deg > 2)
            {
                float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);

                addSphCoeffGrd(params, gId, 9, dRGBdsh9 * dL_dRGB);
                addSphCoeffGrd(params, gId, 10, dRGBdsh10 * dL_dRGB);
                addSphCoeffGrd(params, gId, 11, dRGBdsh11 * dL_dRGB);
                addSphCoeffGrd(params, gId, 12, dRGBdsh12 * dL_dRGB);
                addSphCoeffGrd(params, gId, 13, dRGBdsh13 * dL_dRGB);
                addSphCoeffGrd(params, gId, 14, dRGBdsh14 * dL_dRGB);
                addSphCoeffGrd(params, gId, 15, dRGBdsh15 * dL_dRGB);

                // dRGBdx +=
                //     (SH_C3[0] * getSphCoeff(sphCoefficients, 9) * 3.f * 2.f * xy +
                //      SH_C3[1] * getSphCoeff(sphCoefficients, 10) * yz + SH_C3[2] * getSphCoeff(sphCoefficients, 11) * -2.f * xy +
                //      SH_C3[3] * getSphCoeff(sphCoefficients, 12) * -3.f * 2.f * xz +
                //      SH_C3[4] * getSphCoeff(sphCoefficients, 13) * (-3.f * xx + 4.f * zz - yy) +
                //      SH_C3[5] * getSphCoeff(sphCoefficients, 14) * 2.f * xz +
                //      SH_C3[6] * getSphCoeff(sphCoefficients, 15) * 3.f * (xx - yy));

                // dRGBdy += (SH_C3[0] * getSphCoeff(sphCoefficients, 9) * 3.f * (xx - yy) +
                //            SH_C3[1] * getSphCoeff(sphCoefficients, 10) * xz +
                //            SH_C3[2] * getSphCoeff(sphCoefficients, 11) * (-3.f * yy + 4.f * zz - xx) +
                //            SH_C3[3] * getSphCoeff(sphCoefficients, 12) * -3.f * 2.f * yz +
                //            SH_C3[4] * getSphCoeff(sphCoefficients, 13) * -2.f * xy +
                //            SH_C3[5] * getSphCoeff(sphCoefficients, 14) * -2.f * yz +
                //            SH_C3[6] * getSphCoeff(sphCoefficients, 15) * -3.f * 2.f * xy);

                // dRGBdz += (SH_C3[1] * getSphCoeff(sphCoefficients, 10) * xy +
                //            SH_C3[2] * getSphCoeff(sphCoefficients, 11) * 4.f * 2.f * yz +
                //            SH_C3[3] * getSphCoeff(sphCoefficients, 12) * 3.f * (2.f * zz - xx - yy) +
                //            SH_C3[4] * getSphCoeff(sphCoefficients, 13) * 4.f * 2.f * xz +
                //            SH_C3[5] * getSphCoeff(sphCoefficients, 14) * (xx - yy));
            }
        }

        // The view direction is an input to the computation. View direction
        // is influenced by the Gaussian's mean, so SHs gradients
        // must propagate back into 3D position.
        // const float3 dL_ddir = make_float3(dot(dRGBdx, dL_dRGB), dot(dRGBdy, dL_dRGB), dot(dRGBdz, dL_dRGB));

        // Account for normalization of direction
        // dL_dmean += safe_normalize_bw(sphdiru, dL_ddir);

        // // Gradients of loss w.r.t. Gaussian means, but only the portion
        // // that is caused because the mean affects the view-dependent color.
        // // Additional mean gradient is accumulated in below methods.
        // atomicAdd(&params.mogPosGrd[gId][0], dL_dmean.x);
        // atomicAdd(&params.mogPosGrd[gId][1], dL_dmean.y);
        // atomicAdd(&params.mogPosGrd[gId][2], dL_dmean.z);
    }

    return grad;
}

#endif
