// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#include "../mogTracing/params.h"
#include "../mogTracing/utils.h"
#include "../optix_utils.h"
#include "../random_utils.h"
#include "../ray_data.h"

extern "C"
{
    __constant__ MoGTracingBwdParams params;
}

static constexpr bool MoGSurfPrimitive = (MOGTRACING_PRIMITIVE_TYPE == 6) || (MOGTRACING_PRIMITIVE_TYPE == 7);
static constexpr bool MoGCustomPrimitive = (MOGTRACING_PRIMITIVE_TYPE == 5);
static constexpr unsigned int MoGTracingHitMode = MOGTRACING_HIT_MODE;
static constexpr unsigned int MoGTracingAHMaxNumHitPerSlab = MOGTRACING_MAXNUMHITS_PER_SLAB;
static constexpr unsigned int MOGTracingPatchSize = MOGTRACING_PATCH_SIZE;

struct RayPayload
{
    unsigned int ahNumHits; // number of valid hits in ahHitTable
#if MOGTRACING_SAMPLING_MODE
    unsigned int rndSeed;
#endif
    float2 ahHitTable[MoGTracingAHMaxNumHitPerSlab]; // hit data : x = hitT, y = primId
};

static __device__ __inline__ float getGaussianIndex(uint32_t primitiveIndex)
{
    return MoGCustomPrimitive ? primitiveIndex : primitiveIndex / params.gPrimNumTri;
}

static __device__ __inline__ float2 intersectAABB(const OptixAabb &aabb, const float3 &rayOri, const float3 &rayDir)
{
    const float3 t0 = (make_float3(aabb.minX, aabb.minY, aabb.minZ) - rayOri) / rayDir;
    const float3 t1 = (make_float3(aabb.maxX, aabb.maxY, aabb.maxZ) - rayOri) / rayDir;
    const float3 tmax = maxf3(t0, t1);
    const float3 tmin = minf3(t0, t1);
    return float2{fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z))), fminf(tmax.x, fminf(tmax.y, tmax.z))};
}

static __device__ __inline__ void trace(
    RayPayload *p, const float3 &rayOri, const float3 &rayDir, const float tmin, const float tmax)
{
    const unsigned long long ahHitTablePtr = reinterpret_cast<unsigned long long>(&p->ahHitTable);
    unsigned int ahHitTablePtr0 = ahHitTablePtr >> 32;
    unsigned int ahHitTablePtr1 = ahHitTablePtr & 0x00000000ffffffff;

    // Trace the ray against our scene hierarchy
    optixTrace(params.handle, rayOri, rayDir,
               tmin,                     // Min intersection distance
               tmax,                     // Max intersection distance
               0.0f,                     // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | (MoGSurfPrimitive ? OPTIX_RAY_FLAG_NONE : OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES),
               0, // SBT offset   -- See SBT discussion
               1, // SBT stride   -- See SBT discussion
               0, // missSBTIndex -- See SBT discussion
               reinterpret_cast<unsigned int &>(p->ahNumHits), ahHitTablePtr0, ahHitTablePtr1
#if MOGTRACING_SAMPLING_MODE
               ,
               reinterpret_cast<unsigned int &>(p->rndSeed)
#endif
    );
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 rayOri[MOGTracingPatchSize][MOGTracingPatchSize];
    float3 rayDir[MOGTracingPatchSize][MOGTracingPatchSize];
    float rayTrm[MOGTracingPatchSize][MOGTracingPatchSize];

    float3 totalRayRad[MOGTracingPatchSize][MOGTracingPatchSize];
    float3 rayRadGrd[MOGTracingPatchSize][MOGTracingPatchSize];
    float3 accumulatedRayRad[MOGTracingPatchSize][MOGTracingPatchSize];

    float rayDnsGrd[MOGTracingPatchSize][MOGTracingPatchSize];
    float rayError[MOGTracingPatchSize][MOGTracingPatchSize];
    float accumulatedRayTrm[MOGTracingPatchSize][MOGTracingPatchSize];

    float totalRayHit[MOGTracingPatchSize][MOGTracingPatchSize];
    float rayHitGrd[MOGTracingPatchSize][MOGTracingPatchSize];
    float accumulatedRayHit[MOGTracingPatchSize][MOGTracingPatchSize];

    const int startIdxX = idx.x * MOGTracingPatchSize;
    const int startIdxY = idx.y * MOGTracingPatchSize;

    float2 minMaxT = make_float2(1e9, -1e9);

#pragma unroll
    for (int j = 0; j < MOGTracingPatchSize; ++j)
    {
        const int y = fminf(startIdxY + j, params.frameBounds.y);
#pragma unroll
        for (int i = 0; i < MOGTracingPatchSize; ++i)
        {
            const int x = fminf(startIdxX + i, params.frameBounds.x);
            rayOri[j][i] =
                make_float3(params.rayOri[idx.z][y][x][0], params.rayOri[idx.z][y][x][1], params.rayOri[idx.z][y][x][2]);
            rayDir[j][i] =
                make_float3(params.rayDir[idx.z][y][x][0], params.rayDir[idx.z][y][x][1], params.rayDir[idx.z][y][x][2]);
            rayTrm[j][i] = 1;

            const float2 sampleMinMaxT = intersectAABB(params.aabb, rayOri[j][i], rayDir[j][i]);
            minMaxT.x = fminf(minMaxT.x, sampleMinMaxT.x);
            minMaxT.y = fmaxf(minMaxT.y, sampleMinMaxT.y);

            totalRayRad[j][i] =
                make_float3(params.rayRad[idx.z][y][x][0], params.rayRad[idx.z][y][x][1], params.rayRad[idx.z][y][x][2]);
            rayRadGrd[j][i] = make_float3(
                params.rayRadGrd[idx.z][y][x][0], params.rayRadGrd[idx.z][y][x][1], params.rayRadGrd[idx.z][y][x][2]);
            accumulatedRayRad[j][i] = make_float3(0);

            rayDnsGrd[j][i] = params.rayDnsGrd[idx.z][y][x][0];
            rayError[j][i] = params.rayError[idx.z][y][x][0];
            accumulatedRayTrm[j][i] = 1 - params.rayDns[idx.z][y][x][0];

            totalRayHit[j][i] = params.rayHit[idx.z][y][x][0];
            rayHitGrd[j][i] = params.rayHitGrd[idx.z][y][x][0];
            accumulatedRayHit[j][i] = 0;
        }
    }

    // ray- aabb intersection to determine number of segments
    constexpr float epsT = 1e-9;
    const float slabSpacing = params.slabSpacing;
    float startT = fmaxf(0.0f, minMaxT.x - epsT);

    const float minTransmittance = params.minTransmittance;
    const int sphDegree = params.sphDegree;
    const bool useGWeights = params.renderOpts & MOGRenderUseGWeights;

    float transmit = 1.0f;
    RayPayload p;

#if MOGTRACING_SAMPLING_MODE
    p.rndSeed = tea<16>(dim.x * idx.y + idx.x, params.frameNumber);
#endif

    while ((startT <= minMaxT.y) && (transmit > minTransmittance))
    {
        p.ahNumHits = 0;
#pragma unroll
        for (int i = 0; i < MoGTracingAHMaxNumHitPerSlab; ++i)
        {
            p.ahHitTable[i] = make_float2(1e9, 1e9);
        }

        // TODO : add a GOOD ray jittering scheme over the patch (not bounded by the convex hull of the rayDirs if possible)
        const float3 sampleRayOri = rayOri[MOGTracingPatchSize / 2][MOGTracingPatchSize / 2];
        const float3 sampleRayDir = rayDir[MOGTracingPatchSize / 2][MOGTracingPatchSize / 2];

        trace(&p, sampleRayOri, sampleRayDir, startT + epsT, startT + slabSpacing);
        if (p.ahNumHits == 0)
        {
            startT += slabSpacing;
            continue;
        }

#ifndef MOGTRACING_TOPK_HITS
        // in case we got more hits than available slots, start the next ray from the last hit
        if (p.ahNumHits == MoGTracingAHMaxNumHitPerSlab)
        {
            startT = p.ahHitTable[p.ahNumHits - 1].x;
        }
        else
#endif
        {
            startT += slabSpacing;
        }

        for (int i = 0; (i < p.ahNumHits) && (transmit > minTransmittance); i++)
        {
            transmit = 0.0f;

            const uint32_t primId = float_as_uint(p.ahHitTable[i].y);
            const uint32_t gId = getGaussianIndex(primId);

            float gdns = 1.0f;
#if MOGTRACING_SAMPLING_MODE
            if (!params.renderOpts & MOGRenderDnsHitSampling)
            {
                gdns = params.mogDns[gId][0];
            }
#else
            gdns = params.mogDns[gId][0];
#endif
            const float3 gpos = make_float3(params.mogPos[gId][0], params.mogPos[gId][1], params.mogPos[gId][2]);
            const float4 grot =
                make_float4(params.mogRot[gId][0], params.mogRot[gId][1], params.mogRot[gId][2], params.mogRot[gId][3]);
            const float3 gscl = make_float3(params.mogScl[gId][0], params.mogScl[gId][1], params.mogScl[gId][2]);

            float3 sphCoefficients[MOGTRACING_MAX_NUM_RADIANCE_SPH_COEFFS];
#pragma unroll
            for (unsigned int j = 0; j < MOGTRACING_MAX_NUM_RADIANCE_SPH_COEFFS; ++j)
            {
                const int off = j * 3;
                sphCoefficients[j] = make_float3(params.mogSph[gId][off + 0], params.mogSph[gId][off + 1], params.mogSph[gId][off + 2]);
            }
            
            float33 grotMat;
            rotationMatrix(make_float4(grot.x, grot.y, grot.z, grot.w), grotMat);
            const float3 giscl = make_float3(1 / gscl.x, 1 / gscl.y, 1 / gscl.z);

#pragma unroll
            for (int j = 0; j < MOGTracingPatchSize; ++j)
            {
#pragma unroll
                for (int k = 0; k < MOGTracingPatchSize; ++k)
                {
                    if (rayTrm[k][j] > minTransmittance)
                    {
                        // project ray in the gaussian
                        const float3 gposc = (rayOri[k][j] - gpos);
                        const float3 gposcr = (gposc * grotMat);
                        const float3 gro = giscl * gposcr;
                        const float3 rayDirR = rayDir[k][j] * grotMat;
                        const float3 grdu = giscl * rayDirR;
                        const float3 grd = safe_normalize(grdu);
                        float grayDist;
                        if (MoGSurfPrimitive)
                        {
                            const float3 surfelNm = fetchSurfelNm<MOGTRACING_PRIMITIVE_TYPE>(primId % params.gPrimNumTri);
                            const float ghitT = -dot(surfelNm, gro) / dot(surfelNm, grd);
                            const float3 ghitPos = gro + grd * ghitT;
                            grayDist = dot(ghitPos, ghitPos);
                        }
                        else
                        {
                            const float3 gcrod = cross(grd, gro);
                            grayDist = dot(gcrod, gcrod);
                        }

#if MOGTRACING_TESSERACTIC_KERNEL
                        const float gres = expf(-0.0555f * grayDist * grayDist);
#else
                        const float gres = expf(-0.5f * grayDist);
#endif
                        const float galpha = fminf(gres * gdns, params.alphaMaxValue);

                        if ((gres > params.hitMinGaussianResponse) && (galpha > params.alphaMinThreshold))
                        {

                            const float3 grdd = grd * dot(grd, -1 * gro);
                            const float3 grds = gscl * grdd;
                            const float gsqdist = dot(grds, grds);
                            const float gdist = sqrtf(gsqdist);

                            const float weight = galpha * rayTrm[k][j];

                            const float nextTransmit = (1 - galpha) * rayTrm[k][j];

                            // ---> hitT = accumulatedHitT + galpha * prevTrm * gdist + (1-galpha) * prevTrm * residualHitT
                            accumulatedRayHit[k][j] += weight * gdist;
                            const float residualHitT =
                                fmaxf((nextTransmit <= minTransmittance ? 0 : (totalRayHit[k][j] - accumulatedRayHit[k][j]) / nextTransmit),
                                      0);

                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ---> hitT = accumulatedHitT + galpha * prevTrm * gdist + (1-galpha) * prevTrm * residualHitT
                            //
                            // ===> d_hitT / d_galpha = gdist * prevTrm - residualHitT * prevTrm
                            //                        = (gdist - residualHitT) * prevTrm
                            //
                            const float galphaRayHitGrd = (gdist - residualHitT) * rayTrm[k][j] * rayHitGrd[k][j];
                            //
                            // ===> d_hitT / d_gsqdist = weight / (2*gdist)
                            // ===> d_gsqdist / d_grds =  2 * grds
                            const float3 grdsRayHitGrd = gsqdist > 0.0f ? ((2 * grds * weight) / (2 * gdist)) * rayHitGrd[k][j] : make_float3(0.0f);

                            // ---> grds = gscl * grd * dot(grd, -1 * gro)
                            //
                            // ===> d_grds / d_gscl =  grd * dot(grd, -1 * gro)
                            const float3 gsclRayHitGrd = grdd * grdsRayHitGrd;
                            // ===> d_grds / d_grd =  - gscl * grd * (2 dot(grd, -1 * gro)
                            const float3 grdRayHitGrd = -gscl * make_float3(2 * grd.x * gro.x + grd.y * gro.y + grd.z * gro.z, grd.x * gro.x + 2 * grd.y * gro.y + grd.z * gro.z, grd.x * gro.x + grd.y * gro.y + 2 * grd.z * gro.z) * grdsRayHitGrd;
                            //
                            // ===> d_grds / d_gro = - gscl * grd * grd
                            const float3 groRayHitGrd = -gscl * grd * grd * grdsRayHitGrd;

                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ---> rayDns = 1 - prevTrm * (1-galpha) * nextTrm
                            //             = 1 - (1-galpha) * prevTrm * nextTrm
                            // ===> d_rayDns / d_galpha = prevTrm * nextTrm = residualTrm
                            const float residualTrm = galpha < 0.999999f ? accumulatedRayTrm[k][j] / (1 - galpha) : rayTrm[k][j];
                            const float galphaRayDnsGrd = residualTrm * rayDnsGrd[k][j];

                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // compute the gradient wrt to the sph coefficients and position (through the sph view
                            // direction)
                            float3 gradPosGrd = make_float3(0);
                            const float3 grad = computeColorFromSHBwd(sphDegree, &sphCoefficients[0], rayOri[k][j], gId, gpos, weight, rayRadGrd[k][j], params, gradPosGrd);

                            // >>> rayRadiance = accumulatedRayRad + weigth * rayRad + (1-galpha)*transmit * residualRayRad
                            const float3 rayRad = weight * grad;
                            accumulatedRayRad[k][j] += rayRad;
                            const float3 residualRayRad = maxf3((nextTransmit <= minTransmittance ? make_float3(0) : (totalRayRad[k][j] - accumulatedRayRad[k][j]) / nextTransmit),
                                                                make_float3(0));

                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ---> rayDns = 1 - prevTrm * (1-galpha) * nextTrm
                            //             = 1 - (1-galpha) * prevTrm * nextTrm
                            // ===> d_rayDns / d_gdns = residualTrm * gres
                            //
                            // ---> rayRadiance = accumulatedRayRad + galpha * transmit * grad + (1-galpha) * transmit *
                            // residualRayRad
                            //                  = accumulatedRayRad + gdns * gres * transmit * grad + (1-gdns*gres) *
                            //                  transmit * residualRayRad
                            // ===> d_rayRad / d_gdns = gres * transmit * grad - gres * transmit * residualRayRad
                            atomicAdd(
                                &params.mogDnsGrd[gId][0],
                                gres * (galphaRayHitGrd + galphaRayDnsGrd + rayTrm[k][j] * (grad.x - residualRayRad.x) * rayRadGrd[k][j].x +
                                        rayTrm[k][j] * (grad.y - residualRayRad.y) * rayRadGrd[k][j].y +
                                        rayTrm[k][j] * (grad.z - residualRayRad.z) * rayRadGrd[k][j].z));

                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ---> rayDns = 1 - prevTrm * (1-galpha) * nextTrm
                            //             = 1 - (1-galpha) * prevTrm * nextTrm
                            // ===> d_rayDns / d_gres = residualTrm * gdns
                            //
                            // ---> rayRadiance = accumulatedRayRad + galpha * transmit * grad + (1 - galpha) * transmit *
                            // residualRayRad
                            //                  = accumulatedRayRad + gdns * gres * transmit * grad + (1 - gdns * gres) *
                            //                  transmit * residualRayRad
                            // ===> d_rayRad / d_gres = gdns * transmit * grad - gdns * transmit * residualRayRad
                            const float gresGrd =
                                gdns * (galphaRayHitGrd + galphaRayDnsGrd + rayTrm[k][j] * (grad.x - residualRayRad.x) * rayRadGrd[k][j].x +
                                        rayTrm[k][j] * (grad.y - residualRayRad.y) * rayRadGrd[k][j].y +
                                        rayTrm[k][j] * (grad.z - residualRayRad.z) * rayRadGrd[k][j].z);

#if MOGTRACING_TESSERACTIC_KERNEL
                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ---> gres = exp(-0.0555 * grayDist * grayDist)
                            // ===> d_gres / d_grayDist = -0.111 * grayDist * exp(-0.555 * grayDist * grayDist)
                            //                          = -0.111 * grayDist * gres
                            const float grayDistGrd = -0.111f * grayDist * gres * gresGrd;
#else
                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ---> gres = exp(-0.5 * grayDist)
                            // ===> d_gres / d_grayDist = -0.5 * exp(-0.5 * grayDist)
                            //                          = -0.5 * gres
                            const float grayDistGrd = -0.5f * gres * gresGrd;
#endif

                            float3 grdGrd, groGrd;
                            if (MoGSurfPrimitive)
                            {
                                const float3 surfelNm = fetchSurfelNm<MOGTRACING_PRIMITIVE_TYPE>(primId % params.gPrimNumTri);
                                const float doSurfelGro = dot(surfelNm, gro);
                                const float dotSurfelGrd = dot(surfelNm, grd); // cannot be null otherwise no hit
                                const float ghitT = -doSurfelGro / dotSurfelGrd;
                                const float3 ghitPos = gro + grd * ghitT;

                                // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                // ---> grayDist = dot(ghitPos, ghitPos)
                                //               = ghitPos.x^2 + ghitPos.y^2 + ghitPos.z^2
                                // ===> d_grayDist / d_ghitPos = 2*ghitPos
                                const float3 ghitPosGrd = 2 * ghitPos * grayDistGrd;

                                // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                // ---> ghitPos = gro + grd * ghitT
                                //
                                // ===> d_ghitPos / d_gro = 1
                                // ===> d_ghitPos / d_grd = ghitT
                                groGrd = ghitPosGrd;
                                grdGrd = ghitT * ghitPosGrd;
                                // ===> d_ghitPos / d_ghitT = grd
                                const float ghitTGrd = sum(grd * ghitPosGrd);

                                // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                // ---> ghitT = -dot(surfelNm, gro) / dot(surfNm, grd)
                                //
                                // ===> d_ghitT / d_gro = -surfelNm / dot(surfNm, grd)
                                // ===> d_ghitT / d_dotSurfelGrd = dot(surfelNm, gro) / dotSurfelGrd^2
                                groGrd += (-surfelNm * ghitTGrd) / dotSurfelGrd;
                                const float dotSurfelGrdGrd = (doSurfelGro * ghitTGrd) / (dotSurfelGrd * dotSurfelGrd);
                                // ===> d_dotSurfelGrd / d_grd = surfelNm
                                grdGrd += surfelNm * dotSurfelGrdGrd;
                            }
                            else
                            {
                                const float3 gcrod = cross(grd, gro);

                                // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                // ---> grayDist = dot(gcrod, gcrod)
                                //               = gcrod.x^2 + gcrod.y^2 + gcrod.z^2
                                // ===> d_grayDist / d_gcrod = 2*gcrod
                                const float3 gcrodGrd = 2 * gcrod * grayDistGrd;

                                // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                // ---> gcrod = cross(grd, gro)
                                // ---> gcrod.x = grd.y * gro.z - grd.z * gro.y
                                // ---> gcrod.y = grd.z * gro.x - grd.x * gro.z
                                // ---> gcrod.z = grd.x * gro.y - grd.y * gro.x
                                grdGrd = make_float3(gcrodGrd.z * gro.y - gcrodGrd.y * gro.z,
                                                     gcrodGrd.x * gro.z - gcrodGrd.z * gro.x,
                                                     gcrodGrd.y * gro.x - gcrodGrd.x * gro.y);
                                groGrd = make_float3(gcrodGrd.y * grd.z - gcrodGrd.z * grd.y,
                                                     gcrodGrd.z * grd.x - gcrodGrd.x * grd.z,
                                                     gcrodGrd.x * grd.y - gcrodGrd.y * grd.x);
                            }

                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ---> gro = (1/gscl)*gposcr
                            // ===> d_gro / d_gscl = -gposcr/(gscl*gscl)
                            // ===> d_gro / d_gposcr = (1/gscl)
                            const float3 gsclGrdGro = make_float3((-gposcr.x / (gscl.x * gscl.x)),
                                                                  (-gposcr.y / (gscl.y * gscl.y)),
                                                                  (-gposcr.z / (gscl.z * gscl.z))) *
                                                      (groGrd + groRayHitGrd);
                            const float3 gposcrGrd = giscl * (groGrd + groRayHitGrd);

                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ---> gposcr = matmul(gposc, grotMat)
                            // ===> d_gposcr / d_gposc = matmul_bw_vec(grotMat)
                            // ===> d_gposcr / d_grotmat = matmul_bw_mat(gposc)
                            const float3 gposcGrd = matmul_bw_vec(grotMat, gposcrGrd);
                            const float4 grotGrdPoscr = matmul_bw_quat(gposc, gposcrGrd, grot);

                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ---> gposc = rayOri - gpos
                            // ===> d_gposc / d_gpos = -1
                            const float3 rayMoGPosGrd = gradPosGrd - gposcGrd;
                            atomicAdd(&params.mogPosGrd[gId][0], rayMoGPosGrd.x);
                            atomicAdd(&params.mogPosGrd[gId][1], rayMoGPosGrd.y);
                            atomicAdd(&params.mogPosGrd[gId][2], rayMoGPosGrd.z);

                            atomicAdd(&params.mogPosGrdSq[gId][0], length(rayMoGPosGrd));

                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ---> grd = safe_normalize(grdu)
                            // ===> d_grd / d_grdu = safe_normalize_bw(grd)
                            const float3 grduGrd = safe_normalize_bw(grdu, grdGrd + grdRayHitGrd);

                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ---> grdu = (1/gscl)*rayDirR
                            // ===> d_grdu / d_gscl = -rayDirR/(gscl*gscl)
                            // ===> d_grdu / d_rayDirR = (1/gscl)
                            atomicAdd(&params.mogSclGrd[gId][0], gsclRayHitGrd.x + gsclGrdGro.x + (-rayDirR.x / (gscl.x * gscl.x)) * grduGrd.x);
                            atomicAdd(&params.mogSclGrd[gId][1], gsclRayHitGrd.y + gsclGrdGro.y + (-rayDirR.y / (gscl.y * gscl.y)) * grduGrd.y);
                            atomicAdd(&params.mogSclGrd[gId][2], gsclRayHitGrd.z + gsclGrdGro.z + (-rayDirR.z / (gscl.z * gscl.z)) * grduGrd.z);
                            const float3 rayDirRGrd = giscl * grduGrd;

                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ---> rayDirR = matmul(rayDir, grotMat)
                            // ===> d_rayDirR / d_grotmat = matmul_bw_mat(rayDir, grotMat)
                            const float4 grotGrdRayDirR = matmul_bw_quat(rayDir[k][j], rayDirRGrd, grot);
                            atomicAdd(&params.mogRotGrd[gId][0], grotGrdPoscr.x + grotGrdRayDirR.x);
                            atomicAdd(&params.mogRotGrd[gId][1], grotGrdPoscr.y + grotGrdRayDirR.y);
                            atomicAdd(&params.mogRotGrd[gId][2], grotGrdPoscr.z + grotGrdRayDirR.z);
                            atomicAdd(&params.mogRotGrd[gId][3], grotGrdPoscr.w + grotGrdRayDirR.w);

                            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // error back projection
                            // THIS IS NOT REALLY A GRADIENT, the kernel is being abused to
                            // compute weight * error, since it shared the same computational structure as a gradient
                            if (useGWeights)
                            {
                                atomicAdd(&params.mogErrorBack[gId][0], weight * rayError[k][j]);
                            }

                            rayTrm[k][j] = nextTransmit;
                        }
                    }

                    transmit = fmaxf(transmit, rayTrm[k][j]);
                }
            }
        }
    }
}

extern "C" __global__ void __intersection__is()
{
    const float3 ro = optixGetWorldRayOrigin();
    const float3 rd = optixGetWorldRayDirection();

    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();

    const unsigned int gId = getGaussianIndex(optixGetPrimitiveIndex());

    const float3 gPos = make_float3(params.mogPos[gId][0], params.mogPos[gId][1], params.mogPos[gId][2]);
    const float4 gRot =
        make_float4(params.mogRot[gId][0], params.mogRot[gId][1], params.mogRot[gId][2], params.mogRot[gId][3]);
    const float3 gScl = make_float3(params.mogScl[gId][0], params.mogScl[gId][1], params.mogScl[gId][2]);

    float33 rot;
    rotationMatrix(make_float4(gRot.x, gRot.y, gRot.z, gRot.w), rot);
    const float3 giscl = make_float3(1 / gScl.x, 1 / gScl.y, 1 / gScl.z);
    const float3 gro = giscl * ((ro - gPos) * rot);
    const float3 grdu = giscl * ((rd * rot));
    const float t = fabs(dot(gro, grdu) / dot(grdu, grdu));

    if (t > tmin && t < tmax)
    {
        const float3 grd = safe_normalize(grdu);
        const float gdns = params.mogDns[gId][0];
        const float3 gcrod = cross(grd, gro);
        const float gres = expf(-0.5 * dot(gcrod, gcrod));
        if (gres > params.hitMinGaussianResponse)
        {
            optixReportIntersection(t, 0);
        }
    }
}

extern "C" __global__ void __anyhit__ah()
{
    const unsigned int ahHitTablePtr0 = optixGetPayload_1();
    const unsigned int ahHitTablePtr1 = optixGetPayload_2();
    float2 *ahHitTablePtr =
        reinterpret_cast<float2 *>(static_cast<unsigned long long>(ahHitTablePtr0) << 32 | ahHitTablePtr1);
    const unsigned int gId = getGaussianIndex(optixGetPrimitiveIndex());
    const float hitT = (MoGSurfPrimitive || MoGCustomPrimitive || MoGTracingHitMode != MOGTracingGaussianHit) ? optixGetRayTmax() : computeGHitDistance(gId, optixGetWorldRayOrigin(), optixGetWorldRayDirection(), params);
    if (hitT < ahHitTablePtr[MoGTracingAHMaxNumHitPerSlab - 1].x)
    {
        unsigned int ahNumHits = optixGetPayload_0();

#if MOGTRACING_SAMPLING_MODE
        if (params.renderOpts & MOGRenderDnsHitSampling)
        {
            unsigned int rndSeed = optixGetPayload_3();
            const float sple = rnd(rndSeed);
            optixSetPayload_3(rndSeed);
            const float dns = params.mogDns[gId][0];
            if (dns < sple)
            {
                optixIgnoreIntersection();
                return;
            }
        }
#endif

        float2 ahHit = {hitT, uint_as_float(optixGetPrimitiveIndex())};
#pragma unroll
        for (int i = 0; i < MoGTracingAHMaxNumHitPerSlab; ++i)
        {
            if (ahHit.x < ahHitTablePtr[i].x)
            {
                const float2 swapHit = ahHitTablePtr[i];
                ahHitTablePtr[i] = ahHit;
                ahHit = swapHit;
            }
        }
        if (ahNumHits < MoGTracingAHMaxNumHitPerSlab)
        {
            optixSetPayload_0(ahNumHits + 1);
        }
        // report the last entry
        if (ahHitTablePtr[MoGTracingAHMaxNumHitPerSlab - 1].x != hitT)
        {
            optixIgnoreIntersection();
        }
    }
}
