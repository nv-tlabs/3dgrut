// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#include "../mogTracingBwd/params.h"
#include "../mogTracing/utils.h"
#include "../optix_utils.h"
#include "../ray_data.h"

extern "C"
{
    __constant__ MoGTracingBwdParams params;
}

static constexpr unsigned int MoGTracingAHMaxNumHitPerSlab = 32;

struct RayPayload
{
    unsigned int ahNumHits; // number of valid hits in ahHitTable
    float2 ahHitTable[MoGTracingAHMaxNumHitPerSlab]; // hit data : x = hitT, y = triId
};

static __device__ __inline__ float2 intersectAABB(const OptixAabb& aabb, const float3& rayOri, const float3& rayDir)
{
    const float3 t0 = (make_float3(aabb.minX, aabb.minY, aabb.minZ) - rayOri) / rayDir;
    const float3 t1 = (make_float3(aabb.maxX, aabb.maxY, aabb.maxZ) - rayOri) / rayDir;
    const float3 tmax = maxf3(t0, t1);
    const float3 tmin = minf3(t0, t1);
    return float2{ fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z))), fminf(tmax.x, fminf(tmax.y, tmax.z)) };
}

static __device__ __inline__ void trace(
    RayPayload* p, const float3& rayOri, const float3& rayDir, const float tmin, const float tmax)
{
    const unsigned long long ahHitTablePtr = reinterpret_cast<unsigned long long>(&p->ahHitTable);
    unsigned int ahHitTablePtr0 = ahHitTablePtr >> 32;
    unsigned int ahHitTablePtr1 = ahHitTablePtr & 0x00000000ffffffff;

    // Trace the ray against our scene hierarchy
    optixTrace(params.handle, rayOri, rayDir,
               tmin, // Min intersection distance
               tmax, // Max intersection distance
               0.0f, // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               0, // SBT offset   -- See SBT discussion
               1, // SBT stride   -- See SBT discussion
               0, // missSBTIndex -- See SBT discussion
               reinterpret_cast<unsigned int&>(p->ahNumHits), ahHitTablePtr0, ahHitTablePtr1);
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const float3 rayOri = make_float3(params.rayOri[idx.z][idx.y][idx.x][0], params.rayOri[idx.z][idx.y][idx.x][1],
                                      params.rayOri[idx.z][idx.y][idx.x][2]);

    const float3 rayDir = make_float3(params.rayDir[idx.z][idx.y][idx.x][0], params.rayDir[idx.z][idx.y][idx.x][1],
                                      params.rayDir[idx.z][idx.y][idx.x][2]);

    // ray- aabb intersection to determine number of segments
    const float2 minMaxT = intersectAABB(params.aabb, rayOri, rayDir);
    constexpr float epsT = 1e-9;
    //const float slabSpacing = MoGTracingAHMaxNumHitPerSlab * params.expectedDistanceBetweenHit + epsT;
    const float slabSpacing = params.slabSpacing;
    float startT = fmaxf(0.0f, minMaxT.x - epsT);
    
    float transmit = 1.0f;
    RayPayload p;

    const float3 totalRayRad = make_float3(
        params.rayRad[idx.z][idx.y][idx.x][0],
        params.rayRad[idx.z][idx.y][idx.x][1],
        params.rayRad[idx.z][idx.y][idx.x][2]);

    const float3 rayRadGrd = make_float3(
        params.rayRadGrd[idx.z][idx.y][idx.x][0],
        params.rayRadGrd[idx.z][idx.y][idx.x][1],
        params.rayRadGrd[idx.z][idx.y][idx.x][2]);
    const float rayDnsGrd = params.rayDnsGrd[idx.z][idx.y][idx.x][0];
    const float rayHitGrd = params.rayHitGrd[idx.z][idx.y][idx.x][0];

    float3 accumulatedRayRad = make_float3(0);

    while ((startT <= minMaxT.y) && (transmit > params.minTransmittance))
    {
        p.ahNumHits = 0;
        trace(&p, rayOri, rayDir, startT + epsT, startT + slabSpacing);
        if (p.ahNumHits == 0)
        {
            startT += slabSpacing;
            continue;
        }

        // in case we got more hits than available slots, start the next ray from the last hit
        if (p.ahNumHits == MoGTracingAHMaxNumHitPerSlab)
        {
            startT = p.ahHitTable[MoGTracingAHMaxNumHitPerSlab - 1].x;
        }
        else
        {
            startT += slabSpacing;
        }

        for (int i = 0; (i < p.ahNumHits) && (transmit > params.minTransmittance); i++)
        {
            const uint32_t gId = float_as_uint(p.ahHitTable[i].y) / MOGPrimNumTri;

            const float gdns = params.mogDns[gId][0];
            const float3 gpos = make_float3(params.mogPos[gId][0],params.mogPos[gId][1],params.mogPos[gId][2]);
            const float4 grot = make_float4(params.mogRot[gId][0],params.mogRot[gId][1],params.mogRot[gId][2],params.mogRot[gId][3]);
            const float3 gscl = make_float3(params.mogScl[gId][0],params.mogScl[gId][1],params.mogScl[gId][2]);

            // project ray in the gaussian
            float33 grotMat;
            rotationMatrix(make_float4(grot.x,grot.y,grot.z,grot.w), grotMat);
            const float3 giscl = make_float3(1/gscl.x,1/gscl.y,1/gscl.z);
            const float3 gposc = (rayOri - gpos);
            const float3 gposcr = (gposc*grotMat); 
            const float3 gro =  giscl*gposcr;
            const float3 rayDirR = rayDir*grotMat;
            const float3 grdu = giscl*rayDirR; 
            const float3 grd = safe_normalize(grdu);
            const float3 gcrod = cross(grd, gro);
            const float grayDist = dot(gcrod,gcrod);
            const float gres = expf(-0.5 * grayDist);
            const float galpha = gres * gdns; 
            // const float galpha = fminf(GAlphaMax, gres * gdns);
            // if (galpha < GAlphaMin)
            // {
            //     continue;    
            // }
            const float3 gradu = SH_C0 * make_float3(params.mogSph[gId][0], params.mogSph[gId][1], params.mogSph[gId][2]) + make_float3(0.5);
            const float3 grad = make_float3(
                gradu.x>SHRadMinBound ? gradu.x : expf(gradu.x-SHRadMinBound)*SHRadMinBound,
                gradu.y>SHRadMinBound ? gradu.y : expf(gradu.y-SHRadMinBound)*SHRadMinBound,
                gradu.z>SHRadMinBound ? gradu.z : expf(gradu.z-SHRadMinBound)*SHRadMinBound
            );
            
            const float weight = galpha * transmit;

            // NB : no gradient wrt d_rayDns assert(rayDnsGrd==0)
            
            // gradient computation wrt rayRad

            // >>> rayRadiance = accumulatedRayRad + weigth * rayRad + (1-galpha)*transmit * residualRayRad
            const float3 rayRad = weight * grad;
            accumulatedRayRad += rayRad;
            const float nextTransmit = (1-galpha)*transmit;
            const float3 residualRayRad =
                maxf3((nextTransmit <= params.minTransmittance ? make_float3(0) :
                                                                 (totalRayRad - accumulatedRayRad) / nextTransmit),
                      make_float3(0));

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> rayRad = weight * grad = weight * explu(gsph0 * SH_C0 + 0.5,SHRadMinBound)
            // with explu(x,a) = x if x > a else a*e(x-a)
            // ===> d_rayRad / d_gsph0 =   weight * SH_C0 
            const float shc0w =  SH_C0 * weight;
            atomicAdd(&params.mogSphGrd[gId][0], (gradu.x > SHRadMinBound ? 1 : grad.x) * shc0w * rayRadGrd.x);
            atomicAdd(&params.mogSphGrd[gId][1], (gradu.y > SHRadMinBound ? 1 : grad.y) * shc0w * rayRadGrd.y);
            atomicAdd(&params.mogSphGrd[gId][2], (gradu.z > SHRadMinBound ? 1 : grad.z) * shc0w * rayRadGrd.z);

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> rayRadiance = accumulatedRayRad + galpha * transmit * grad + (1-galpha) * transmit * residualRayRad
            //                  = accumulatedRayRad + gdns * gres * transmit * grad + (1-gdns*gres) * transmit * residualRayRad
            // ===> d_rayRad / d_gdns = gres * transmit * grad - gres * transmit * residualRayRad
            atomicAdd(&params.mogDnsGrd[gId][0], 
                gres * transmit * (grad.x - residualRayRad.x) * rayRadGrd.x +
                gres * transmit * (grad.y - residualRayRad.y) * rayRadGrd.y +
                gres * transmit * (grad.z - residualRayRad.z) * rayRadGrd.z);
            
            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> rayRadiance = accumulatedRayRad + galpha * transmit * grad + (1 - galpha) * transmit * residualRayRad 
            //                  = accumulatedRayRad + gdns * gres * transmit * grad + (1 - gdns * gres) * transmit * residualRayRad
            // ===> d_rayRad / d_gres = gdns * transmit * grad - gdns * transmit * residualRayRad
            const float gresGrd = gdns * transmit * (grad.x - residualRayRad.x) * rayRadGrd.x +
                                  gdns * transmit * (grad.y - residualRayRad.y) * rayRadGrd.y +
                                  gdns * transmit * (grad.z - residualRayRad.z) * rayRadGrd.z;

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> gres = exp(-0.5 * grayDist)        
            // ===> d_gres / d_grayDist = -0.5 * exp(-0.5 * grayDist) 
            //                          = -0.5 * gres
            const float grayDistGrd = -0.5 * gres * gresGrd;

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
            const float3 grdGrd = make_float3(
                gcrodGrd.z * gro.y - gcrodGrd.y * gro.z,
                gcrodGrd.x * gro.z - gcrodGrd.z * gro.x,
                gcrodGrd.y * gro.x - gcrodGrd.x * gro.y
            );
            const float3 groGrd = make_float3(
                gcrodGrd.y * grd.z - gcrodGrd.z * grd.y,
                gcrodGrd.z * grd.x - gcrodGrd.x * grd.z,
                gcrodGrd.x * grd.y - gcrodGrd.y * grd.x
            );

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> gro = (1/gscl)*gposcr
            // ===> d_gro / d_gscl = -gposcr/(gscl*gscl)
            // ===> d_gro / d_gposcr = (1/gscl)
            const float3 gsclGrdGro =
                make_float3((-gposcr.x / (gscl.x * gscl.x)) * groGrd.x, (-gposcr.y / (gscl.y * gscl.y)) * groGrd.y,
                            (-gposcr.z / (gscl.z * gscl.z)) * groGrd.z);
            const float3 gposcrGrd = giscl * groGrd;
            
            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> gposcr = matmul(gposc, grotMat)
            // ===> d_gposcr / d_gposc = matmul_bw_vec(grotMat)
            // ===> d_gposcr / d_grotmat = matmul_bw_mat(gposc)
            const float3 gposcGrd = matmul_bw_vec(grotMat, gposcrGrd);
            const float4 grotGrdPoscr = matmul_bw_quat(gposc, gposcrGrd, grot);
            
            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> gposc = rayOri - gpos
            // ===> d_gposc / d_gpos = -1
            atomicAdd(&params.mogPosGrd[gId][0],-gposcGrd.x); 
            atomicAdd(&params.mogPosGrd[gId][1],-gposcGrd.y);
            atomicAdd(&params.mogPosGrd[gId][2],-gposcGrd.z);

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> grd = safe_normalize(grdu)
            // ===> d_grd / d_grdu = safe_normalize_bw(grd)
            const float3 grduGrd = safe_normalize_bw(grdu, grdGrd);
            
            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> grdu = (1/gscl)*rayDirR
            // ===> d_grdu / d_gscl = -rayDirR/(gscl*gscl)
            // ===> d_grdu / d_rayDirR = (1/gscl)
            atomicAdd(&params.mogSclGrd[gId][0], gsclGrdGro.x + (-rayDirR.x / (gscl.x*gscl.x)) * grduGrd.x); 
            atomicAdd(&params.mogSclGrd[gId][1], gsclGrdGro.y + (-rayDirR.y / (gscl.y*gscl.y)) * grduGrd.y);
            atomicAdd(&params.mogSclGrd[gId][2], gsclGrdGro.z + (-rayDirR.z / (gscl.z*gscl.z)) * grduGrd.z);
            const float3 rayDirRGrd = giscl * grduGrd;

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> rayDirR = matmul(rayDir, grotMat)
            // ===> d_rayDirR / d_grotmat = matmul_bw_mat(rayDir, grotMat)
            const float4 grotGrdRayDirR = matmul_bw_quat(rayDir, rayDirRGrd, grot);
            atomicAdd(&params.mogRotGrd[gId][0], grotGrdPoscr.x + grotGrdRayDirR.x); 
            atomicAdd(&params.mogRotGrd[gId][1], grotGrdPoscr.y + grotGrdRayDirR.y);
            atomicAdd(&params.mogRotGrd[gId][2], grotGrdPoscr.z + grotGrdRayDirR.z);
            atomicAdd(&params.mogRotGrd[gId][3], grotGrdPoscr.w + grotGrdRayDirR.w);

            transmit = nextTransmit;
        }
    }
}

extern "C" __global__ void __miss__ms()
{
    // TODO : fetch background or generate random background (maybe done outside of tracer)
}

extern "C" __global__ void __anyhit__ah()
{
    const unsigned int ahHitTablePtr0 = optixGetPayload_1();
    const unsigned int ahHitTablePtr1 = optixGetPayload_2();
    float2* ahHitTablePtr =
        reinterpret_cast<float2*>(static_cast<unsigned long long>(ahHitTablePtr0) << 32 | ahHitTablePtr1);
    unsigned int ahNumHits = optixGetPayload_0();
    float2 ahHit = { optixGetRayTmax(), uint_as_float(optixGetPrimitiveIndex()) };
    if ((ahNumHits < MoGTracingAHMaxNumHitPerSlab) || (ahHit.x < ahHitTablePtr[MoGTracingAHMaxNumHitPerSlab - 1].x))
    {
        ahNumHits = min(ahNumHits + 1, MoGTracingAHMaxNumHitPerSlab); // increment num hit
        int i = ahNumHits - 1; // assert(ahNumHits-1 is garbage)
        for (; (i > 0) && (ahHit.x < ahHitTablePtr[i - 1].x); --i)
        {
            ahHitTablePtr[i] = ahHitTablePtr[i - 1];
        }
        // assert(i==0 || (ahHit.x >= ahHitTablePtr[i-1].x))
        ahHitTablePtr[i] = ahHit;
        optixSetPayload_0(ahNumHits);
        optixIgnoreIntersection();
    }
}
