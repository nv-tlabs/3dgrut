#pragma once

#include "../math_utils.h"

#ifdef __CUDACC__

using uint32_t = unsigned int;
template<int num>
using SphCoefficients = float3[num];

static constexpr float SHRadMinBound = 0.001f;
// Spherical harmonics coefficients
static constexpr float SH_C0 = 0.28209479177387814f;
static constexpr float SH_C1 = 0.4886025119029199f;
static constexpr float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
static constexpr float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

// This code has been adapated from ::::::::::::::::::::::::::::::::::::::
//
/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

template<int deg=3, typename TParams>
static __device__ float3 computeColorFromSH(
	const float3&gpos, 
	const float3& rori,
	uint32_t gId,
	TParams& params 
)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	float3 rad = SH_C0 * make_float3(params.mogSph[gId][0],params.mogSph[gId][1],params.mogSph[gId][2]);
	if (deg > 0)
	{
		const float3 dir = safe_normalize(gpos - rori);
		
		const float x = dir.x;
		const float y = dir.y;
		const float z = dir.z;
		rad = rad - SH_C1 * y * params.mogSph[gId][1] + SH_C1 * z * params.mogSph[gId][2] - SH_C1 * x * params.mogSph[gId][3];

		if (deg > 1)
		{
			const float xx = x * x, yy = y * y, zz = z * z;
			const float xy = x * y, yz = y * z, xz = x * z;
			rad = rad +
				SH_C2[0] * xy * params.mogSph[gId][4] +
				SH_C2[1] * yz * params.mogSph[gId][5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * params.mogSph[gId][6] +
				SH_C2[3] * xz * params.mogSph[gId][7] +
				SH_C2[4] * (xx - yy) * params.mogSph[gId][8];

			if (deg > 2)
			{
				rad = rad +
					SH_C3[0] * y * (3.0f * xx - yy) * params.mogSph[gId][9] +
					SH_C3[1] * xy * z * params.mogSph[gId][10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * params.mogSph[gId][11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * params.mogSph[gId][12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * params.mogSph[gId][13] +
					SH_C3[5] * z * (xx - yy) * params.mogSph[gId][14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * params.mogSph[gId][15];
			}
		}
	}
	rad += 0.5f;
	return make_float3(
                rad.x>SHRadMinBound ? rad.x : expf(rad.x-SHRadMinBound)*SHRadMinBound,
                rad.y>SHRadMinBound ? rad.y : expf(rad.y-SHRadMinBound)*SHRadMinBound,
                rad.z>SHRadMinBound ? rad.z : expf(rad.z-SHRadMinBound)*SHRadMinBound
    );
}

// template <int deg=3, numCoeffs=16, typename TParams>
// __device__ float3 computeColorFromSHBwd(
// 	const float3&gpos, 
// 	const float3& rori,
// 	const float3& rayRadGrd,
// 	uint32_t gid,
// 	TParams& params 
// 	float3& dL_grad, 
// 	float3& dL_gpos, 
// 	float3& dL_gsph)
// {
// 	const float3 dir = safe_normalize(gpos - rori);
	
// 	float3 rad = SH_C0 * params.mogSph[gId][0];
	
// 	float3 dL_dRGB = rayRadGrd;
// 	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
// 	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
// 	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

// 	glm::vec3 dRGBdx(0, 0, 0);
// 	glm::vec3 dRGBdy(0, 0, 0);
// 	glm::vec3 dRGBdz(0, 0, 0);
// 	float x = dir.x;
// 	float y = dir.y;
// 	float z = dir.z;

// 	// Target location for this Gaussian to write SH gradients to
// 	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

// 	// No tricks here, just high school-level calculus.
// 	float dRGBdsh0 = SH_C0;
// 	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
// 	if (deg > 0)
// 	{
// 		float dRGBdsh1 = -SH_C1 * y;
// 		float dRGBdsh2 = SH_C1 * z;
// 		float dRGBdsh3 = -SH_C1 * x;
// 		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
// 		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
// 		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

// 		dRGBdx = -SH_C1 * sh[3];
// 		dRGBdy = -SH_C1 * sh[1];
// 		dRGBdz = SH_C1 * sh[2];

// 		if (deg > 1)
// 		{
// 			float xx = x * x, yy = y * y, zz = z * z;
// 			float xy = x * y, yz = y * z, xz = x * z;

// 			float dRGBdsh4 = SH_C2[0] * xy;
// 			float dRGBdsh5 = SH_C2[1] * yz;
// 			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
// 			float dRGBdsh7 = SH_C2[3] * xz;
// 			float dRGBdsh8 = SH_C2[4] * (xx - yy);
// 			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
// 			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
// 			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
// 			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
// 			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

// 			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
// 			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
// 			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

// 			if (deg > 2)
// 			{
// 				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
// 				float dRGBdsh10 = SH_C3[1] * xy * z;
// 				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
// 				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
// 				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
// 				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
// 				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
// 				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
// 				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
// 				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
// 				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
// 				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
// 				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
// 				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

// 				dRGBdx += (
// 					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
// 					SH_C3[1] * sh[10] * yz +
// 					SH_C3[2] * sh[11] * -2.f * xy +
// 					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
// 					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
// 					SH_C3[5] * sh[14] * 2.f * xz +
// 					SH_C3[6] * sh[15] * 3.f * (xx - yy));

// 				dRGBdy += (
// 					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
// 					SH_C3[1] * sh[10] * xz +
// 					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
// 					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
// 					SH_C3[4] * sh[13] * -2.f * xy +
// 					SH_C3[5] * sh[14] * -2.f * yz +
// 					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

// 				dRGBdz += (
// 					SH_C3[1] * sh[10] * xy +
// 					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
// 					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
// 					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
// 					SH_C3[5] * sh[14] * (xx - yy));
// 			}
// 		}
// 	}

// 	// The view direction is an input to the computation. View direction
// 	// is influenced by the Gaussian's mean, so SHs gradients
// 	// must propagate back into 3D position.
// 	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

// 	// Account for normalization of direction
// 	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

// 	// Gradients of loss w.r.t. Gaussian means, but only the portion 
// 	// that is caused because the mean affects the view-dependent color.
// 	// Additional mean gradient is accumulated in below methods.
// 	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
// }

#endif