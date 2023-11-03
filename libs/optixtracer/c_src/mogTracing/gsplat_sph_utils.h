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

template<int deg=3, int numCoeffs=16>
static __device__ float3 computeColorFromSH(const float3&gpos, const SphCoefficients<numCoeffs>& gsphs, const float3& rori)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	const float3 dir = safe_normalize(gpos - rori);
	
	float3 rad = SH_C0 * gsphs[0];
	if (deg > 0)
	{
		const float x = dir.x;
		const float y = dir.y;
		const float z = dir.z;
		rad = rad - SH_C1 * y * gsphs[1] + SH_C1 * z * gsphs[2] - SH_C1 * x * gsphs[3];

		if (deg > 1)
		{
			const float xx = x * x, yy = y * y, zz = z * z;
			const float xy = x * y, yz = y * z, xz = x * z;
			rad = rad +
				SH_C2[0] * xy * gsphs[4] +
				SH_C2[1] * yz * gsphs[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * gsphs[6] +
				SH_C2[3] * xz * gsphs[7] +
				SH_C2[4] * (xx - yy) * gsphs[8];

			if (deg > 2)
			{
				rad = rad +
					SH_C3[0] * y * (3.0f * xx - yy) * gsphs[9] +
					SH_C3[1] * xy * z * gsphs[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * gsphs[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * gsphs[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * gsphs[13] +
					SH_C3[5] * z * (xx - yy) * gsphs[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * gsphs[15];
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

#endif