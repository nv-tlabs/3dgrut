// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

// PPISP (Physically Plausible ISP) SPG Shader — controller-aware variant.
//
// Identical to ppisp_usd_spg.slang in maths. The only difference is that
// `exposureOffset` and the eight colour latents come from a 1x9 single-
// channel float texture produced by the per-camera PPISP controller
// shader (ppisp_controller_<cam>.slang), instead of from
// time-sampled USD attributes.
//
// Texel layout of the controller output (matches ppisp_controller.slang):
//   (0, 0):       exposureOffset
//   (1..2, 0):    colorLatentBlue.xy
//   (3..4, 0):    colorLatentRed.xy
//   (5..6, 0):    colorLatentGreen.xy
//   (7..8, 0):    colorLatentNeutral.xy

struct PPISPDynParams
{
    float2 vignettingCenterR;
    float vignettingAlpha1R;
    float vignettingAlpha2R;
    float vignettingAlpha3R;

    float2 vignettingCenterG;
    float vignettingAlpha1G;
    float vignettingAlpha2G;
    float vignettingAlpha3G;

    float2 vignettingCenterB;
    float vignettingAlpha1B;
    float vignettingAlpha2B;
    float vignettingAlpha3B;

    float crfToeR;
    float crfShoulderR;
    float crfGammaR;
    float crfCenterR;

    float crfToeG;
    float crfShoulderG;
    float crfGammaG;
    float crfCenterG;

    float crfToeB;
    float crfShoulderB;
    float crfGammaB;
    float crfCenterB;
};

[[vk::binding(0, 1)]] ParameterBlock<PPISPDynParams> g_Params;
[[vk::binding(1, 1)]] Texture2D<float4>  g_InTex;
[[vk::binding(2, 1)]] Texture2D<float>   g_ControllerOut;
[[vk::binding(3, 1)]] RWTexture2D<float4> g_OutTex;

static const float2x2 ZCA_BLUE    = float2x2( 0.0480542, -0.0043631, -0.0043631,  0.0481283);
static const float2x2 ZCA_RED     = float2x2( 0.0580570, -0.0179872, -0.0179872,  0.0431061);
static const float2x2 ZCA_GREEN   = float2x2( 0.0433336, -0.0180537, -0.0180537,  0.0580500);
static const float2x2 ZCA_NEUTRAL = float2x2( 0.0128369, -0.0034654, -0.0034654,  0.0128158);

float3x3 computeHomography(float2 bLat, float2 rLat, float2 gLat, float2 nLat)
{
    float2 bd = mul(ZCA_BLUE,    bLat);
    float2 rd = mul(ZCA_RED,     rLat);
    float2 gd = mul(ZCA_GREEN,   gLat);
    float2 nd = mul(ZCA_NEUTRAL, nLat);

    float3 tB    = float3(0.0 + bd.x, 0.0 + bd.y, 1.0);
    float3 tR    = float3(1.0 + rd.x, 0.0 + rd.y, 1.0);
    float3 tG    = float3(0.0 + gd.x, 1.0 + gd.y, 1.0);
    float3 tGray = float3(1.0 / 3.0 + nd.x, 1.0 / 3.0 + nd.y, 1.0);

    float3x3 T = float3x3(tB.x, tR.x, tG.x,
                          tB.y, tR.y, tG.y,
                          tB.z, tR.z, tG.z);

    float3x3 skew = float3x3(0.0,     -tGray.z,  tGray.y,
                             tGray.z,  0.0,     -tGray.x,
                            -tGray.y,  tGray.x,  0.0);

    float3x3 M = mul(skew, T);

    float3 r0 = M[0];
    float3 r1 = M[1];
    float3 r2 = M[2];

    float3 lam = cross(r0, r1);
    if (dot(lam, lam) < 1.0e-20)
    {
        lam = cross(r0, r2);
        if (dot(lam, lam) < 1.0e-20)
            lam = cross(r1, r2);
    }

    float3x3 Sinv = float3x3(-1.0, -1.0, 1.0,
                              1.0,  0.0, 0.0,
                              0.0,  1.0, 0.0);

    float3x3 D = float3x3(lam.x, 0.0,   0.0,
                          0.0,   lam.y, 0.0,
                          0.0,   0.0,   lam.z);

    float3x3 H = mul(mul(T, D), Sinv);

    float s = H[2][2];
    if (abs(s) > 1.0e-20)
        H = H * (1.0 / s);

    return H;
}

float applyVignetting(float value, float2 uv, float2 opticalCenter, float a1, float a2, float a3)
{
    float2 d = uv - opticalCenter;
    float r2 = dot(d, d);

    float falloff = 1.0;
    float r2Pow = r2;
    falloff += a1 * r2Pow;
    r2Pow *= r2;
    falloff += a2 * r2Pow;
    r2Pow *= r2;
    falloff += a3 * r2Pow;

    return value * clamp(falloff, 0.0, 1.0);
}

float boundedSoftplus(float raw, float minValue) { return minValue + log(1.0 + exp(raw)); }
float sigmoidF(float raw) { return 1.0 / (1.0 + exp(-raw)); }

float applyCRF(float x, float toeRaw, float shoulderRaw, float gammaRaw, float centerRaw)
{
    x = clamp(x, 0.0, 1.0);
    float toe      = boundedSoftplus(toeRaw, 0.3);
    float shoulder = boundedSoftplus(shoulderRaw, 0.3);
    float gamma    = boundedSoftplus(gammaRaw, 0.1);
    float center   = sigmoidF(centerRaw);

    float lerpVal = (shoulder - toe) * center + toe;
    float a = (shoulder * center) / lerpVal;
    float b = 1.0 - a;

    float y;
    if (x <= center)
        y = a * pow(x / center, toe);
    else
        y = 1.0 - b * pow((1.0 - x) / (1.0 - center), shoulder);
    return pow(max(0.0, y), gamma);
}

float3 applyColorCorrection(float3 rgb, float3x3 H)
{
    float intensity = rgb.x + rgb.y + rgb.z;
    float3 rgi = float3(rgb.x, rgb.y, intensity);
    rgi = mul(H, rgi);
    rgi = rgi * (intensity / (rgi.z + 1.0e-5));
    return float3(rgi.x, rgi.y, rgi.z - rgi.x - rgi.y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ppispProcessDyn(uint3 tid : SV_DispatchThreadID)
{
    uint w = 0, h = 0;
    g_InTex.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h)
        return;

    float4 pixel = g_InTex.Load(int3(tid.xy, 0));
    float3 rgb = pixel.rgb;

    float maxRes = max(float(w), float(h));
    float2 uv = float2(tid.x + 0.5 - float(w) * 0.5,
                       tid.y + 0.5 - float(h) * 0.5) / maxRes;

    // Read controller output (1x9 single-channel float texture).
    float exposureOffset = g_ControllerOut.Load(int3(0, 0, 0));
    float2 colorLatentBlue    = float2(g_ControllerOut.Load(int3(1, 0, 0)),
                                       g_ControllerOut.Load(int3(2, 0, 0)));
    float2 colorLatentRed     = float2(g_ControllerOut.Load(int3(3, 0, 0)),
                                       g_ControllerOut.Load(int3(4, 0, 0)));
    float2 colorLatentGreen   = float2(g_ControllerOut.Load(int3(5, 0, 0)),
                                       g_ControllerOut.Load(int3(6, 0, 0)));
    float2 colorLatentNeutral = float2(g_ControllerOut.Load(int3(7, 0, 0)),
                                       g_ControllerOut.Load(int3(8, 0, 0)));

    rgb = rgb * exp2(exposureOffset);

    rgb.r = applyVignetting(rgb.r, uv, g_Params.vignettingCenterR,
                            g_Params.vignettingAlpha1R, g_Params.vignettingAlpha2R, g_Params.vignettingAlpha3R);
    rgb.g = applyVignetting(rgb.g, uv, g_Params.vignettingCenterG,
                            g_Params.vignettingAlpha1G, g_Params.vignettingAlpha2G, g_Params.vignettingAlpha3G);
    rgb.b = applyVignetting(rgb.b, uv, g_Params.vignettingCenterB,
                            g_Params.vignettingAlpha1B, g_Params.vignettingAlpha2B, g_Params.vignettingAlpha3B);

    float3x3 H = computeHomography(colorLatentBlue, colorLatentRed,
                                   colorLatentGreen, colorLatentNeutral);
    rgb = applyColorCorrection(rgb, H);

    rgb.r = applyCRF(rgb.r, g_Params.crfToeR, g_Params.crfShoulderR, g_Params.crfGammaR, g_Params.crfCenterR);
    rgb.g = applyCRF(rgb.g, g_Params.crfToeG, g_Params.crfShoulderG, g_Params.crfGammaG, g_Params.crfCenterG);
    rgb.b = applyCRF(rgb.b, g_Params.crfToeB, g_Params.crfShoulderB, g_Params.crfGammaB, g_Params.crfCenterB);

    g_OutTex[tid.xy] = float4(saturate(rgb), 1.0);
}
