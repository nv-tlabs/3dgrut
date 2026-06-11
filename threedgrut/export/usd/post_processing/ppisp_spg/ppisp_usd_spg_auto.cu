// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

// PPISP (Physically Plausible Image Signal Processing) SPG CUDA shader.
//
// Automatic-parameter variant: exposure and color latents are read from
// the 1x9 ControllerParams buffer produced by ppisp_controller.cu.

static __device__ __forceinline__ float2 makeFloat2(float x, float y) {
  const float2 out = {x, y};
  return out;
}

static __device__ __forceinline__ float3 makeFloat3(float x, float y, float z) {
  const float3 out = {x, y, z};
  return out;
}

static __device__ __forceinline__ uchar4 makeUchar4(unsigned char x,
                                                    unsigned char y,
                                                    unsigned char z,
                                                    unsigned char w) {
  const uchar4 out = {x, y, z, w};
  return out;
}

static __device__ __forceinline__ float2 mul2x2(float m00, float m01, float m10,
                                                float m11, float2 v) {
  return makeFloat2(m00 * v.x + m01 * v.y, m10 * v.x + m11 * v.y);
}

struct Mat3 {
  float m[3][3];
};

static __device__ __forceinline__ float3 row(const Mat3 &a, int r) {
  return makeFloat3(a.m[r][0], a.m[r][1], a.m[r][2]);
}

static __device__ __forceinline__ float3 cross3(float3 a, float3 b) {
  return makeFloat3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x);
}

static __device__ __forceinline__ float dot3(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __device__ __forceinline__ Mat3 matmul3(const Mat3 &a, const Mat3 &b) {
  Mat3 out;
#pragma unroll
  for (int r = 0; r < 3; ++r) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      out.m[r][c] =
          a.m[r][0] * b.m[0][c] + a.m[r][1] * b.m[1][c] + a.m[r][2] * b.m[2][c];
    }
  }
  return out;
}

static __device__ __forceinline__ float3 matvec3(const Mat3 &a, float3 v) {
  return makeFloat3(a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
                    a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
                    a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z);
}

static __device__ __forceinline__ Mat3
computeHomography(float2 colorLatentBlue, float2 colorLatentRed,
                  float2 colorLatentGreen, float2 colorLatentNeutral) {
  const float2 bd =
      mul2x2(0.0480542f, -0.0043631f, -0.0043631f, 0.0481283f, colorLatentBlue);
  const float2 rd =
      mul2x2(0.0580570f, -0.0179872f, -0.0179872f, 0.0431061f, colorLatentRed);
  const float2 gd = mul2x2(0.0433336f, -0.0180537f, -0.0180537f, 0.0580500f,
                           colorLatentGreen);
  const float2 nd = mul2x2(0.0128369f, -0.0034654f, -0.0034654f, 0.0128158f,
                           colorLatentNeutral);

  const float3 tB = makeFloat3(bd.x, bd.y, 1.0f);
  const float3 tR = makeFloat3(1.0f + rd.x, rd.y, 1.0f);
  const float3 tG = makeFloat3(gd.x, 1.0f + gd.y, 1.0f);
  const float3 tGray = makeFloat3(1.0f / 3.0f + nd.x, 1.0f / 3.0f + nd.y, 1.0f);

  Mat3 t = {{
      {tB.x, tR.x, tG.x},
      {tB.y, tR.y, tG.y},
      {tB.z, tR.z, tG.z},
  }};
  Mat3 skew = {{
      {0.0f, -tGray.z, tGray.y},
      {tGray.z, 0.0f, -tGray.x},
      {-tGray.y, tGray.x, 0.0f},
  }};
  const Mat3 m = matmul3(skew, t);

  float3 lam = cross3(row(m, 0), row(m, 1));
  if (dot3(lam, lam) < 1.0e-20f) {
    lam = cross3(row(m, 0), row(m, 2));
    if (dot3(lam, lam) < 1.0e-20f) {
      lam = cross3(row(m, 1), row(m, 2));
    }
  }

  Mat3 d = {{
      {lam.x, 0.0f, 0.0f},
      {0.0f, lam.y, 0.0f},
      {0.0f, 0.0f, lam.z},
  }};
  Mat3 sinv = {{
      {-1.0f, -1.0f, 1.0f},
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
  }};

  Mat3 h = matmul3(matmul3(t, d), sinv);
  const float s = h.m[2][2];
  if (fabsf(s) > 1.0e-20f) {
    const float inv = 1.0f / s;
#pragma unroll
    for (int r = 0; r < 3; ++r) {
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        h.m[r][c] *= inv;
      }
    }
  }
  return h;
}

static __device__ __forceinline__ float
applyVignetting(float value, float2 uv, float2 opticalCenter, float alpha1,
                float alpha2, float alpha3) {
  const float2 delta =
      makeFloat2(uv.x - opticalCenter.x, uv.y - opticalCenter.y);
  const float r2 = delta.x * delta.x + delta.y * delta.y;
  const float r4 = r2 * r2;
  const float r6 = r4 * r2;
  const float falloff =
      fminf(fmaxf(1.0f + alpha1 * r2 + alpha2 * r4 + alpha3 * r6, 0.0f), 1.0f);
  return value * falloff;
}

static __device__ __forceinline__ float boundedSoftplus(float raw,
                                                        float minValue) {
  return minValue + log1pf(expf(raw));
}

static __device__ __forceinline__ float sigmoid(float raw) {
  return 1.0f / (1.0f + expf(-raw));
}

static __device__ __forceinline__ float applyCRF(float x, float toeRaw,
                                                 float shoulderRaw,
                                                 float gammaRaw,
                                                 float centerRaw) {
  x = fminf(fmaxf(x, 0.0f), 1.0f);

  const float toe = boundedSoftplus(toeRaw, 0.3f);
  const float shoulder = boundedSoftplus(shoulderRaw, 0.3f);
  const float gamma = boundedSoftplus(gammaRaw, 0.1f);
  const float eps = 1.0e-6f;
  const float center = fminf(fmaxf(sigmoid(centerRaw), eps), 1.0f - eps);

  const float lerpVal = fmaxf((shoulder - toe) * center + toe, eps);
  const float a = (shoulder * center) / lerpVal;
  const float b = 1.0f - a;

  const float below = a * powf(x / center, toe);
  const float above = 1.0f - b * powf((1.0f - x) / (1.0f - center), shoulder);
  return powf(fmaxf(x <= center ? below : above, 0.0f), gamma);
}

// Tile-local UV for a tiled RenderProduct atlas (see ppisp_usd_spg.cu).
// Reduces exactly to the full-image UV when tileCountX == tileCountY == 1.
static __device__ __forceinline__ float2 computeTileUv(int x, int y, int width,
                                                       int height,
                                                       int tileCountX,
                                                       int tileCountY) {
  const int tcx = max(tileCountX, 1);
  const int tcy = max(tileCountY, 1);
  const int tileW = max(1, width / tcx);
  const int tileH = max(1, height / tcy);
  int tx = x / tileW;
  int ty = y / tileH;
  tx = min(tx, tcx - 1);
  ty = min(ty, tcy - 1);
  const int localX = x - tx * tileW;
  const int localY = y - ty * tileH;
  const float maxRes = fmaxf(float(tileW), float(tileH));
  return makeFloat2((float(localX) + 0.5f - float(tileW) * 0.5f) / maxRes,
                    (float(localY) + 0.5f - float(tileH) * 0.5f) / maxRes);
}

// Linear tile index (ty * tileCountX + tx) selecting which per-tile
// ControllerParams group this pixel reads. Zero for an untiled product.
static __device__ __forceinline__ int computeTileIndex(int x, int y, int width,
                                                       int height,
                                                       int tileCountX,
                                                       int tileCountY) {
  const int tcx = max(tileCountX, 1);
  const int tcy = max(tileCountY, 1);
  const int tileW = max(1, width / tcx);
  const int tileH = max(1, height / tcy);
  int tx = x / tileW;
  int ty = y / tileH;
  tx = min(tx, tcx - 1);
  ty = min(ty, tcy - 1);
  return ty * tcx + tx;
}

static __device__ __forceinline__ float3 applyPPISPColor(
    float3 rgb, float2 uv, float responsivity, float exposureOffset,
    float2 vignettingCenterR, float vignettingAlpha1R, float vignettingAlpha2R,
    float vignettingAlpha3R, float2 vignettingCenterG, float vignettingAlpha1G,
    float vignettingAlpha2G, float vignettingAlpha3G, float2 vignettingCenterB,
    float vignettingAlpha1B, float vignettingAlpha2B, float vignettingAlpha3B,
    float2 colorLatentBlue, float2 colorLatentRed, float2 colorLatentGreen,
    float2 colorLatentNeutral, float crfToeR, float crfShoulderR,
    float crfGammaR, float crfCenterR, float crfToeG, float crfShoulderG,
    float crfGammaG, float crfCenterG, float crfToeB, float crfShoulderB,
    float crfGammaB, float crfCenterB) {
  rgb.x *= responsivity;
  rgb.y *= responsivity;
  rgb.z *= responsivity;

  const float exposureScale = exp2f(exposureOffset);
  rgb.x *= exposureScale;
  rgb.y *= exposureScale;
  rgb.z *= exposureScale;

  rgb.x = applyVignetting(rgb.x, uv, vignettingCenterR, vignettingAlpha1R,
                          vignettingAlpha2R, vignettingAlpha3R);
  rgb.y = applyVignetting(rgb.y, uv, vignettingCenterG, vignettingAlpha1G,
                          vignettingAlpha2G, vignettingAlpha3G);
  rgb.z = applyVignetting(rgb.z, uv, vignettingCenterB, vignettingAlpha1B,
                          vignettingAlpha2B, vignettingAlpha3B);

  const Mat3 h = computeHomography(colorLatentBlue, colorLatentRed,
                                   colorLatentGreen, colorLatentNeutral);
  const float intensity = rgb.x + rgb.y + rgb.z;
  float3 rgi = makeFloat3(rgb.x, rgb.y, intensity);
  rgi = matvec3(h, rgi);
  const float scale = intensity / (rgi.z + 1.0e-5f);
  rgi.x *= scale;
  rgi.y *= scale;
  rgi.z *= scale;
  rgb = makeFloat3(rgi.x, rgi.y, rgi.z - rgi.x - rgi.y);

  rgb.x = applyCRF(rgb.x, crfToeR, crfShoulderR, crfGammaR, crfCenterR);
  rgb.y = applyCRF(rgb.y, crfToeG, crfShoulderG, crfGammaG, crfCenterG);
  rgb.z = applyCRF(rgb.z, crfToeB, crfShoulderB, crfGammaB, crfCenterB);
  return rgb;
}

static __device__ __forceinline__ unsigned char toU8(float x) {
  return static_cast<unsigned char>(fminf(fmaxf(x, 0.0f), 1.0f) * 255.0f);
}

extern "C" __global__ void ppispProcessAuto(
    int width, int height, int tileCountX, int tileCountY,
    cudaTextureObject_t inHdrColor, const float *__restrict__ controllerParams,
    const float *__restrict__ params, cudaSurfaceObject_t outPPISPColor) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  // ControllerParams holds 9 floats per tile, laid out tile-major. Each
  // pixel reads the group for the tile it falls in (tile 0 when untiled).
  const int tile =
      computeTileIndex(x, y, width, height, tileCountX, tileCountY);
  const float *__restrict__ cp = controllerParams + tile * 9;
  const float2 uv = computeTileUv(x, y, width, height, tileCountX, tileCountY);
  const float4 pixel = tex2D<float4>(inHdrColor, x, y);
  const float exposureOffset = cp[0];
  const float2 colorLatentBlue = makeFloat2(cp[1], cp[2]);
  const float2 colorLatentRed = makeFloat2(cp[3], cp[4]);
  const float2 colorLatentGreen = makeFloat2(cp[5], cp[6]);
  const float2 colorLatentNeutral = makeFloat2(cp[7], cp[8]);
  const float3 rgb = applyPPISPColor(
      makeFloat3(pixel.x, pixel.y, pixel.z), uv, params[0], exposureOffset,
      makeFloat2(params[1], params[2]), params[3], params[4], params[5],
      makeFloat2(params[6], params[7]), params[8], params[9], params[10],
      makeFloat2(params[11], params[12]), params[13], params[14], params[15],
      colorLatentBlue, colorLatentRed, colorLatentGreen, colorLatentNeutral,
      params[16], params[17], params[18], params[19], params[20], params[21],
      params[22], params[23], params[24], params[25], params[26], params[27]);

  const uchar4 out = makeUchar4(toU8(rgb.x), toU8(rgb.y), toU8(rgb.z), 255);
  surf2Dwrite<uchar4>(out, outPPISPColor, x * int(sizeof(uchar4)), y);
}
