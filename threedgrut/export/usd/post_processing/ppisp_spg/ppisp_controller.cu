// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

// PPISP Controller SPG Kernel Template (CUDA, embedded-weights variant).
//
// The Python exporter generates one per-camera CUDA source from this
// template by replacing __PPISP_CONTROLLER_EMBEDDED_WEIGHTS__ with a
// file-scope device array containing the trained controller weights.
// The exported graph uses two CUDA nodes backed by that generated
// source: controllerPoolProcess runs the CNN + adaptive average pool
// with one CUDA block per 5x5 output cell, then controllerProcess runs
// the small MLP and writes 9 floats (1 exposure offset + 8 color latents)
// into the ControllerParams AOV.
//
// Architecture mirrors ppisp._PPISPController (default config); the
// authoring-side mirror is locked by
// threedgrut/export/usd/writers/ppisp_controller_weights.py:
//
//   Conv1x1(3->16, +bias)
//   MaxPool 3x3 stride 3
//   ReLU
//   Conv1x1(16->32, +bias)
//   ReLU
//   Conv1x1(32->64, +bias)
//   AdaptiveAvgPool2d((5,5))
//   Flatten -> 1600
//   concat prior_exposure -> 1601
//   MLP: 1601 -> 128 -> 128 -> 128, ReLU after each hidden layer
//   exposure_head: 128 -> 1
//   color_head:    128 -> 8
//
// Output buffer (9 floats, flat row-major from cuda.empty({1, 9})):
//   [0]    : exposureOffset
//   [1..8] : color latents
//        [colorBlue.x, colorBlue.y,
//         colorRed.x,  colorRed.y,
//         colorGreen.x, colorGreen.y,
//         colorNeutral.x, colorNeutral.y]

// ---------------------------------------------------------------------------
// Architecture sizes (must match ``_PPISPController`` defaults and the
// ControllerArchitectureSpec authored on the Python side).
// ---------------------------------------------------------------------------
static const int CNN_FEATURE_DIM = 64;
static const int CNN_FEATURE_CHUNK = 16;
static const int POOL_GRID_H = 5;
static const int POOL_GRID_W = 5;
static const int POOL_CELL_COUNT = POOL_GRID_H * POOL_GRID_W;          //   25
static const int POOL_FEATURE_LEN = POOL_CELL_COUNT * CNN_FEATURE_DIM; // 1600
static const int MLP_INPUT_DIM = POOL_FEATURE_LEN + 1;                 // 1601
static const int MLP_HIDDEN_DIM = 128;
static const int COLOR_PARAMS_PER_FRAME = 8;
static const int INPUT_DOWNSAMPLING = 3;
static const int POOL_THREAD_GROUP_SIZE = 256;
static const int MLP_THREAD_GROUP_SIZE = 128;
// ControllerParams floats emitted per tile: 1 exposure offset + 8 color
// latents. For a tiled RenderProduct the controller produces one such
// group per tile (tile-major), so the auto-PPISP shader can select the
// group matching the tile a pixel falls in.
static const int OUT_PARAMS_PER_TILE = 1 + COLOR_PARAMS_PER_FRAME;

// ---------------------------------------------------------------------------
// Weight buffer offsets. The Python writer in
// ``weights.py`` flattens trained weights into a single
// float buffer in this exact layout; ``TOTAL_WEIGHTS`` must match
// ``EXPECTED_CONTROLLER_WEIGHTS_LEN`` on the Python side (= 241,961).
// ---------------------------------------------------------------------------
static const int OFF_CONV1_W = 0;                     //  16 * 3      =  48
static const int OFF_CONV1_B = OFF_CONV1_W + 16 * 3;  // +16          =  64
static const int OFF_CONV2_W = OFF_CONV1_B + 16;      // +32 * 16     = 576
static const int OFF_CONV2_B = OFF_CONV2_W + 32 * 16; // +32          = 608
static const int OFF_CONV3_W = OFF_CONV2_B + 32;      // +64 * 32     = 2656
static const int OFF_CONV3_B = OFF_CONV3_W + 64 * 32; // +64          = 2720
static const int OFF_TRUNK0_W = OFF_CONV3_B + 64;     // +128 * 1601  = 207648
static const int OFF_TRUNK0_B = OFF_TRUNK0_W + 128 * MLP_INPUT_DIM;
static const int OFF_TRUNK1_W = OFF_TRUNK0_B + 128;
static const int OFF_TRUNK1_B = OFF_TRUNK1_W + 128 * 128;
static const int OFF_TRUNK2_W = OFF_TRUNK1_B + 128;
static const int OFF_TRUNK2_B = OFF_TRUNK2_W + 128 * 128;
static const int OFF_EXP_W = OFF_TRUNK2_B + 128;
static const int OFF_EXP_B = OFF_EXP_W + 128;
static const int OFF_COL_W = OFF_EXP_B + 1;
static const int OFF_COL_B = OFF_COL_W + 8 * 128;
static const int TOTAL_WEIGHTS = OFF_COL_B + 8;
// __PPISP_CONTROLLER_EMBEDDED_WEIGHTS__

// ---------------------------------------------------------------------------
// Per-pixel CNN building blocks.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void
conv1Forward(float r, float g, float b, const float *__restrict__ weights,
             float feat[16]) {
#pragma unroll
  for (int o = 0; o < 16; ++o) {
    float v = weights[OFF_CONV1_B + o];
    v += r * weights[OFF_CONV1_W + o * 3 + 0];
    v += g * weights[OFF_CONV1_W + o * 3 + 1];
    v += b * weights[OFF_CONV1_W + o * 3 + 2];
    feat[o] = v;
  }
}

static __device__ __forceinline__ void
conv2Forward(const float fin[16], const float *__restrict__ weights,
             float fout[32]) {
#pragma unroll
  for (int o = 0; o < 32; ++o) {
    float v = weights[OFF_CONV2_B + o];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      v += fin[i] * weights[OFF_CONV2_W + o * 16 + i];
    }
    fout[o] = v;
  }
}

static __device__ __forceinline__ void
conv3ForwardChunk(const float fin[32], const float *__restrict__ weights,
                  int firstChannel, float fout[CNN_FEATURE_CHUNK]) {
#pragma unroll
  for (int k = 0; k < CNN_FEATURE_CHUNK; ++k) {
    const int o = firstChannel + k;
    float v = weights[OFF_CONV3_B + o];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      v += fin[i] * weights[OFF_CONV3_W + o * 32 + i];
    }
    fout[k] = v;
  }
}

static __device__ __forceinline__ void cnnForwardAtDownsampledPixelChunk(
    int ox, int oy, int tileW, int tileH, int inW, int inH, int dx, int dy,
    float responsivity, cudaTextureObject_t inHdrColor,
    const float *__restrict__ weights, int firstChannel,
    float featChunk[CNN_FEATURE_CHUNK]) {
  // (dx, dy) are tile-local downsampled coordinates; map them back to
  // absolute atlas pixels within this tile's [ox, ox+tileW) x
  // [oy, oy+tileH) sub-region (also clamped to the atlas bounds).
  const int x0 = ox + dx * INPUT_DOWNSAMPLING;
  const int y0 = oy + dy * INPUT_DOWNSAMPLING;
  const int x1 = min(min(x0 + INPUT_DOWNSAMPLING, ox + tileW), inW);
  const int y1 = min(min(y0 + INPUT_DOWNSAMPLING, oy + tileH), inH);

  float pooled[16];
#pragma unroll
  for (int c = 0; c < 16; ++c) {
    pooled[c] = -3.402823e+38f;
  }

  for (int yy = y0; yy < y1; ++yy) {
    for (int xx = x0; xx < x1; ++xx) {
      float4 sample = tex2D<float4>(inHdrColor, xx, yy);
      // Match the image PPISP shader: scale HDR radiance by the
      // achromatic responsivity before feature extraction so controller
      // predictions see the same signal PPISP will process.
      float conv1Out[16];
      conv1Forward(sample.x * responsivity, sample.y * responsivity,
                   sample.z * responsivity, weights, conv1Out);
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        pooled[c] = fmaxf(pooled[c], conv1Out[c]);
      }
    }
  }

#pragma unroll
  for (int c = 0; c < 16; ++c) {
    pooled[c] = fmaxf(0.0f, pooled[c]);
  }

  float feat32[32];
  conv2Forward(pooled, weights, feat32);
#pragma unroll
  for (int c = 0; c < 32; ++c) {
    feat32[c] = fmaxf(0.0f, feat32[c]);
  }

  conv3ForwardChunk(feat32, weights, firstChannel, featChunk);
}

// ---------------------------------------------------------------------------
// Pool entry point. GridDim.x is POOL_CELL_COUNT. Each block computes one
// adaptive-pool cell and writes the channel-major flattened 64x5x5 tensor to
// outControllerFeatures.
// ---------------------------------------------------------------------------
extern "C" __global__ void
controllerPoolProcess(int inW, int inH, int tileCountX, int tileCountY,
                      float responsivity, cudaTextureObject_t inHdrColor,
                      float *__restrict__ outControllerFeatures) {
  const float *__restrict__ weights = kControllerWeights;
  __shared__ float gsReduce[POOL_THREAD_GROUP_SIZE];

  const int tid = (int)threadIdx.x;
  const int cell = (int)blockIdx.x;
  if (cell >= POOL_CELL_COUNT) {
    return;
  }

  // gridDim.y == tileCountX * tileCountY: one column of blocks per tile.
  // Each tile pools its own [ox, ox+tileW) x [oy, oy+tileH) sub-region of
  // the atlas, writing its 1600 features at base tile * POOL_FEATURE_LEN.
  // For an untiled product (1x1) this is the original whole-image pool.
  const int tcx = max(tileCountX, 1);
  const int tcy = max(tileCountY, 1);
  const int tile = (int)blockIdx.y;
  if (tile >= tcx * tcy) {
    return;
  }
  const int tileW = max(1, inW / tcx);
  const int tileH = max(1, inH / tcy);
  const int ox = (tile % tcx) * tileW;
  const int oy = (tile / tcx) * tileH;

  const int dsW = max(1, tileW / INPUT_DOWNSAMPLING);
  const int dsH = max(1, tileH / INPUT_DOWNSAMPLING);

  // Layout note: PyTorch's nn.Flatten on the [N, C, H, W] CNN output
  // produces a *channel-major* flat layout -- feat[c * H*W + h*W + w].
  // The trunk0 weight matrix was trained against that layout, so
  // outControllerFeatures MUST be stored channel-major as well, i.e.
  //     outControllerFeatures[tile * POOL_FEATURE_LEN + c * POOL_CELL_COUNT +
  //     cell].
  // (cell-major would silently permute every controller output.)
  const int gy = cell / POOL_GRID_W;
  const int gx = cell % POOL_GRID_W;

  int hStart = (gy * dsH) / POOL_GRID_H;
  int hEnd = ((gy + 1) * dsH + POOL_GRID_H - 1) / POOL_GRID_H;
  int wStart = (gx * dsW) / POOL_GRID_W;
  int wEnd = ((gx + 1) * dsW + POOL_GRID_W - 1) / POOL_GRID_W;
  hEnd = min(hEnd, dsH);
  wEnd = min(wEnd, dsW);

  const int cellW = max(0, wEnd - wStart);
  const int cellH = max(0, hEnd - hStart);
  const int count = cellW * cellH;

  for (int firstChannel = 0; firstChannel < CNN_FEATURE_DIM;
       firstChannel += CNN_FEATURE_CHUNK) {
    float partial[CNN_FEATURE_CHUNK];
#pragma unroll
    for (int c = 0; c < CNN_FEATURE_CHUNK; ++c) {
      partial[c] = 0.0f;
    }

    for (int idx = tid; idx < count; idx += POOL_THREAD_GROUP_SIZE) {
      const int dy = hStart + idx / cellW;
      const int dx = wStart + idx - (idx / cellW) * cellW;

      float featChunk[CNN_FEATURE_CHUNK];
      cnnForwardAtDownsampledPixelChunk(ox, oy, tileW, tileH, inW, inH, dx, dy,
                                        responsivity, inHdrColor, weights,
                                        firstChannel, featChunk);

#pragma unroll
      for (int c = 0; c < CNN_FEATURE_CHUNK; ++c) {
        partial[c] += featChunk[c];
      }
    }

#pragma unroll
    for (int c = 0; c < CNN_FEATURE_CHUNK; ++c) {
      gsReduce[tid] = partial[c];
      __syncthreads();

      for (int stride = POOL_THREAD_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          gsReduce[tid] += gsReduce[tid + stride];
        }
        __syncthreads();
      }

      if (tid == 0) {
        const int channel = firstChannel + c;
        const float invCount = (count > 0) ? (1.0f / (float)count) : 0.0f;
        outControllerFeatures[tile * POOL_FEATURE_LEN +
                              channel * POOL_CELL_COUNT + cell] =
            gsReduce[0] * invCount;
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// MLP entry point. Consumes the channel-major 1600-float tensor produced by
// controllerPoolProcess and writes [exposureOffset, colorLatents...] to
// outControllerParams.
// ---------------------------------------------------------------------------
extern "C" __global__ void
controllerProcess(const float *__restrict__ controllerFeatures,
                  float priorExposure,
                  float *__restrict__ outControllerParams) {
  const float *__restrict__ weights = kControllerWeights;
  __shared__ float gsHiddenA[MLP_HIDDEN_DIM];
  __shared__ float gsHiddenB[MLP_HIDDEN_DIM];

  const int tid = (int)threadIdx.x;
  // gridDim.x == tileCountX * tileCountY: one MLP block per tile. Each
  // block reads its tile's pooled features and writes its 9-float group.
  // For an untiled product (1 block) this is the original behavior.
  const int tile = (int)blockIdx.x;
  const float *__restrict__ features =
      controllerFeatures + tile * POOL_FEATURE_LEN;
  float *__restrict__ outParams =
      outControllerParams + tile * OUT_PARAMS_PER_TILE;
  __syncthreads();

  // Phase 1: trunk0 (1601 -> 128). 128 output rows are distributed
  // across the MLP block.
  for (int o = tid; o < MLP_HIDDEN_DIM; o += MLP_THREAD_GROUP_SIZE) {
    float v = weights[OFF_TRUNK0_B + o];
    for (int i = 0; i < POOL_FEATURE_LEN; ++i) {
      v += features[i] * weights[OFF_TRUNK0_W + o * MLP_INPUT_DIM + i];
    }
    v += priorExposure *
         weights[OFF_TRUNK0_W + o * MLP_INPUT_DIM + POOL_FEATURE_LEN];
    gsHiddenA[o] = fmaxf(0.0f, v);
  }
  __syncthreads();

  // Phase 2: trunk1 (128 -> 128). gsHiddenA -> gsHiddenB.
  for (int o = tid; o < MLP_HIDDEN_DIM; o += MLP_THREAD_GROUP_SIZE) {
    float v = weights[OFF_TRUNK1_B + o];
    for (int i = 0; i < MLP_HIDDEN_DIM; ++i) {
      v += gsHiddenA[i] * weights[OFF_TRUNK1_W + o * MLP_HIDDEN_DIM + i];
    }
    gsHiddenB[o] = fmaxf(0.0f, v);
  }
  __syncthreads();

  // Phase 3: trunk2 (128 -> 128). gsHiddenB -> gsHiddenA.
  for (int o = tid; o < MLP_HIDDEN_DIM; o += MLP_THREAD_GROUP_SIZE) {
    float v = weights[OFF_TRUNK2_B + o];
    for (int i = 0; i < MLP_HIDDEN_DIM; ++i) {
      v += gsHiddenB[i] * weights[OFF_TRUNK2_W + o * MLP_HIDDEN_DIM + i];
    }
    gsHiddenA[o] = fmaxf(0.0f, v);
  }
  __syncthreads();

  // Phase 4: heads.
  if (tid == 0) {
    float v = weights[OFF_EXP_B];
    for (int i = 0; i < MLP_HIDDEN_DIM; ++i) {
      v += gsHiddenA[i] * weights[OFF_EXP_W + i];
    }
    outParams[0] = v;
  }
  if (tid < COLOR_PARAMS_PER_FRAME) {
    const int o = tid;
    float v = weights[OFF_COL_B + o];
    for (int i = 0; i < MLP_HIDDEN_DIM; ++i) {
      v += gsHiddenA[i] * weights[OFF_COL_W + o * MLP_HIDDEN_DIM + i];
    }
    outParams[1 + o] = v;
  }
}
