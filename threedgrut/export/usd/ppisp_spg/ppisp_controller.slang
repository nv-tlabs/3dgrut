// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

// PPISP Controller SPG Shader.
//
// Generic compute shader: weights are bound at dispatch time as a flat
// ``StructuredBuffer<float>``. Per-camera variation lives entirely in
// USD attributes; the Slang/Lua/USDA assets are shared.
//
// Architecture mirrors ppisp._PPISPController (default config):
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
// Output texture (1x9 float, RWTexture2D<float>):
//   pixel (0,0): exposureOffset
//   pixel (1,0)..(8,0): color latents
//        [colorBlue.x, colorBlue.y,
//         colorRed.x,  colorRed.y,
//         colorGreen.x, colorGreen.y,
//         colorNeutral.x, colorNeutral.y]

// ---------------------------------------------------------------------------
// Architecture sizes (must match ``_PPISPController`` defaults).
// ---------------------------------------------------------------------------
static const int CNN_FEATURE_DIM = 64;
static const int POOL_GRID_H     = 5;
static const int POOL_GRID_W     = 5;
static const int POOL_CELL_COUNT = POOL_GRID_H * POOL_GRID_W;                     // 25
static const int POOL_FEATURE_LEN = POOL_CELL_COUNT * CNN_FEATURE_DIM;            // 1600
static const int MLP_INPUT_DIM    = POOL_FEATURE_LEN + 1;                         // 1601
static const int MLP_HIDDEN_DIM   = 128;
static const int COLOR_PARAMS_PER_FRAME = 8;
static const int INPUT_DOWNSAMPLING = 3;
static const int THREAD_GROUP_SIZE  = 32;

// ---------------------------------------------------------------------------
// Weight buffer offsets (the Python writer flattens weights in this order
// into a single float buffer that gets bound as weights).
// ---------------------------------------------------------------------------
static const int OFF_CONV1_W = 0;                                // 16 * 3       = 48
static const int OFF_CONV1_B = OFF_CONV1_W + 16 * 3;             // + 16          = 64
static const int OFF_CONV2_W = OFF_CONV1_B + 16;                 // + 32 * 16     = 576
static const int OFF_CONV2_B = OFF_CONV2_W + 32 * 16;            // + 32          = 608
static const int OFF_CONV3_W = OFF_CONV2_B + 32;                 // + 64 * 32     = 2656
static const int OFF_CONV3_B = OFF_CONV3_W + 64 * 32;            // + 64          = 2720
static const int OFF_TRUNK0_W = OFF_CONV3_B + 64;                // + 128 * 1601  = 207648
static const int OFF_TRUNK0_B = OFF_TRUNK0_W + 128 * MLP_INPUT_DIM;
static const int OFF_TRUNK1_W = OFF_TRUNK0_B + 128;
static const int OFF_TRUNK1_B = OFF_TRUNK1_W + 128 * 128;
static const int OFF_TRUNK2_W = OFF_TRUNK1_B + 128;
static const int OFF_TRUNK2_B = OFF_TRUNK2_W + 128 * 128;
static const int OFF_EXP_W    = OFF_TRUNK2_B + 128;
static const int OFF_EXP_B    = OFF_EXP_W + 128;
static const int OFF_COL_W    = OFF_EXP_B + 1;
static const int OFF_COL_B    = OFF_COL_W + 8 * 128;
static const int TOTAL_WEIGHTS = OFF_COL_B + 8;

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

// SPG resolves USD ``inputs:foo`` attributes against fields of the slang
// ParameterBlock -- its reflection lookup is ``params:foo``. Putting
// ``weights`` directly inside the ParameterBlock struct lets SPG's
// auto-binding find it by name, and silences the per-dispatch warning
// "Failed to find parameter 'params:weights' in shader reflection".
struct PPISPControllerParams
{
    float priorExposure;
    StructuredBuffer<float> weights;
};

[[vk::binding(0, 1)]] ParameterBlock<PPISPControllerParams> g_Params;
[[vk::binding(1, 1)]] Texture2D<float4>          g_InTex;
[[vk::binding(2, 1)]] RWTexture2D<float>         g_OutTex;

// ---------------------------------------------------------------------------
// Per-pixel CNN building blocks
// ---------------------------------------------------------------------------

void conv1Forward(float3 rgb, out float feat[16])
{
    [unroll] for (int o = 0; o < 16; ++o)
    {
        float v = g_Params.weights[OFF_CONV1_B + o];
        v += rgb.r * g_Params.weights[OFF_CONV1_W + o * 3 + 0];
        v += rgb.g * g_Params.weights[OFF_CONV1_W + o * 3 + 1];
        v += rgb.b * g_Params.weights[OFF_CONV1_W + o * 3 + 2];
        feat[o] = v;
    }
}

void conv2Forward(float fin[16], out float fout[32])
{
    [unroll] for (int o = 0; o < 32; ++o)
    {
        float v = g_Params.weights[OFF_CONV2_B + o];
        [unroll] for (int i = 0; i < 16; ++i)
            v += fin[i] * g_Params.weights[OFF_CONV2_W + o * 16 + i];
        fout[o] = v;
    }
}

void conv3Forward(float fin[32], out float fout[64])
{
    [unroll] for (int o = 0; o < CNN_FEATURE_DIM; ++o)
    {
        float v = g_Params.weights[OFF_CONV3_B + o];
        [unroll] for (int i = 0; i < 32; ++i)
            v += fin[i] * g_Params.weights[OFF_CONV3_W + o * 32 + i];
        fout[o] = v;
    }
}

void cnnForwardAtDownsampledPixel(
    int inW,
    int inH,
    int dx,
    int dy,
    out float feat64[64])
{
    int x0 = dx * INPUT_DOWNSAMPLING;
    int y0 = dy * INPUT_DOWNSAMPLING;
    int x1 = min(x0 + INPUT_DOWNSAMPLING, inW);
    int y1 = min(y0 + INPUT_DOWNSAMPLING, inH);

    float pooled[16];
    [unroll] for (int c = 0; c < 16; ++c)
        pooled[c] = -3.402823e+38;

    for (int yy = y0; yy < y1; ++yy)
    {
        for (int xx = x0; xx < x1; ++xx)
        {
            float4 sample = g_InTex.Load(int3(xx, yy, 0));
            float conv1Out[16];
            conv1Forward(sample.rgb, conv1Out);
            [unroll] for (int c = 0; c < 16; ++c)
                pooled[c] = max(pooled[c], conv1Out[c]);
        }
    }

    [unroll] for (int c = 0; c < 16; ++c)
        pooled[c] = max(0.0, pooled[c]);

    float feat32[32];
    conv2Forward(pooled, feat32);
    [unroll] for (int c = 0; c < 32; ++c)
        feat32[c] = max(0.0, feat32[c]);

    conv3Forward(feat32, feat64);
}

void adaptiveCellAverage(
    int inW,
    int inH,
    int dsW,
    int dsH,
    int gx,
    int gy,
    out float cellFeat[64])
{
    int hStart = (gy * dsH) / POOL_GRID_H;
    int hEnd   = ((gy + 1) * dsH + POOL_GRID_H - 1) / POOL_GRID_H;
    int wStart = (gx * dsW) / POOL_GRID_W;
    int wEnd   = ((gx + 1) * dsW + POOL_GRID_W - 1) / POOL_GRID_W;
    hEnd = min(hEnd, dsH);
    wEnd = min(wEnd, dsW);

    [unroll] for (int c = 0; c < CNN_FEATURE_DIM; ++c)
        cellFeat[c] = 0.0;

    int count = 0;
    for (int dy = hStart; dy < hEnd; ++dy)
    {
        for (int dx = wStart; dx < wEnd; ++dx)
        {
            float feat64[CNN_FEATURE_DIM];
            cnnForwardAtDownsampledPixel(inW, inH, dx, dy, feat64);
            [unroll] for (int c = 0; c < CNN_FEATURE_DIM; ++c)
                cellFeat[c] += feat64[c];
            count += 1;
        }
    }

    float invCount = (count > 0) ? (1.0 / float(count)) : 0.0;
    [unroll] for (int c = 0; c < CNN_FEATURE_DIM; ++c)
        cellFeat[c] *= invCount;
}

groupshared float gsPooled[POOL_FEATURE_LEN];     // 1600 floats
groupshared float gsHiddenA[MLP_HIDDEN_DIM];      //  128 floats
groupshared float gsHiddenB[MLP_HIDDEN_DIM];      //  128 floats

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void controllerProcess(uint3 gtid : SV_GroupThreadID)
{
    uint inW = 0, inH = 0;
    g_InTex.GetDimensions(inW, inH);

    int dsW = max(1u, inW / INPUT_DOWNSAMPLING);
    int dsH = max(1u, inH / INPUT_DOWNSAMPLING);

    // Phase 1: pool cells. With THREAD_GROUP_SIZE=32 threads and 25 cells,
    // only the first 25 are active in this phase.
    //
    // Layout note: PyTorch's nn.Flatten on the [N, C, H, W] CNN output
    // produces a *channel-major* flat layout — feat[c * H*W + h*W + w].
    // The trunk0 weight matrix was trained against that layout, so
    // gsPooled MUST be stored channel-major as well, i.e.
    //     gsPooled[c * POOL_CELL_COUNT + cell].
    // (cell-major would silently permute every controller output.)
    int cell = int(gtid.x);
    if (cell < POOL_CELL_COUNT)
    {
        int gy = cell / POOL_GRID_W;
        int gx = cell % POOL_GRID_W;

        float cellFeat[CNN_FEATURE_DIM];
        adaptiveCellAverage(int(inW), int(inH), dsW, dsH, gx, gy, cellFeat);

        [unroll] for (int c = 0; c < CNN_FEATURE_DIM; ++c)
            gsPooled[c * POOL_CELL_COUNT + cell] = cellFeat[c];
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: trunk0 (1601 -> 128). 128 output rows are distributed
    // across the THREAD_GROUP_SIZE threads.
    for (int o = int(gtid.x); o < MLP_HIDDEN_DIM; o += THREAD_GROUP_SIZE)
    {
        float v = g_Params.weights[OFF_TRUNK0_B + o];
        for (int i = 0; i < POOL_FEATURE_LEN; ++i)
            v += gsPooled[i] * g_Params.weights[OFF_TRUNK0_W + o * MLP_INPUT_DIM + i];
        v += g_Params.priorExposure
             * g_Params.weights[OFF_TRUNK0_W + o * MLP_INPUT_DIM + POOL_FEATURE_LEN];
        gsHiddenA[o] = max(0.0, v);
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: trunk1 (128 -> 128). gsHiddenA -> gsHiddenB.
    for (int o = int(gtid.x); o < MLP_HIDDEN_DIM; o += THREAD_GROUP_SIZE)
    {
        float v = g_Params.weights[OFF_TRUNK1_B + o];
        for (int i = 0; i < MLP_HIDDEN_DIM; ++i)
            v += gsHiddenA[i] * g_Params.weights[OFF_TRUNK1_W + o * MLP_HIDDEN_DIM + i];
        gsHiddenB[o] = max(0.0, v);
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 4: trunk2 (128 -> 128). gsHiddenB -> gsHiddenA.
    for (int o = int(gtid.x); o < MLP_HIDDEN_DIM; o += THREAD_GROUP_SIZE)
    {
        float v = g_Params.weights[OFF_TRUNK2_B + o];
        for (int i = 0; i < MLP_HIDDEN_DIM; ++i)
            v += gsHiddenB[i] * g_Params.weights[OFF_TRUNK2_W + o * MLP_HIDDEN_DIM + i];
        gsHiddenA[o] = max(0.0, v);
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 5: heads.
    if (gtid.x == 0)
    {
        float v = g_Params.weights[OFF_EXP_B];
        for (int i = 0; i < MLP_HIDDEN_DIM; ++i)
            v += gsHiddenA[i] * g_Params.weights[OFF_EXP_W + i];
        g_OutTex[int2(0, 0)] = v;
    }
    if (gtid.x < uint(COLOR_PARAMS_PER_FRAME))
    {
        int o = int(gtid.x);
        float v = g_Params.weights[OFF_COL_B + o];
        for (int i = 0; i < MLP_HIDDEN_DIM; ++i)
            v += gsHiddenA[i] * g_Params.weights[OFF_COL_W + o * MLP_HIDDEN_DIM + i];
        g_OutTex[int2(1 + o, 0)] = v;
    }
}
