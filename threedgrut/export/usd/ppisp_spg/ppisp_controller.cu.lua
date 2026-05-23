-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- PPISP Controller SPG Launchers (CUDA, embedded-weights variant).
--
-- Per-camera differences are compiled into generated CUDA sources as
-- file-scope device constants. This launcher intentionally binds no
-- weight asset so Kit does not upload a large weight buffer per frame.

function controllerPoolProcess(inputs, outputs)
    local in_hdr = inputs["HdrColor"]
    assert(in_hdr ~= nil, "controllerPoolProcess: HdrColor input is missing")
    assert(#in_hdr.shape == 2, "controllerPoolProcess: HdrColor must be a 2D image")

    outputs["ControllerFeatures"] = cuda.empty({ 1, 1600 }, cuda.float)

    return cuda.kernel({
        args = {
            cuda.int(in_hdr.shape[2]),                      -- inW
            cuda.int(in_hdr.shape[1]),                      -- inH
            cuda.TextureObject(in_hdr),                     -- inHdrColor
            cuda.array(outputs["ControllerFeatures"]),      -- outControllerFeatures
        },
        block = { 256, 1, 1 },
        grid  = { 25, 1, 1 },
    })
end

function controllerProcess(inputs, outputs)
    local features = inputs["ControllerFeatures"]
    assert(features ~= nil, "controllerProcess: ControllerFeatures input is missing")

    -- 1x9 single-channel float buffer holding [exposureOffset, color
    -- latents]. Modeled as a flat cuda.empty (not cuda.image) so the
    -- chained auto-PPISP kernel can consume it as a const float*; this
    -- mirrors the kit-legendre DisparityKernel -> DepthKernel chain.
    outputs["ControllerParams"] = cuda.empty({ 1, 9 }, cuda.float)

    return cuda.kernel({
        args = {
            cuda.array(features, cuda.float),               -- controllerFeatures
            cuda.float(inputs["priorExposure"]),            -- priorExposure
            cuda.array(outputs["ControllerParams"]),        -- outControllerParams
        },
        block = { 128, 1, 1 },
        grid  = { 1, 1, 1 },
    })
end
