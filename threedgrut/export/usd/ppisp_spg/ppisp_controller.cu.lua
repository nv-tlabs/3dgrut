-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
-- All rights reserved. SPDX-License-Identifier: Apache-2.0

-- PPISP Controller SPG Launchers (CUDA, embedded-weights variant).
--
-- Per-camera differences are compiled into generated CUDA sources as
-- file-scope device constants. This launcher intentionally binds no
-- weight asset so Kit does not upload a large weight buffer per frame.

local function getTileCount(inputs, name)
    local count = cuda.int(inputs[name] or 1).value
    if count < 1 then
        count = 1
    end
    return count
end

function controllerPoolProcess(inputs, outputs)
    local in_hdr = inputs["HdrColor"]
    assert(in_hdr ~= nil, "controllerPoolProcess: HdrColor input is missing")
    assert(#in_hdr.shape == 2, "controllerPoolProcess: HdrColor must be a 2D image")

    -- Tile grid for tiled RenderProduct atlases (default 1x1 = untiled).
    -- Each tile gets its own 1600-float pooled feature block (tile-major),
    -- so the MLP can produce independent params per tile.
    local tileCountX = getTileCount(inputs, "tileCountX")
    local tileCountY = getTileCount(inputs, "tileCountY")
    local tileCount = tileCountX * tileCountY

    -- Achromatic HDR multiplier; scales the controller input radiance to
    -- match the image PPISP shader (default 1.0 = no scaling).
    local responsivity = cuda.float(inputs["responsivity"] or 1.0).value

    outputs["ControllerFeatures"] = cuda.empty({ 1, 1600 * tileCount }, cuda.float)

    return cuda.kernel({
        args = {
            cuda.int(in_hdr.shape[2]),                      -- inW
            cuda.int(in_hdr.shape[1]),                      -- inH
            cuda.int(tileCountX),                           -- tileCountX
            cuda.int(tileCountY),                           -- tileCountY
            cuda.float(responsivity),                       -- responsivity
            cuda.TextureObject(in_hdr),                     -- inHdrColor
            cuda.array(outputs["ControllerFeatures"]),      -- outControllerFeatures
        },
        block = { 256, 1, 1 },
        grid  = { 25, tileCount, 1 },
    })
end

function controllerProcess(inputs, outputs)
    local features = inputs["ControllerFeatures"]
    assert(features ~= nil, "controllerProcess: ControllerFeatures input is missing")

    local tileCountX = getTileCount(inputs, "tileCountX")
    local tileCountY = getTileCount(inputs, "tileCountY")
    local tileCount = tileCountX * tileCountY

    -- 9 floats per tile holding [exposureOffset, color latents], laid out
    -- tile-major. Modeled as a flat cuda.empty (not cuda.image) so the
    -- chained auto-PPISP kernel can consume it as a const float*; this
    -- mirrors the kit-legendre DisparityKernel -> DepthKernel chain. One
    -- MLP block per tile (grid.x).
    outputs["ControllerParams"] = cuda.empty({ 1, 9 * tileCount }, cuda.float)

    return cuda.kernel({
        args = {
            cuda.array(features, cuda.float),               -- controllerFeatures
            cuda.float(inputs["priorExposure"]),            -- priorExposure
            cuda.array(outputs["ControllerParams"]),        -- outControllerParams
        },
        block = { 128, 1, 1 },
        grid  = { tileCount, 1, 1 },
    })
end
