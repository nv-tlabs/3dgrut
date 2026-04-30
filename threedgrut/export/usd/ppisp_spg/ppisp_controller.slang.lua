-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- PPISP Controller SPG Launcher.
--
-- Single shared launcher for every camera. Per-camera differences are
-- carried by the ``weights`` USD attribute, so this file does not need
-- to be regenerated.

-- Bind the controller weight buffer using whichever buffer-helper SPG's
-- slang lua API exposes. The trained weights live as a USD float[]
-- attribute (params["weights"]) and the slang shader reads them as a
-- read-only StructuredBuffer<float>.
local function bind_weights(w)
    -- Probe a long list of plausible names. The first non-nil wins.
    local candidates = {
        "StructuredBuffer", "RWStructuredBuffer",
        "Buffer", "RWBuffer",
        "ByteAddressBuffer", "RWByteAddressBuffer",
        "ConstantBuffer",
        "buffer", "Array", "array",
        "float_array", "FloatArray", "floatArray",
        "FloatBuffer", "floatBuffer",
        "image", "Image",
        "uniform", "Uniform",
        "list", "List",
    }
    local hits = {}
    for _, name in ipairs(candidates) do
        if slang[name] ~= nil then
            table.insert(hits, name)
        end
    end
    if #hits > 0 then
        return slang[hits[1]](w)
    end
    -- No buffer helper. List EVERY direct slang.* key plus every
    -- candidate we tried (so the metatable surface is also probed via
    -- __index above). The error message goes to Kit's log.
    local direct = {}
    for k, _ in pairs(slang) do table.insert(direct, tostring(k)) end
    table.sort(direct)
    error("ppisp_controller: no slang buffer-binding helper found. " ..
          "Tried: " .. table.concat(candidates, ",") ..
          " | direct keys = " .. table.concat(direct, ","))
end

function controllerProcess(inputs, outputs, params)
    local in_rgba = inputs["HdrColor"]
    assert(in_rgba and in_rgba.rank == 2, "HdrColor input must be a 2D texture")

    local weights = params["weights"]
    assert(weights, "controllerProcess needs the inputs:weights attribute")

    -- 1x9 single-channel float image holding [exposure, color latents].
    outputs["ControllerParams"] = slang.empty({ 1, 9 }, slang.float)

    return slang.dispatch({
        stage = "compute",
        numthreads = { 32, 1, 1 },
        grid = { 1, 1, 1 },
        bind = {
            slang.ParameterBlock(
                slang.float(params["priorExposure"] or 0.0)
            ),
            slang.Texture2D(in_rgba),
            bind_weights(weights),
            slang.RWTexture2D(outputs["ControllerParams"]),
        },
    })
end
