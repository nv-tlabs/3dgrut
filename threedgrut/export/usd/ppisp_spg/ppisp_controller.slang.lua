-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- PPISP Controller SPG Launcher.
--
-- Single shared launcher for every camera. Per-camera differences are
-- carried by the ``weights`` USD attribute, so this file does not need
-- to be regenerated.

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
            slang.StructuredBuffer(weights),
            slang.RWTexture2D(outputs["ControllerParams"]),
        },
    })
end
