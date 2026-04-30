-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- PPISP SPG Launcher (controller-aware variant).
--
-- Reads ``exposureOffset`` and the eight colour latents from the
-- controller's output texture; the static USD inputs only carry the
-- per-camera vignetting and CRF parameters. The HdrColor input still
-- comes from the RenderProduct's primary AOV.

function ppispProcessDyn(inputs, outputs, params)
    local in_rgba = inputs["HdrColor"]
    assert(in_rgba and in_rgba.rank == 2, "HdrColor input must be a 2D texture")

    local controller = inputs["ControllerParams"]
    assert(controller, "ppispProcessDyn needs a ControllerParams input texture")

    local height = in_rgba.shape[1]
    local width = in_rgba.shape[2]
    outputs["PPISPColor"] = slang.empty({ height, width }, slang.uchar4)

    local function getFloat2(name)
        local p = params[name]
        return p and slang.float2(p) or slang.float2(0.0, 0.0)
    end

    return slang.dispatch({
        stage = "compute",
        numthreads = { 16, 16, 1 },
        grid = { math.ceil(width / 16), math.ceil(height / 16), 1 },
        bind = {
            slang.ParameterBlock(
                getFloat2("vignettingCenterR"),
                slang.float(params["vignettingAlpha1R"] or 0.0),
                slang.float(params["vignettingAlpha2R"] or 0.0),
                slang.float(params["vignettingAlpha3R"] or 0.0),

                getFloat2("vignettingCenterG"),
                slang.float(params["vignettingAlpha1G"] or 0.0),
                slang.float(params["vignettingAlpha2G"] or 0.0),
                slang.float(params["vignettingAlpha3G"] or 0.0),

                getFloat2("vignettingCenterB"),
                slang.float(params["vignettingAlpha1B"] or 0.0),
                slang.float(params["vignettingAlpha2B"] or 0.0),
                slang.float(params["vignettingAlpha3B"] or 0.0),

                slang.float(params["crfToeR"] or 0.013659),
                slang.float(params["crfShoulderR"] or 0.013659),
                slang.float(params["crfGammaR"] or 0.378165),
                slang.float(params["crfCenterR"] or 0.0),

                slang.float(params["crfToeG"] or 0.013659),
                slang.float(params["crfShoulderG"] or 0.013659),
                slang.float(params["crfGammaG"] or 0.378165),
                slang.float(params["crfCenterG"] or 0.0),

                slang.float(params["crfToeB"] or 0.013659),
                slang.float(params["crfShoulderB"] or 0.013659),
                slang.float(params["crfGammaB"] or 0.378165),
                slang.float(params["crfCenterB"] or 0.0)
            ),
            slang.Texture2D(in_rgba),
            slang.Texture2D(controller),
            slang.RWTexture2D(outputs["PPISPColor"]),
        },
    })
end
