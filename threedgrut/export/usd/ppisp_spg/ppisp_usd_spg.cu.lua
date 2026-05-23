-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- PPISP SPG CUDA launcher.

function ppispProcess(inputs, outputs)
    local in_hdr = inputs["HdrColor"]
    assert(in_hdr and in_hdr.rank == 2, "HdrColor input must be a 2D texture")

    local height = in_hdr.shape[1]
    local width = in_hdr.shape[2]
    outputs["PPISPColor"] = cuda.image(width, height, cuda.uchar4)

    local function getFloat(name, default)
        return cuda.float(inputs[name] or default).value
    end

    local function getFloat2(name)
        local value = inputs[name]
        local packed = value and cuda.float2(value) or cuda.float2(0.0, 0.0)
        return packed.value
    end

    local vignettingCenterR = getFloat2("vignettingCenterR")
    local vignettingCenterG = getFloat2("vignettingCenterG")
    local vignettingCenterB = getFloat2("vignettingCenterB")
    local colorLatentBlue = getFloat2("colorLatentBlue")
    local colorLatentRed = getFloat2("colorLatentRed")
    local colorLatentGreen = getFloat2("colorLatentGreen")
    local colorLatentNeutral = getFloat2("colorLatentNeutral")
    local params = {
        getFloat("responsivity", 1.0),
        getFloat("exposureOffset", 0.0),
        vignettingCenterR[1],
        vignettingCenterR[2],
        getFloat("vignettingAlpha1R", 0.0),
        getFloat("vignettingAlpha2R", 0.0),
        getFloat("vignettingAlpha3R", 0.0),
        vignettingCenterG[1],
        vignettingCenterG[2],
        getFloat("vignettingAlpha1G", 0.0),
        getFloat("vignettingAlpha2G", 0.0),
        getFloat("vignettingAlpha3G", 0.0),
        vignettingCenterB[1],
        vignettingCenterB[2],
        getFloat("vignettingAlpha1B", 0.0),
        getFloat("vignettingAlpha2B", 0.0),
        getFloat("vignettingAlpha3B", 0.0),
        colorLatentBlue[1],
        colorLatentBlue[2],
        colorLatentRed[1],
        colorLatentRed[2],
        colorLatentGreen[1],
        colorLatentGreen[2],
        colorLatentNeutral[1],
        colorLatentNeutral[2],
        getFloat("crfToeR", 0.013659),
        getFloat("crfShoulderR", 0.013659),
        getFloat("crfGammaR", 0.378165),
        getFloat("crfCenterR", 0.0),
        getFloat("crfToeG", 0.013659),
        getFloat("crfShoulderG", 0.013659),
        getFloat("crfGammaG", 0.378165),
        getFloat("crfCenterG", 0.0),
        getFloat("crfToeB", 0.013659),
        getFloat("crfShoulderB", 0.013659),
        getFloat("crfGammaB", 0.378165),
        getFloat("crfCenterB", 0.0),
    }

    return cuda.kernel({
        args = {
            cuda.int(width),
            cuda.int(height),
            cuda.TextureObject(in_hdr),
            cuda.array(params, cuda.float),
            cuda.SurfaceObject(outputs["PPISPColor"]),
        },
        block = { 16, 16, 1 },
        grid = { math.ceil(width / 16), math.ceil(height / 16), 1 },
    })
end
