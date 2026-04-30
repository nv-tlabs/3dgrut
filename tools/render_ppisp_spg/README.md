# render_ppisp_spg

Headless **slangpy** harness for the PPISP SPG sidecars. Lets you
validate the exported `*.slang` / `*.slang.lua` chain end-to-end without
booting Omniverse Kit.

## What it does

- Loads `ppisp_controller.slang` (and the dynamic / static
  `ppisp_usd_spg*.slang`) directly from the on-disk SPG sidecar set.
- Strips the `[[vk::binding(*, *)]]` annotations that Kit's SPG layer
  consumes (slangpy uses its own auto-binding) and dispatches the same
  compute kernel.
- Reads time-sampled USD attributes off a PPISP-bearing
  `RenderProduct` and walks frame-by-frame against an HDR input dir.

## Three entry points

| Function | Use |
| --- | --- |
| `run_controller(slang, hdr, weights, prior=0)` | Returns the 9-float controller output: `[exposureOffset, blue.xy, red.xy, green.xy, neutral.xy]`. |
| `run_ppisp_dyn(slang, hdr, ctrl_out, vignette, crf)` | Reads colour / exposure from a controller output texture and returns an LDR uint8 image. |
| `run_ppisp_static(slang, hdr, exposure, color_latents, vignette, crf)` | The legacy controller-free path; reads exposure / colour from explicit args. |

## CLI

```
python tools/render_ppisp_spg/render_renderproduct.py \
    out.usdz hdr_inputs/ ldr_outputs/
```

The HDR input layout is one folder per camera-name, with files named
`<frame_index>.{npy,exr,png}`.

## Validation

```
python tools/render_ppisp_spg/validate_controller.py --tol 1e-4
```

Generates a synthetic torch `_PPISPController`, bakes its weights via
`flatten_controller_weights`, dispatches the SPG controller shader, and
compares the 9-element result against the torch reference. Typical max
abs diff is around 4e-6.

## Dependencies

`slangpy`, `numpy`, `Pillow`, `usd-core`, and (only for
`validate_controller.py`) `torch`. `OpenEXR`/`Imath` are optional and
only loaded when an `.exr` HDR input is encountered.
