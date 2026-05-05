# PPISP Controller SPG Export — Feasibility & Plan

Scope: extend the existing PPISP SPG export to include the **controller** —
the per-camera CNN+MLP that predicts per-frame `exposure` and 8-d
`color_latents` from the rendered image.

Status: feasible. This document records the design before implementation.

---

## 1. Controller summary (from `ppisp.PPISP._PPISPController`)

Fixed architecture, one instance per camera. Inputs are the raw rendered
HDR image and an optional `prior_exposure` scalar.

```
Conv2d(3→16, 1×1, +bias)
MaxPool2d(3, stride=3)
ReLU
Conv2d(16→32, 1×1, +bias)
ReLU
Conv2d(32→64, 1×1, +bias)
AdaptiveAvgPool2d((5, 5))
Flatten                        # → 1600 features
concat(prior_exposure)         # → 1601
Linear(1601 → 128) + ReLU
Linear(128  → 128) + ReLU
Linear(128  → 128) + ReLU
exposure_head: Linear(128 → 1)
color_head:    Linear(128 → 8)
```

Total weights ≈ **240 K floats per camera**.

The output is **two scalar values for the whole image** (1 exposure, 8 colour
latents). Those nine numbers replace the current static USD time-samples on
the PPISP shader.

---

## 2. SPG capabilities used

The 3DGRUT SPG pipeline already uses **Slang** in the SPG runtime
(`*.slang`, `*.slang.lua`, `*.slang.usda`) — the public docs that describe
only CUDA kernels are out-of-date; the existing PPISP shader proves Slang
launch is supported.

Confirmed primitives:

- `slang.dispatch{ stage="compute", numthreads=…, grid=…, bind={…} }` per
  shader prim.
- `slang.ParameterBlock(...)` for grouped scalar/vector inputs that map to
  USD attributes.
- `slang.Texture2D / slang.RWTexture2D / slang.empty(shape, dtype)` for
  bound textures and lua-allocated outputs.
- Shader-to-shader chaining via USD `omni:rtx:aov` connections on
  `RenderVar` prims (the existing `LdrColor` → `PPISP` wiring uses this).

What we **do not** rely on:
- Multi-dispatch within one Lua launcher (one dispatch per shader prim).
- CooperativeVector / coopvec — not assumed available in the target Kit.
- Non-2D output buffers — only 2D images via `slang.empty`.

---

## 3. Two challenges and how we solve them

### 3.1 Adaptive avg pooling on a runtime-sized input

PyTorch's `AdaptiveAvgPool2d((5,5))` partitions the input into exactly 25
near-equal cells. The cell bounds are:

```
i = 0..4 (output row)
start_h = floor(i * H_in / 5)
end_h   = ceil((i + 1) * H_in / 5)
```

Each Slang thread group computes one output cell `(i, j)` by reading every
input pixel in `[start_h, end_h) × [start_w, end_w)`, applying the
per-pixel CNN forward (3×3 max-pool fused with the surrounding 1×1
convolutions), and reducing the sum / divide-by-count in shared memory.

This works for arbitrary input resolution because the cell bounds are
computed inside the shader from `H_in, W_in`.

### 3.2 Baking MLP and CNN weights into Slang

Each camera's controller has unique weights. We generate **one Slang file
per camera** at export time, with all weights emitted as
`static const float[]` arrays. Slang's compiler can fold these into
constant memory, and there is no runtime upload step.

The generated file `ppisp_controller_<camIdx>.slang` includes a fixed
shared template (CNN forward, pool, MLP) and only differs in the weight
constants. The matching `*.slang.lua` and `*.slang.usda` are emitted per
camera as well so each `RenderProduct` references its own controller.

If weights ever exceed Slang's static-data limits we can fall back to
USD `float[]` inputs bound as a `StructuredBuffer<float>`, but for the
default architecture (~240 K floats) static arrays are fine.

---

## 4. SPG graph

```
HdrColor (RenderVar)
   │
   ▼  (omni:rtx:aov connection)
PPISPController_<cam>      Slang compute, single thread group
   │  outputs ControllerParams (1×9 float image)
   ▼
PPISP                        Slang compute, grid sized to image
   │  reads HdrColor + ControllerParams + static vignetting/CRF
   ▼  outputs PPISPColor
LdrColor (RenderVar)
```

The existing `ppisp_writer.py` builds the second half. The new
controller writer creates the first stage and connects its output as
an additional input to the PPISP shader.

The PPISP slang shader is **generalised** to read the exposure and the 8
colour latents from a 1×9 single-channel float texture when one is bound,
falling back to its `ParameterBlock` defaults otherwise. This keeps the
legacy "static parameters per frame" path unchanged — important for users
who train without a controller.

---

## 5. Testing

Two-pronged approach:

1. **Unit-level Python check** that the generated Slang reproduces the
   PyTorch controller's outputs to within a tight tolerance, using
   `slangpy` to dispatch the controller shader against a reference image.

2. **Tool: `tools/render_ppisp_spg/`** — a slangpy-based runner
   that opens an exported USD/USDZ, walks `/Render/<cam>` prims, finds
   their SPG shader chain, and replays the chain on a supplied HDR input
   for every authored time sample. Useful for visual regression and for
   cross-checking that the PPISP USD asset produces the same image
   sequence as the in-process `apply_post_processing` path used during
   training.

The render tool intentionally does not try to reproduce Kit's full
RenderProduct pipeline; it executes only the SPG `compute` stages so it
remains independent of Kit and useful in headless CI.

---

## 6. Out of scope for this iteration

- Multi-dispatch optimisation of the controller (currently one slow but
  correct compute pass).
- CoopVec acceleration of the MLP matmul.
- Quantising weights to fp16/bf16 to reduce shader source size.
- Runtime weight upload (large `float[]` USD inputs).

These can be added later if the basic export proves correct.
