# PPISP USD Export

Deep dive into the PPISP integration paths exposed by 3DGRUT's USD
exporter. The general export workflows (USD / PLY / NuRec export,
transcoding, format/schema/half-precision options) are documented in the
export README [`../../README.md`](../../README.md); this document covers only the
PPISP-related parameters and the authored USD surface they produce.

The two entry points share the same PPISP behavior:

- Training-time export via the `export_usd` config block:
  ```bash
  python train.py --config-name apps/colmap_3dgut.yaml path=data/... \
      out_dir=runs experiment_name=scene export_usd.enabled=true
  ```
- Standalone export from a checkpoint:
  ```bash
  python -m threedgrut.export.scripts.export_usd \
      --checkpoint path/to/checkpoint.pt \
      --dataset path/to/dataset \
      --output path/to/asset.usdz \
      --ppisp-integration-mode spg-runtime
  ```

The exporter's `--help` is the canonical CLI surface:

```bash
python -m threedgrut.export.scripts.export_usd --help
```

## Contents

- [Overview](#overview)
- [Story 1 — Color matching for USD integration](#story-1--color-matching-for-usd-integration)
- [Story 2 — SH-optimized export (PPISP folded into SH)](#story-2--sh-optimized-export-ppisp-folded-into-sh)
- [Story 3 — Controller export](#story-3--controller-export)
- [PPISP CLI / config reference](#ppisp-cli--config-reference)
- [Authored USD surface](#authored-usd-surface)
- [Traps and gotchas](#traps-and-gotchas)
- [Troubleshooting](#troubleshooting)
- [Best practices](#best-practices)

## Overview

PPISP (Physically-Plausible Image Signal Processing) is the learned
post-processing module from
[`nv-tlabs/ppisp`](https://github.com/nv-tlabs/ppisp): per-frame
exposure, per-camera vignetting, per-frame color correction, and
per-camera CRF, plus a controller CNN that predicts exposure and color
from the rendered radiance image at inference time.

When a checkpoint contains a supported PPISP module, USD export includes
post-processing by default (`export_usd.export_post_processing=true`). A
Gaussian USD asset can carry these effects two different ways depending on
the target viewer's capabilities:

- **`spg-runtime`** _(default)_: PPISP is authored as Omniverse Sensor
  Processing Graph (SPG) shader prims on every RenderProduct. The viewer
  applies vignetting, ZCA color, exposure and CRF at render time. Highest
  fidelity. Requires an SPG-capable viewer (Omniverse Kit / RTX).
  **Sensor-specific.**
- **`sh-optimized`**: PPISP is folded into the Gaussian spherical harmonic
  coefficients before export. Any USD viewer can render the corrected
  output without runtime PPISP support. Lower fidelity (PPISP becomes a
  per-asset baked approximation tied to one reference camera/frame).
  **Single-sensor static look.**

A third path — the per-camera PPISP **controller** — is used by default in
`spg-runtime` export when the loaded PPISP module was trained with
controllers; see [Story 3](#story-3--controller-export).

The seven flags that drive PPISP integration are:

```
--ppisp-integration-mode {spg-runtime,sh-optimized}
--scene-radiance-scale FLOAT
--ppisp-responsivity FLOAT
--ppisp-reference-camera-id INT
--ppisp-reference-frame-id INT
--enable-ppisp-controller-export / --disable-ppisp-controller-export
--sh-optimization-num-iterations INT
```

All other flags (`--checkpoint`, `--dataset`, `--output`, `--format`,
`--half*`, `--no-cameras`, `--no-background`, ...) are described in the
repository README and are unchanged by PPISP export.

The equivalent training-config keys live under `export_usd` (hyphenated to
match the CLI flags):

```yaml
export_usd:
  export_post_processing: true
  ppisp-integration-mode: spg-runtime      # spg-runtime | sh-optimized
  ppisp-reference-camera-id: null
  ppisp-reference-frame-id: null
  ppisp-responsivity: 1.0
  enable-ppisp-controller-export: null     # null = auto
  sh-optimization-num-iterations: null     # null = 3000
  scene-radiance-scale: 1.0
```

## Story 1 — Color matching for USD integration

A Gaussian asset is trained against an arbitrary, scene-internal
radiometric scale. The USD scene it gets composited into has its _own_
radiometric scale (whatever physical units the target renderer expects).
Dropping the trained asset in unmodified leaves the Gaussian radiance
authored at the wrong scale relative to the rest of the scene's lights and
surfaces, so the asset reads either too bright or too dark, and downstream
exposure / tonemap stages misbehave.

`--scene-radiance-scale` is the **radiometric matching knob**: multiply
Gaussian SH by `k` to bring the asset into the target renderer's radiance
scale. `--ppisp-responsivity` is the **post-ISP compensator**: when the
Gaussian radiance is rescaled by `k`, the HDR buffer arriving at PPISP is
also rescaled by `k`, so PPISP's exposure / vignetting / CRF stages see a
different input and the LDR look drifts. Setting
`inputs:responsivity = 1/k` puts the value PPISP sees back to its
training-time magnitude, so the post-ISP look is preserved.

Pipeline:

```
Gaussian SH coefficients
        |  multiplied by --scene-radiance-scale = k (asset-level, baked)
        v
   per-view render -> HDR buffer at k * (training radiance)
        |
        |  multiplied by inputs:responsivity (= --ppisp-responsivity, default 1.0)
        v
   PPISP shader (spg-runtime only): exposure -> vignette -> ZCA color -> CRF
        v
   LDR
```

Two intent-driven recipes:

- **Match the target renderer's units, keep the original look**: set
  `--scene-radiance-scale k` to whatever rescaling the target scene needs,
  and set `--ppisp-responsivity 1/k`. PPISP sees the same input it saw at
  training time, the LDR look is preserved.
- **Match the target renderer's units, also retune the look**: set
  `--scene-radiance-scale k` for the radiometric matching, then set
  `--ppisp-responsivity` to whatever post-ISP brightness you want
  (independent of `k`). The two flags are independent — the `1/k` is just
  the "no look change" choice.

Both flags can also be used in isolation:

- Only `--scene-radiance-scale` (responsivity left at `1.0`): rescale
  Gaussian radiance and _let_ PPISP react to it. The look changes; useful
  when the original look was wrong and the rescale should drive the new
  exposure.
- Only `--ppisp-responsivity` (scale left at `1.0`): retune the post-ISP
  look without touching the asset's radiance. Useful for per-shot grading
  at consume time — the attribute is overridable in USD, so consumers can
  edit it without re-exporting.

### `--scene-radiance-scale FLOAT` (default `1.0`)

Multiplies the SH-evaluated RGB output of every Gaussian _before_ it is
written to the USD asset (the DC offset is compensated so rendered output
equals `scene_radiance_scale × original eval`). Permanent (baked into the
asset); retuning requires re-exporting. Layered _after_ the SH bake in
`sh-optimized` mode so the bake happens on the unscaled SH and the scale
multiplies the bake (see
[SH bake runs before radiance scaling](#sh-bake-runs-before-radiance-scaling)).

Strictly positive finite floats only; `0.0`, negatives, `inf`, `nan` and
non-numeric values are rejected by `_normalize_positive_finite_float` in
[`../exporter.py`](../exporter.py).

### `--ppisp-responsivity FLOAT` (default `1.0`)

Authors the same float on every PPISP shader's `inputs:responsivity` USD
attribute (the image PPISP shader plus, in controller mode, the
`PPISPControllerPool` shader). The SPG shader multiplies the HDR buffer by
this value _before_ any PPISP stage; the controller pool applies the same
scale to its HDR input before feature extraction so its exposure / color
predictions are computed on the same signal the image shader processes.
The attribute is overridable in USD: a downstream consumer can edit it per
shot without re-exporting the asset.

Same fail-loud validation rules as `--scene-radiance-scale`.

> **Trap**: `--ppisp-responsivity` is consumed only by the SPG shader,
> which only runs in `spg-runtime` mode. In `sh-optimized` mode there is no
> SPG shader to read the attribute, so a non-default value is currently a
> silent no-op. See
> [Traps and gotchas](#--ppisp-responsivity-is-a-no-op-in-sh-optimized).

## Story 2 — SH-optimized export (PPISP folded into SH)

When the target viewer cannot run PPISP at runtime (no SPG / no RTX, or the
asset must work in any USD-aware tool), pick
`--ppisp-integration-mode sh-optimized`. The exporter then bakes PPISP into
the Gaussian SH coefficients so the rendered image already looks
PPISP-corrected without a shader. The bake lives in
[`sh_bake.py`](sh_bake.py)
(`bake_post_processing_into_sh`).

The bake is a two-phase optimization:

1. **DC warm-start.** Each Gaussian's diffuse RGB is passed through PPISP at
   the resolved `(camera_id, frame_id)` and written back into the SH DC
   band of `features_albedo` (via the adapter's `initialize_fit`). This
   gives the refinement loop a much better starting point than the trained
   coefficients (which live in pre-PPISP space). `features_specular` is
   intentionally not warm-started.
2. **Multi-view Adam refinement loop.** Each iteration draws a training
   view, renders the _current_ model and a deep-copied _reference_ model
   under that view, computes the L1 loss between `PPISP(reference)` and the
   current render, and takes one Adam step on the SH coefficients (and on
   per-Gaussian density, which absorbs spatial frequencies SH cannot). The
   reference model is a `deepcopy` taken once at loop entry so its SH does
   not drift while the live model is updated.

The reference render disables vignetting (`FixedPPISP(..., include_vignetting=False)`),
matching the warm-start, so the bake target stays color-space-consistent
and avoids folding a per-pixel vignetting falloff into per-Gaussian SH
(which SH cannot represent properly anyway).

### Reference camera/frame in `sh-optimized`

`--ppisp-reference-camera-id` and `--ppisp-reference-frame-id` pick _which_
PPISP configuration gets folded:

- Both unset: defaults to camera `0`, frame `0`.
- One set, the other unset: the missing axis falls back to `0`.
- Both set: the bake targets exactly that `(camera_id, frame_id)` pair. The
  pair must be in range; out-of-range values are rejected with
  `camera_id must be in [0, N-1], got <value>` /
  `frame_id must be in [0, N-1], got <value>`.

> Once baked, the asset has a single PPISP look. To compare or swap looks,
> run the export twice with different reference IDs and ship the two assets
> separately.

### `--sh-optimization-num-iterations INT` (default `None` ⇒ 3000)

Overrides the optimizer step budget. Strictly positive int; `0`,
negatives, floats and bools are rejected (`num_iterations must be int, ...`
/ `num_iterations must be >= 1, ...`). Only meaningful in `sh-optimized`
mode.

The default 3000 (~7 epochs × a few hundred training views) brings parity
with the SH-bake baseline on representative scenes. Reduce if export time
matters more than bake quality; raise if the bake still looks
under-converged. The Adam learning rates are _not_ CLI-tunable — they are
algorithm-tuned (`features_albedo` `2.5e-3`, `features_specular` = albedo /
20, `density` `5e-2`) and only exposed as keyword arguments to
`bake_post_processing_into_sh` (e.g. tests forcing `device="cpu"`).

## Story 3 — Controller export

The PPISP controller is a small CNN that predicts per-frame PPISP exposure
/ color latents from the per-view feature buffer. During training it runs
every iteration; it is the path that yields the most realistic novel-view
PPISP behavior. In `spg-runtime` mode, USD export uses the controller by
default when the loaded PPISP module was trained with controllers
(`ppisp_has_controller` /
`resolve_ppisp_controller_export_enabled` in
[`ppisp_spg/__init__.py`](ppisp_spg/__init__.py)). The exporter generates
per-camera CUDA sidecars with weights embedded in device constants, so
target viewers must be able to compile and resolve those generated
sidecars. No separate controller-weight sidecar or runtime
`inputs:weights` binding is authored.

When the checkpoint contains controller-trained PPISP, export authors:

- a per-camera `PPISPControllerPool` shader prim wired to the pre-PPISP HDR
  RenderVar (it scales that HDR by `inputs:responsivity` before feature
  extraction, matching the image PPISP shader),
- a per-camera `PPISPController` MLP shader prim wired to the intermediate
  `ControllerParams` RenderVar,
- a per-camera _automatic-parameter_ PPISP shader (variant of the static
  SPG shader, reads exposure / color from the controller output instead of
  from USD attributes).

Controller exports package generated per-camera `ppisp_controller_<camera>.cu`
/ `.cu.lua` sidecars plus the `ppisp_usd_spg_auto.*` sidecar set. The
static `ppisp_usd_spg.*` set is not packaged in controller mode. Selection
is done by `select_spg_files_for_export`; the static and controller file
sets share no filenames.

Use `--disable-ppisp-controller-export` to force the static SPG fallback
path for a controller-trained checkpoint. Use
`--enable-ppisp-controller-export` to fail loudly if the loaded checkpoint
does not contain trained controllers.

### Incompatibility with `sh-optimized`

`--enable-ppisp-controller-export` and `--ppisp-integration-mode sh-optimized`
are mutually exclusive; the validator rejects the combination with a
message beginning:

```
enable_ppisp_controller_export is incompatible with ppisp_integration_mode='sh-optimized'
```

SH-match folds PPISP into the Gaussian SH coefficients, leaving no
per-frame work for the runtime controller. Use `spg-runtime` mode for
controller export.

### Static `spg-runtime` fallback table

When the controller is not present, or `--disable-ppisp-controller-export`
forces it off, the static `spg-runtime` path picks one of three authoring
strategies based on the reference IDs:

| `--ppisp-reference-camera-id` | `--ppisp-reference-frame-id` | Authored |
| --- | --- | --- |
| unset | unset | Time-sampled per-camera (one USD time sample per training frame) |
| set   | unset | Static, single-camera, neutral exposure and identity color latents |
| set   | set   | Static, single-frame: exposure / color latents pulled from the PPISP params at `frame_id` |
| unset | set   | Rejected — frame-only fixing has no camera axis to anchor vignetting/CRF to |

The fourth-row rejection message is:

```
ppisp_reference_frame_id was set without ppisp_reference_camera_id
in spg-runtime export mode. Frame-only fixing is ambiguous because
vignetting and CRF live on the camera axis.
```

## PPISP CLI / config reference

Mode-of-effect is one of `spg-runtime` (consumed by SPG shaders),
`sh-optimized` (consumed by the SH bake), or `both`.

### `--ppisp-integration-mode {spg-runtime,sh-optimized}`

- **Type**: choice • **Default**: `spg-runtime`
- **Mode-of-effect**: meta-flag (selects the path)
- **Effect**: picks the integration story — Story 1 (color matching)
  applies in either, Story 2 selects the bake, the default selects runtime
  SPG.
- **Rejection**: any other value →
  `Unsupported PPISP integration mode '<mode>'. Expected one of: ['spg-runtime', 'sh-optimized']`.

### `--scene-radiance-scale FLOAT`

- **Type**: float, strictly positive, finite • **Default**: `1.0` (no-op)
- **Mode-of-effect**: both
- **Effect**: multiplies every Gaussian's SH output before export.
  Permanent. Layered after the SH bake.
- **Rejection**: `0.0`, negatives, `inf`, `nan`, non-numeric →
  `... must be strictly positive, got ...` / `... must be finite, got ...`
  / `... must be a real number, got <type>`.

### `--ppisp-responsivity FLOAT`

- **Type**: float, strictly positive, finite • **Default**: `1.0` (no-op)
- **Mode-of-effect**: `spg-runtime` only (silent no-op in `sh-optimized`;
  see [Traps and gotchas](#--ppisp-responsivity-is-a-no-op-in-sh-optimized))
- **Effect**: authored verbatim on every PPISP shader's
  `inputs:responsivity` attribute. Multiplies the HDR buffer before PPISP.
  To preserve the post-ISP look while applying a radiance scale `k`, set
  this to `1/k`. Overridable downstream without re-export.
- **Rejection**: same as `--scene-radiance-scale`.

### `--ppisp-reference-camera-id INT`

- **Type**: int • **Default**: unset
- **Mode-of-effect**: both, with different semantics:
  - `spg-runtime`: pins the static export path to that camera. Setting the
    camera without a frame yields neutral-exposure / identity-color
    authoring (vignetting + CRF still come from the camera).
  - `sh-optimized`: selects the camera whose vignetting / CRF / exposure /
    color get folded into SH. Defaults to `0` if unset.
- **Rejection**: out of range → `... must be in [0, N-1], got <value>`.

### `--ppisp-reference-frame-id INT`

- **Type**: int • **Default**: unset
- **Mode-of-effect**: both, with the same dual semantics as
  `--ppisp-reference-camera-id` (see
  [fallback table](#static-spg-runtime-fallback-table)).
- **Rejection**: out of range, or set without a camera id in `spg-runtime`
  mode (fourth row of the fallback table).

### `--enable-ppisp-controller-export` / `--disable-ppisp-controller-export`

- **Type**: optional boolean pair • **Default**: auto
- **Mode-of-effect**: `spg-runtime` only
- **Effect**: when omitted, exports the controller path whenever the loaded
  PPISP module was trained with controllers. `--enable` forces the
  controller path and fails if trained controllers are absent. `--disable`
  forces the static SPG path.
- **Rejection**: combined with `--ppisp-integration-mode sh-optimized`
  (see [Story 3](#incompatibility-with-sh-optimized)).

### `--sh-optimization-num-iterations INT`

- **Type**: int, strictly positive • **Default**: unset (⇒ `3000`)
- **Mode-of-effect**: `sh-optimized` only
- **Effect**: overrides the SH-bake step budget.
- **Rejection**: `0`, negatives, floats, bools.

## Authored USD surface

This section enumerates the prims and attributes the PPISP export authors,
split into **standard USD** (works on any USD runtime) and **Omniverse /
RTX-specific** (only meaningful on SPG-capable viewers). On non-OV runtimes
the latter are inert custom data; consumers can ignore them safely.

### Standard USD

- `UsdVol ParticleField3DGaussianSplat` (or `UsdGeomPoints`) primitive
  carrying the Gaussian payload. Schema selected by `--format`.
- `UsdLux DomeLight` for the trained sky environment, when one exists.
- The original rig `Camera` prims, untouched.

### Omniverse / RTX-specific

The following are _not_ part of Pixar/standard USD; they are conventions
consumed by Omniverse's RTX renderer and SPG runtime
([`ppisp_writer.py`](ppisp_writer.py),
[`ppisp_controller_writer.py`](ppisp_controller_writer.py),
[`../writers/render_product.py`](../writers/render_product.py)).

- **SPG sidecar files** packaged into the USDZ archive next to the USD
  layers (loaded from [`ppisp_spg/`](ppisp_spg/)):
  - Static path: `ppisp_usd_spg.cu`, `ppisp_usd_spg.cu.lua`,
    `ppisp_usd_spg.usda`.
  - Controller path: the static set is replaced by generated per-camera
    `ppisp_controller_<camera>.cu` / `.cu.lua` files with controller weights
    embedded in CUDA device constants, plus `ppisp_usd_spg_auto.cu`,
    `ppisp_usd_spg_auto.cu.lua`, and `ppisp_usd_spg_auto.usda`. The static
    and controller file sets share no filenames.
- **PPISP Shader prims** authored on every RenderProduct:
  - Read from the `HdrColor` RenderVar (`omni:rtx:aov` connection).
  - Write to the `PPISPColor` RenderVar.
  - The viewer's display RenderVar `LdrColor` is rewired to `PPISPColor`.
  - Inputs (`inputs:responsivity`, per-camera vignetting / CRF, per-frame
    exposure / color latents) are authored as USD _connections_ to the
    `<cam>_ppisp.ppisp:*` attributes rather than literal values, so the
    shader pulls parameters from whichever camera the RenderProduct points
    at.
- **`<cam>_ppisp` camera**: a hidden sibling of each rig camera that
  inherits its intrinsics, pins the `exposure:*` namespace to neutral
  values, and carries every PPISP parameter as a `ppisp:*` attribute (the
  source of truth the shader inputs connect to). The RenderProduct's
  `camera` relationship is rewired to it; the user-facing rig camera is
  untouched. See
  [Why the camera and the exposure overrides](#why-the-cam_ppisp-camera-and-the-exposure-overrides).
- **`PPISPControllerPool` / `PPISPController` shader prims** (controller
  path only): per-camera CNN/pool and MLP stages, wired to the
  auto-parameter PPISP shader's exposure / color inputs through the
  `ControllerParams` AOV. Both carry `inputs:tileCountX/Y` constant inputs
  for tiled RenderProduct atlases. `PPISPControllerPool` additionally
  carries `inputs:responsivity` (mirrors `--ppisp-responsivity`).

## Traps and gotchas

### Why the `<cam>_ppisp` camera and the `exposure:*` overrides

Without intervention, Omniverse Kit applies its own exposure / shutter /
ISO model to the camera bound to a RenderProduct _before_ the SPG PPISP
shader runs. PPISP itself applies an exposure stage, so the two stages
compose multiplicatively and the asset looks roughly two stops too dark in
default Kit settings.

The fix authors a hidden `<cam>_ppisp` sibling of the rig camera
(inheriting its intrinsics via a path-based `inherits` arc), neutralizes
the entire `exposure:*` namespace on it (`exposure = 0.0`,
`exposure:fStop = 1.0`, `exposure:iso = 100.0`, `exposure:responsivity = 1.0`,
`exposure:time = 1.0`, `visibility = invisible`), then rewires the
RenderProduct's `camera` relationship to it. The rig camera prim itself is
untouched. This `<cam>_ppisp` camera also carries the PPISP parameters as
`ppisp:*` attributes that the shader inputs connect to.

### `--ppisp-responsivity` is a no-op in `sh-optimized`

The flag is wired to the PPISP shader's `inputs:responsivity` attribute; in
`sh-optimized` mode no PPISP shader is authored, so nothing reads it.
Validation does _not_ currently reject a non-default value in
`sh-optimized` mode. Concretely, the "set `responsivity = 1/k` to preserve
the look after a radiance rescale" recipe from
[Story 1](#story-1--color-matching-for-usd-integration) does not apply to
`sh-optimized` exports. Workaround: in `sh-optimized` exports, treat
`--scene-radiance-scale` as a "match-and-accept-the-look-shift" knob and
leave `--ppisp-responsivity` at `1.0`.

### `--ppisp-reference-frame-id` requires a camera id in `spg-runtime`

PPISP's per-frame parameters (exposure, color) are scoped _under_ a camera;
vignetting and CRF are pure camera attributes. Fixing just the frame would
leave the camera axis time-sampled or defaulted, making the static-frame
look non-deterministic across viewers. The exporter rejects this loudly
(see the [fallback table](#static-spg-runtime-fallback-table)).

### Controller export and `sh-optimized` are mutually exclusive

Folding PPISP into SH leaves nothing for the controller to do at runtime,
so exporting both would ship dead weights. The validator rejects explicit
`--enable-ppisp-controller-export` with `--ppisp-integration-mode sh-optimized`.

### SH bake runs before radiance scaling

Order matters: scaling is multiplicative on SH, so applying it before the
bake would cause the optimization to undo it (the reference is rendered
from a deepcopy taken _before_ the scale, so the loss would push the live
SH back to the unscaled values). The orchestrator runs the SH bake first,
then applies `--scene-radiance-scale`.

### Only `num_iterations` is CLI-tunable on the SH bake

The Adam learning rates (`features_albedo` `2.5e-3`, `features_specular` =
albedo / 20, `density` `5e-2`) and the optimization device are tuned for
the algorithm; surfacing them as CLI flags would invite mis-tuning. Python
callers that need to override them (e.g. tests forcing `device="cpu"`) pass
them as keyword arguments to `bake_post_processing_into_sh`.

### Controller export and embedded weights

The controller's CNN/MLP weights are authored directly into generated
per-camera CUDA sources. This avoids a per-frame binary weight upload and
avoids packaging any external weight sidecar, but downstream USD viewers
still need CUDA SPG support that can resolve and compile the generated
`ppisp_controller_<camera>.cu` files. Use `--disable-ppisp-controller-export`
when the target viewer should receive the static SPG shader instead.

## Troubleshooting

| Error message | Cause | Fix |
| --- | --- | --- |
| `Unsupported PPISP integration mode '...'. Expected one of: ['spg-runtime', 'sh-optimized']` | Typo in `--ppisp-integration-mode`. | Use `spg-runtime` or `sh-optimized`. |
| `... must be strictly positive, got ...` | Passed `0.0` or a negative to a scale/responsivity flag. | Use a strictly positive float. |
| `... must be finite, got ...` | Passed `inf` or `nan`. | Use a finite float. |
| `... must be a real number, got <type>` | Passed a non-numeric value. | Pass a float. |
| `enable_ppisp_controller_export is incompatible with ppisp_integration_mode='sh-optimized'` | Explicit controller export requested with `sh-optimized`. | Drop `--enable-ppisp-controller-export`, or switch mode. |
| `num_iterations must be int, got ...` / `num_iterations must be >= 1, got ...` | `--sh-optimization-num-iterations` got a float/bool or a non-positive int. | Pass a positive int. |
| `camera_id must be in [0, N-1], got V` / `frame_id must be in [0, N-1], got V` | Out-of-range reference id. | Use a valid index. |
| `ppisp_reference_frame_id was set without ppisp_reference_camera_id in spg-runtime export mode. ...` | Frame-only fixing in `spg-runtime`. | Set `--ppisp-reference-camera-id` too, or drop the frame id. |

If PPISP looks wrong on novel views in controller mode, the most likely
cause is a generated CUDA controller sidecar not being resolved or compiled
by the target viewer. Fall back to the static path with
`--disable-ppisp-controller-export`.

## Best practices

- **SPG-capable viewers (Omniverse / RTX)**: default mode (`spg-runtime`).
  Controller-trained checkpoints use the controller path by default;
  non-controller checkpoints use the static SPG path. Do not set reference
  IDs unless you need a single-frame static look on the static path.
- **Non-SPG viewers**: `sh-optimized` mode with reference IDs picking the
  look you want baked.
- **Asset radiance does not match the target renderer's scale**: set
  `--scene-radiance-scale k`, then `--ppisp-responsivity 1/k` if you want
  the post-ISP look unchanged. Drop the `1/k` if the rescale should also
  drive a look change.
- **Per-shot grading expected**: ship the asset with the
  `--ppisp-responsivity` value matching the radiance scale, and let
  consumers override `inputs:responsivity` in USD per shot (no re-export
  needed).
- **SH bake feels under-converged**: bump `--sh-optimization-num-iterations`
  to e.g. `6000`. Going much higher rarely helps because the learning rates
  are tuned for the 3000-step reference setup.
