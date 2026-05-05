# PPISP Omniverse USD Post-Processing Fallback Plan

Scope: investigate whether the PPISP effect currently planned for SPG export can
also be approximated by existing Omniverse USD post-processing settings.

Goal: provide an Omniverse USD post-processing fallback for Kit versions where SPG is unavailable,
not supported in the target deployment, or affected by SPG bugs. This is not a
replacement for the exact SPG export path.

User-facing control: expose the fallback as an export parameter named
`ov-post-processing`. The implementation should live in a dedicated USD writer
file and stay separate from the SPG PPISP writer.

This document is intentionally a plan only. No implementation is approved here.

Status: `BLOCKED_ON_SPG_IMPLEMENTATION`

Implementation gate: the SPG PPISP export path is being implemented first. The
`ov-post-processing` fallback should be implemented on top of that work, after
the shared camera grouping and `/Render`/`RenderProduct` authoring are available.

---

## 1. Context

The current PPISP USD export plan in `docs/ppisp-controller-export-plan.md` uses a custom
SPG shader on each `RenderProduct` because PPISP is a post-blend image-space
operator:

1. Exposure: per-frame scalar `rgb *= 2**e`.
2. Vignetting: per-camera, per-channel, per-pixel multiplicative falloff.
3. Color correction: per-frame 3x3 homography in RGI space with intensity
   renormalisation.
4. CRF: per-camera, per-channel 4-parameter toe/shoulder/gamma curve.

The question here is whether a secondary PPISP export path can map enough of the
effect onto existing Omniverse USD post-processing controls to be useful when
SPG is not viable in a given Kit runtime.

Investigation target:

- Kit rendering post-processing implementation.
- Generated USD render-settings schema.
- Existing USD stages that author post-processing attributes on `RenderProduct`
  prims.

---

## 2. USD post-processing surface found

Kit exposes post-processing in two layers.

The C++ renderer layer:

- `Postprocessing::addPostprocessing`
- `Postprocessing::addTonemapping`
- `Postprocessing::addTvNoise`
- `Postprocessing::addRegisteredCompositing`

The USD/render-settings layer:

- `RenderProduct` prims can apply post-processing settings API schemas.
- Example stages author post-processing attributes directly on
  `/Render/<RenderProduct>`.
- The generated schema exposes:
  - camera exposure settings.
  - tonemapping settings.
  - color grading settings.
  - vignette settings.

Relevant setting families:

- Camera exposure:
  - `exposure:time`
  - `exposure:fStop`
  - `exposure:iso`
  - `exposure:responsivity`
- Tonemapping:
  - tonemap operator.
  - tonemap dither.
  - advanced carb-backed controls such as `exposureKey`, `whiteScale`,
    `maxWhiteLuminance`, `whitepoint`, and `enableSrgbToGamma`
- Color grading:
  - grade enabled.
  - `blackPoint`, `whitePoint`, `contrast`, `lift`, `gain`, `multiply`,
    `offset`, `gamma`, `saturation`
- TV-noise vignette:
  - effect enabled.
  - vignetting enabled.
  - vignetting size.
  - vignetting strength.

Important observed shader behavior:

- Tonemapping applies exposure through `computeExposureScale`, then one of the
  built-in operators: raw/clamp, linear, Reinhard, modified Reinhard,
  Hejl-Hable, Hable UC2, ACES approximation, or Iray Reinhard.
- Color correction/grading can run before tonemapping in ACES mode or after
  tonemapping in Standard mode.
- TV-noise vignetting uses one scalar radial-ish function:
  `pow(uv.x * (1 - uv.x) * uv.y * (1 - uv.y) * (size + 14), strength)`.

---

## 3. Mapping assessment

### 3.1 Exposure

Assessment: exact scalar mapping is likely possible.

Reasoning:

- PPISP exposure is a scalar multiply by `2**exposure_params[frame_idx]`.
- USD exposure scale is proportional to `exposure:time`, `filmIso`, and
  `responsivity`, and inversely proportional to `fStop**2`.
- If all other exposure parameters are held fixed, time-sampling
  `exposure:time` as `baseExposureTime * 2**e` should reproduce the scalar
  exposure factor.

Recommended USD mapping:

- Apply the camera exposure API schema to each camera prim, or author the
  equivalent camera exposure attributes if already supported by the target USD
  version.
- Time-sample `exposure:time` per frame.
- Keep `exposure:fStop`, `exposure:iso`, and `exposure:responsivity` fixed.
- Disable auto exposure and histogram adaptation for validation.

Risks:

- Exposure is embedded in tonemapping. Exactness only holds if the rest
  of the tone pipeline is configured so the exposure scale is not folded into a
  different nonlinear look.
- Kit has a Gaussian-specific skip-tonemapping path. If active, it may bypass
  the tone pass for Gaussian primary hits and therefore bypass the intended
  exposure mapping.

Confidence: 0.75.

### 3.2 Vignetting

Assessment: approximate only.

Reasoning:

- PPISP vignetting is per-camera, per-channel, and parameterized by five values
  per channel.
- USD vignette control is a scalar function shared across RGB, controlled by
  only `size` and `strength`.
- The vignette function is centered in normalized screen space and does not expose
  per-channel coefficients or arbitrary polynomial/radial terms.

Recommended USD mapping:

- Use the USD vignette API schema only for an approximation path.
- Enable TV noise and vignetting, but disable film grain, scanlines, ghosting,
  scrolling, random splotches, wave distortion, vertical lines, and flicker.
- Fit one scalar vignette curve per physical camera to the luminance average
  of the PPISP RGB vignette map.
- Record the per-channel residual, because color-dependent vignetting cannot be
  represented by this scalar control.

Risks:

- The TV-noise vignette pass is semantically part of an analog TV effect, not a
  calibrated camera response model.
- It may run after tonemapping, while PPISP vignetting is before color
  correction and CRF. This changes the result when later nonlinear operations
  are enabled.

Confidence: 0.45.

### 3.3 Color Correction

Assessment: approximate only, and likely weak for scenes with cross-channel
mixing.

Reasoning:

- PPISP color correction is a per-frame 3x3 homography in RGI space with
  intensity renormalisation.
- USD color correction and color grading expose channel-wise saturation,
  contrast, gain, gamma, offset, lift, multiply, black point, and white point.
- These controls do not expose a general 3x3 matrix or a homography with
  intensity renormalisation.

Recommended USD mapping:

- Prefer the USD color-grading API schema over legacy color-correction carb
  settings because it is present in the generated USD schema and examples.
- Use Standard mode for validation if the desired fit is after a linear
  tonemap, and ACES mode only if validation shows the color space conversion is
  closer to PPISP's RGI-space transform.
- Fit per-frame `gain`, `offset`, `gamma`, `contrast`, and `saturation` to
  sampled RGB pairs generated by the trained PPISP transform.
- Treat any off-diagonal color coupling in the PPISP homography as residual
  error, not as exportable data.

Risks:

- The generated USD schema exposes color grading attributes, but not every
  advanced carb setting is necessarily intended for portable USD authoring.
- Time-sampled `RenderProduct` attributes should be verified in Kit, because the
  schema examples are mostly static.

Confidence: 0.35.

### 3.4 CRF

Assessment: no exact mapping in existing USD post-processing.

Reasoning:

- PPISP CRF is per-camera, per-channel, and has four learned parameters per
  channel.
- USD tonemapping provides a small set of global operators. Iray Reinhard adds
  `crushBlacks`, `burnHighlights`, and saturation, but not per-channel
  toe/shoulder/gamma parameters.
- USD color grading `gamma` is per-channel, but it is not a learned
  toe/shoulder CRF.

Recommended USD mapping:

- Use the built-in tonemapper only as a coarse approximation.
- Evaluate two candidate fits:
  - `operator = "none"` or `"raw"` plus color grading gamma/gain/offset.
  - `operator = "iray"` plus Iray Reinhard crush/burn controls and color
    grading compensation.
- Fit per camera, not per frame, because PPISP CRF is per camera.

Risks:

- A fitted USD tonemapper may interact with exposure and color grading in ways
  that make individual PPISP components hard to validate independently.
- Per-channel CRF differences are not representable by global tonemap
  operators.

Confidence: 0.25.

---

## 4. Candidate architectures

### Option R0 — SPG-only export

Keep the SPG plan from `docs/ppisp-controller-export-plan.md` as the only PPISP-preserving
export path.

Use USD post-processing only for user-authored artistic settings unrelated to
PPISP.

Recommendation: best default path when the target Kit version has reliable SPG
support.

### Option R1 — Exposure-only USD fallback

Export only PPISP exposure through time-sampled camera exposure attributes.
Leave vignetting, color correction, and CRF unexported or keep them in SPG.

Recommendation: useful as the lowest-risk fallback for older Kit versions where
SPG is unavailable but some PPISP brightness matching is better than no PPISP
signal.

### Option R2 — USD post-processing fallback

Fit the full PPISP effect into existing USD settings:

- Exposure via `exposure:time`.
- Vignetting via TV-noise vignette.
- Color correction via color grading.
- CRF via tonemap plus color grading.

Recommendation: primary USD fallback candidate for Kit versions with no SPG
support or known SPG bugs. It should be advertised as approximate and version
gated.

### Option R3 — Hybrid export for validation and migration

Export both:

- Exact PPISP SPG `RenderVar` path for validation and high fidelity.
- Approximate USD post-processing attributes for viewers that do not support the
  custom SPG shader.

Recommendation: best investigation mode when validating the USD fallback against
SPG in newer Kit versions, or when the same asset must run across mixed Kit
deployments.

---

## 5. Proposed reviewable tasks

### T-R0 — Build a PPISP reference response sampler

Purpose: create a test-only numeric reference for comparing PPISP against USD
approximations.

Inputs:

- A trained or synthetic PPISP instance.
- A small set of RGB sample grids.
- Camera index and frame index.

Output:

- Per-stage reference outputs:
  - after exposure
  - after vignetting
  - after color correction
  - after CRF
  - final

Test write-up:

- Use identity PPISP parameters and assert output equals input.
- Enable only exposure and assert output equals `rgb * 2**e`.
- Enable one non-identity operation at a time and store deterministic numeric
  fixtures.

### T-R1 — Validate USD exposure equivalence

Purpose: prove whether `exposure:time = baseExposureTime * 2**e` matches PPISP
exposure under controlled USD settings.

Test write-up:

- Create a USD stage with one camera and one `RenderProduct`.
- Disable auto exposure, dither, color grading, TV noise, and nonlinear
  tonemapping.
- Render a known flat-color target at several exposure values.
- Compare captured output ratios against `2**e`.
- Repeat with Gaussian skip-tonemapping enabled and disabled.

Pass criterion:

- Relative error below a chosen tolerance, proposed initial threshold: `1e-3`
  for linear floating-point captures.

### T-R2 — Fit and validate USD vignette

Purpose: quantify how close the built-in USD vignette can get to PPISP
vignetting.

Test write-up:

- Generate PPISP vignetting maps for each camera.
- Fit `vignetting:size` and `vignetting:strength` to the luminance-average
  PPISP map.
- Render a flat-color image through the vignette pass with all
  other TV effects disabled.
- Compare spatial error and per-channel residual.

Pass criterion:

- Report RMSE and max error. Do not enforce pass/fail until real datasets are
  sampled.

### T-R3 — Fit USD color grading to PPISP color correction

Purpose: determine whether the color grading controls can approximate the PPISP
3x3 RGI homography acceptably.

Test write-up:

- Sample RGB values across the training color range.
- Apply PPISP color correction for selected frames.
- Fit USD grade controls to minimize color error.
- Validate on held-out RGB samples and on rendered frames.

Pass criterion:

- Report `meanDeltaRgb`, `p95DeltaRgb`, and max channel error.
- Flag frames where off-diagonal homography terms dominate the residual.

### T-R4 — Fit USD tonemap/grade to PPISP CRF

Purpose: quantify CRF approximation quality with built-in USD tone operators.

Test write-up:

- For each camera, sample the PPISP per-channel CRF curves.
- Fit candidate USD settings:
  - raw or none tonemap plus grade gamma/gain/offset
  - Iray Reinhard plus grade compensation
- Validate per-channel curve error and final image error.

Pass criterion:

- Report per-camera curve RMSE and max error.
- Reject USD-only export for cameras whose CRF fit exceeds the selected
  threshold.

### T-R5 — Author a minimal USD post-processing prototype

Purpose: verify the USD authoring model without touching the production exporter.

Expected prototype shape:

- `/Render/<cameraName>` `RenderProduct`
- Applied schemas:
  - tonemapping API schema.
  - color-grading API schema.
  - vignette API schema.
- Camera exposure attributes on the referenced camera prim.
- `orderedVars` containing `LdrColor`.

Test write-up:

- Open the generated USD in Kit with the required render-settings schema enabled.
- Verify authored attributes appear in the active render settings context.
- Capture output and compare against the PPISP reference sampler.

### T-R6 — Define USD fallback policy

Purpose: choose when the USD post-processing fallback should be offered after
the numeric validation tasks.

Decision points:

- Identify the minimum Kit version where SPG is reliable enough to prefer
  `spgExact`.
- Identify older Kit versions or known SPG bug IDs where the USD fallback should
  be available.
- If only exposure is accurate, add an exposure-only USD fallback mode.
- If fitted error is acceptable for target datasets, add an approximate USD
  post-processing fallback mode.
- If errors are high, keep USD post-processing as an explicit degraded fallback
  and document SPG as required for fidelity.

Test write-up:

- Produce a short validation report with per-stage errors and example captures.
- Require explicit approval before implementing any exporter changes.

---

## 6. Feasibility Report

Assessment: feasible with moderate implementation risk.

The standard USD exporter already has the right integration points:

- `configs/base_gs.yaml` contains an `export_usd` block for export parameters.
- `USDExporter.from_config` centralizes conversion from config to exporter
  constructor arguments.
- `USDExporter.export` already has access to `model`, `dataset`, `conf`, and
  `background`.
- `trainer.py` already chooses `USDExporter` when `export_usd.format` is
  `standard`.

The main missing dependency is not the export parameter itself, but access to the
trained PPISP module during USD export. Today `trainer.py` calls:

```text
exporter.export(..., dataset=self.train_dataset, conf=conf, background=...)
```

For any PPISP-derived fallback, this call must also pass
`post_processing=self.post_processing` when `post_processing.method == "ppisp"`.
That is already required by the SPG export plan, so the fallback should reuse
the same exporter-facing data path.

Recommended export parameters:

```yaml
export_usd:
  export_ppisp: false
  ov-post-processing: none
```

`export_ppisp` is the gate for PPISP export. If it is `false`, no PPISP effect
is exported and `ov-post-processing` must be `none`.

Allowed `ov-post-processing` values when `export_ppisp` is `true`:

- `none`: use the full SPG PPISP path and do not author fallback settings.
- `ppisp-exposure-fallback`: export only PPISP exposure through USD camera exposure.
- `ppisp-fitted-post-processing-fallback`: export the fitted USD post-processing approximation.
- `ppisp-spg-plus-fitted-post-processing-fallback`: author the fitted fallback attributes alongside the SPG
  path for validation or mixed Kit deployments.

Implementation note: because `ov-post-processing` is hyphenated, Python code
should read it with `export_conf.get("ov-post-processing", "none")`, not
`conf.export_usd.ov-post-processing` or normal dot access.

Dedicated file recommendation:

```text
threedgrut/export/usd/writers/ov_post_processing.py
```

Suggested public API:

```python
def add_ov_post_processing(
    stage,
    render_product_entries,
    post_processing,
    dataset,
    mode: str,
) -> None:
    ...
```

Responsibilities of the dedicated file:

- Validate that `mode` is one of the supported `ov-post-processing` values.
- Validate that `post_processing` is a PPISP instance for PPISP-derived modes.
- Author USD post-processing API schemas and attributes on `RenderProduct`
  prims.
- Author camera exposure attributes for the exposure fallback.
- Fit or consume fitted parameters for vignette, color grading, and CRF
  approximation.
- Log an explicit warning when falling back to degraded behavior.

Responsibilities that should stay outside the dedicated file:

- Camera prim creation.
- `/Render` scope and `RenderProduct` creation.
- SPG shader authoring and sidecar packaging.
- Exporter config parsing beyond passing the selected mode.

Feasibility by mode:

- `none`: high feasibility. Uses the existing full SPG PPISP path when
  `export_ppisp` is true.
- `ppisp-exposure-fallback`: high feasibility. Requires camera prims and time-sampled
  exposure authoring only.
- `ppisp-fitted-post-processing-fallback`: medium feasibility. USD authoring is straightforward, but
  fitting PPISP vignetting/color/CRF into USD controls needs validation and may
  have visible residuals.
- `ppisp-spg-plus-fitted-post-processing-fallback`: high feasibility after SPG and fallback paths both
  exist. It is mostly orchestration and validation.

Primary risks:

- Current `USDExporter` exports one camera per frame; the SPG plan already notes
  this must become one prim per physical camera with time-sampled transforms
  before per-camera `RenderProduct` post-processing can be cleanly authored.
- The current exporter does not create `/Render` or `RenderProduct` prims.
  The fallback depends on the same `render_product.py` foundation as the SPG
  plan.
- Time-sampled `RenderProduct` USD attributes need validation in the target Kit
  versions.
- The fallback is approximate by design. Documentation and logs must make this
  visible so users do not mistake it for SPG fidelity.

Overall recommendation: implement the feature as a small orchestrated extension
after the shared camera and `RenderProduct` groundwork from the SPG plan. Keep
`export_ppisp` disabled by default. When `export_ppisp` is enabled, use
`ov-post-processing` to choose between full SPG and explicit fallback modes.

Execution dependency: wait for the SPG implementation to land, then add the OV
post-processing writer as a follow-up layer that reuses the SPG path's camera,
time-code, and `RenderProduct` infrastructure.

Confidence: 0.8 for exporter/config feasibility, 0.45 for final visual fidelity
of the full PPISP approximation.

---

## 7. Recommended architecture if approved later

Use `export_ppisp` as the PPISP export gate and `ov-post-processing` as the
implementation selector:

```text
export_usd:
  export_ppisp: false
  ov-post-processing: none
```

Behavior:

- `export_ppisp: false`: no PPISP export. `ov-post-processing` must be `none`.
- `export_ppisp: true`, `ov-post-processing: none`: export PPISP through the
  full SPG path.
- `export_ppisp: true`, `ov-post-processing: ppisp-exposure-fallback`: export
  PPISP through camera exposure fallback only.
- `export_ppisp: true`, `ov-post-processing: ppisp-fitted-post-processing-fallback`:
  export PPISP through fitted Omniverse USD post-processing fallback only.
- `export_ppisp: true`, `ov-post-processing: ppisp-spg-plus-fitted-post-processing-fallback`:
  write both the SPG exact path and the fitted USD fallback attributes for
  validation or mixed-version deployment.

Keep the approximation code isolated from the exact SPG writer:

```text
threedgrut/export/usd/
    writers/
        ov_post_processing.py
        ppisp_writer.py
```

Rationale:

- The USD mapping is a fitted approximation, not a semantic equivalent of
  PPISP.
- The USD fallback exists for deployment compatibility with older or buggy Kit
  SPG support, not to displace the high-fidelity SPG path.
- Keeping a separate backend makes review easier and prevents silent quality
  regressions in the exact export path.

---

## 8. Open questions

- Which Kit versions need the USD fallback because SPG is unavailable?
- Which known SPG bugs should trigger or recommend the USD fallback path?
- What error threshold is acceptable for a degraded fallback export?
- Should the approximation target linear floating-point `LdrColor`, gamma
  output, or Kit viewport screenshots?
- Are time-sampled `RenderProduct` post-processing attributes supported and
  stable in the target Kit version?
- Should Gaussian skip-tonemapping be disabled for PPISP USD approximation, or
  is the exported Gaussian material already authored for that path?
---

## 9. Current recommendation

Use USD post-processing only as an explicit fallback alternative for older
Kit versions or known SPG failure modes.

Exact PPISP export should remain SPG-based for Kit versions where SPG is
available and reliable. The USD fallback path should be version-gated, labeled
approximate, and validated against SPG/reference PPISP before use on target
datasets.

Recommended next step: review this document and edit the task list or thresholds
before any implementation begins.
