# 3DGRT half-precision feature support

Goal: make `conf.render.particle_feature_half=true` and `conf.render.feature_output_half=true`
work end-to-end in `threedgrt_tracer`, matching the behavior already implemented in
`threedgut_tracer`. Gradient buffers remain fp32 on both paths.

Semantics (mirroring 3dgut):
- `particle_feature_half=true`: storage for `particleRadiance` (per-particle feature buffer)
  is fp16. Slang entry points already expect `feat_elem_t*` (`__half*` when the macro is set).
  Gradient `particleRadianceGrad` stays fp32.
- `feature_output_half=true`: storage for the per-ray integrated feature buffer (`rayRadiance`)
  is fp16. Gradient `rayRadianceGrad` stays fp32. Tracer `.forward()` casts fp16 back to fp32
  before returning, mirroring 3dgut.

## Scope

Files to touch (by layer):

- C++ pipeline type layer
  - `threedgrt_tracer/include/3dgrt/pipelineParameters.h`
    Introduce `TFeatureDensityElem` (output/ray feature) and `TParticleFeatureElem` (particle
    storage) typedefs, guarded on the two macros. Change `particleRadiance` from `const float*`
    to `const TParticleFeatureElem*`, and `rayRadiance` from
    `PackedTensorAccessor32<float, 4>` to `PackedTensorAccessor32<TFeatureDensityElem, 4>`.
    `particleRadianceGrad` and `rayRadianceGrad` stay fp32.
- OptiX raygen kernels
  - `threedgrt_tracer/src/kernels/cuda/referenceSlangOptix.cu`
  - `threedgrt_tracer/src/kernels/cuda/referenceSlangBwdOptix.cu`
    1. Replace `const_cast<float*>(params.particleRadiance)` with
       `const_cast<TParticleFeatureElem*>(params.particleRadiance)`.
    2. FWD write to `rayRadiance`: wrap with `__float2half` when `FEATURE_OUTPUT_HALF`.
    3. BWD read from `rayRadiance`: wrap with `__half2float` when `FEATURE_OUTPUT_HALF`.
       `rayRadianceGrad` reads stay fp32.
- Host launcher
  - `threedgrt_tracer/src/optixTracer.cpp`
    1. `trace()`: allocate `rayRad` with `torch::kHalf` when `FEATURE_OUTPUT_HALF=1`,
       build `packed_accessor32<TFeatureDensityElem, 4>(rayRad)`, and
       `getPtr<const TParticleFeatureElem>(particleRadiance)`.
    2. `traceBwd()`: same dtype for the forward `rayRad` input; the grad tensors remain fp32.
- Python tracer
  - `threedgrt_tracer/tracer.py`
    1. Cast `gaussians.get_features()` to `.half()` when `conf.render.particle_feature_half`.
    2. Keep `ray_features.float()` return to caller; `ray_features` saved in ctx may be fp16
       when `feature_output_half=true` (already saves the raw output, consistent with 3dgut).

No changes required in Slang `.slang` or generated `.cuh`: the generalization already landed
and compiles correctly once `SLANG_CUDA_ENABLE_HALF=1` is set (done).

## Task breakdown

Each task is independently reviewable and testable (run validate.py for the relevant flag
combinations after each).

### T1 — Introduce typedefs in `pipelineParameters.h`
- Add `TFeatureDensityElem` and `TParticleFeatureElem` (guarded by `FEATURE_OUTPUT_HALF` and
  `PARTICLE_FEATURE_HALF`), include `cuda_fp16.h` when either is set.
- Change `particleRadiance` to `const TParticleFeatureElem*` and `rayRadiance` accessor to
  `PackedTensorAccessor32<TFeatureDensityElem, 4>`.
- No functional change when both macros are 0 (typedefs resolve to `float`).
- Test: build with both flags false (current default) → no-op rebuild; CI NeRF-Synthetic 3dgrt
  smoke test still passes.

### T2 — Update OptiX kernels for fp16 reads/writes
- Apply the `__float2half` / `__half2float` wrappers in `referenceSlangOptix.cu` and
  `referenceSlangBwdOptix.cu` under `FEATURE_OUTPUT_HALF`.
- Update `const_cast` sites to `TParticleFeatureElem*`.
- Test: build with both flags false → identical numerical output to baseline (no wrappers
  compiled in).

### T3 — Host buffer allocation and accessor typing
- `optixTracer.cpp`: select dtype `kHalf` vs `kFloat32` for `rayRad`; use
  `packed_accessor32<TFeatureDensityElem, 4>(rayRad)`.
- `getPtr<const TParticleFeatureElem>(particleRadiance)` for the particle buffer.
- Test: with flags false → unchanged; build-time assert that tensor dtype matches the
  typedef via `TORCH_CHECK(rayRad.scalar_type() == ...)` in DEBUG.

### T4 — Python cast for `particle_feature_half`
- `tracer.py`: mirror 3dgut's conditional `.half()` cast on `gaussians.get_features()`.
- Test: flags false → unchanged.

### T5 — End-to-end validation with flags enabled
- Run `validate.py` with `render.particle_feature_half=true render.feature_output_half=true`
  using an existing NHT config (e.g. `nerf_synthetic_3dgrt_mcmc_nht.yaml`).
- Compare PSNR after N iterations against the fp32 baseline — expected within 0.1 dB.
- Gradients: single backward pass on a fixed seed; check that
  `particleRadianceGrad` and `rayRadianceGrad` are finite and within tolerance of the
  fp32 reference.

### T6 — Rename `*Radiance*` → `*Features*` in 3dgrt
Naming cleanup to align with the post-SH NHT feature abstraction. The legacy `Radiance`
suffix comes from the SH-only era; the buffers now carry arbitrary per-particle / per-ray
features. Purely mechanical rename, no behavioral change. Runs AFTER T1–T5 land so we are
not also chasing name drift during the fp16 functional work.

Rename mapping (all scopes):
- `PipelineParameters::particleRadiance`       → `particleFeatures`
- `PipelineParameters::rayRadiance`            → `rayFeatures`
- `PipelineBackwardParameters::particleRadianceGrad` → `particleFeaturesGrad`
- `PipelineBackwardParameters::rayRadianceGrad`      → `rayFeaturesGrad`
- `OptixTracer::trace(..., torch::Tensor particleRadiance, ...)` arg                  → `particleFeatures`
- `OptixTracer::traceBwd(..., torch::Tensor particleRadiance, rayRad, rayRadGrd, ...)` args
  → `particleFeatures`, `rayFeat`, `rayFeatGrd` (local tensors + Python side kwargs).
- `particleRadianceGrad` local in `optixTracer.cpp::traceBwd` → `particleFeaturesGrad`.
- Python: `tracer.py` local variables `ray_features` / `ray_features_grd` are already
  feature-named; cross-check that the pybind11 binding signature in `bindings.cpp` uses
  the new C++ arg names.

Out of scope for T6 (per resolved decisions above):
- `particleRadianceSphDegree` C++ field and `conf.render.particle_radiance_sph_degree` YAML.
- `shRadiativeParticles.slang` filename and internal `shRadiance*` identifiers (SH path).
- Any `*Radiance*` identifiers that only exist on the SH-specific code path.

Test:
- Build + full `validate.py` run with fp32 flags (both false) → identical numerical
  output to pre-T6 baseline (bit-identical expected since only identifier renames).
- Build + `validate.py` with fp16 flags (both true) → identical output to T5 result.

## Tests to write up-front

- `tests/test_3dgrt_half_flags.py` (new, small)
  - Parametrize over `(particle_feature_half, feature_output_half) ∈ {(F,F),(T,F),(F,T),(T,T)}`.
  - Forward only, single frame, fixed scene; compare `pred_features.float()` to the (F,F)
    baseline with `atol=5e-3, rtol=1e-2`.
  - Forward + backward; compare `mog_sph.grad` to the (F,F) baseline at the same tolerance.

## Decisions (resolved with user)

1. T5 validation ownership: user runs validation; the plan only needs to keep the hooks in
   place (no tolerance tuning required from the implementer).
2. Gradient buffers stay fp32 end-to-end (no half-grad path).
3. T6 rename scope is restricted to identifiers naming buffers that can carry NHT features
   (i.e. the per-particle feature storage and per-ray integrated feature output, plus their
   fp32 gradients). Scalars and SH-specific paths are NOT renamed:
     - keep `particleRadianceSphDegree` (C++ field) and `conf.render.particle_radiance_sph_degree`
       (YAML) — scalar, shared with the SH path
     - keep `shRadiativeParticles.slang` filename and its internal `shRadiance*` identifiers —
       SH-only code path.
4. T6 runs AFTER T1–T5.

## Non-goals

- No changes to CUDA fallback path (`gaussianParticles.cuh`) — per the existing TODO that is
  a separate workstream.
- No changes to `threedgrt_playground`.
