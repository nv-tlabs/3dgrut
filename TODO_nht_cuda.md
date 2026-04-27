# Handwritten CUDA port of `featuresIntegrateBwdToLocalGrad` (NHT path)

## Status
- [x] **T1** — Tetrahedron constants (`tetraV0`, `tetraN0..N3`) placed in
      `nht_detail` namespace at the top of `shRadiativeGaussianParticles.cuh`.
      Values derived from Slang's vertex ordering; verified via script
      (w_k == 1 at v_k, 0 at other vertices).
- [x] **T2** — Method body replaced, gated by `#if NHT_FEATURES_BWD_LOCAL_GRAD_CUDA`.
      Default = `1` (native CUDA). Flip to `0` in
      `threedgut_tracer/include/3dgut/kernels/cuda/models/shRadiativeGaussianParticles.cuh`
      to restore the Slang-autodiff path (kept unchanged in the `#else` branch).
- [ ] **T3** — Rebuild, run `validate.py` (or one training step) with the macro
      at 1 vs 0. Compare:
      - feature gradient buffer L2  (primary parity check)
      - density / position gradients (sanity; should be identical since we don't
        touch those paths)
      - `renderBackward` ms in nsys.
- [ ] **T4** — If parity holds, keep default = 1. Otherwise flip to 0 and iterate.

## What the handwritten CUDA does (semantics to match Slang exactly)

Replicates the sequence inside Slang's `particleFeaturesIntegrateBwdToBuffer`
called with `exclusiveGradient=true` and the shifted `featureLocalGrad` buffer:

1. Early-out when `alpha <= 0`.
2. Recover pre-hit accumulator:
   `acc_prev[i] = (integratedFeatures[i] - features[i]*alpha) / (1-alpha)`.
3. VJP of back-to-front `y_i = (1-alpha)*acc_prev_i + alpha*f_i` against
   incoming `dy = integratedFeaturesGrad`:
   - `dFeatures[i] = alpha * dy_i`
   - `alphaGrad += sum_i (features[i] - acc_prev[i]) * dy_i`
   - `integratedFeaturesGrad[i] = (1-alpha) * dy_i`  (new accumulator grad)
4. Barycentric weights `w[0..3]` from `canonicalIntersection` (Cramer form
   matching Slang, precomputed `N_k` face normals).
5. Load 4 vertex feature blocks × `InterpPointFeatureDim` once
   (`__half2float` when `PARTICLE_FEATURE_HALF=1`).
6. Activation backward → `dBase[InterpPointFeatureDim]`:
   | Activation | Forward | Backward |
   |---|---|---|
   | None (0) | `out = base` | `dBase = dFeatures` |
   | Siren (1) | `sin(base * 2^f)` | `dBase += cos(base*freq) * freq * dOut` |
   | Sincos (2) | `sin + cos` | `dBase += (cos - sin) * freq * dOut` |
   | Relu (3) | `max(0, base)` | `dBase = (features[i] > 0) ? dFeatures[i] : 0` |
7. Barycentric backward:
   - `featureLocalGrad[k*IPFD + i] += w[k] * dBase[i]`  (matches Slang's `+=` with exclusiveGradient=true)
   - `canonicalIntersectionGrad += sum_k (sum_i vert[k][i] * dBase[i]) * N_k`

## Guardrails
- `static_assert(FeatureTransformType == 1)` — NHT-only.
- `static_assert(FEATURE_INTERPOLATION_TYPE == 0)` — barycentric only.
- `static_assert(FEATURE_INTERPOLATION_SUPPORT == 1)` — tetrahedra only.
- `static_assert` on `RAY_FEATURE_DIM` / `INTERP_POINT_FEATURE_DIM` / activation consistency.
- `static_assert(4 * IPFD == ParticleFeatureDim)` — buffer layout.

Any unsupported config fails at compile time — fallback is to flip the macro to 0.

## Confidence

- **Forward parity** (interpolation + integration, current config `activation=relu`):
  high (see comparison with `neural-harmonic-textures/Interpolation.cuh` — same
  tetrahedron geometry, different indexing; same integration math).
- **Backward numerical parity**: medium-high. The Relu path is trivial. The
  (1-α)/α lerp VJP + barycentric VJP is standard. Main risk is a sign or
  vertex-index swap — covered by T3 gradient diff.
- **Perf win**: medium-high. Expected 3–5× on this single kernel.

## Open reference points

- Slang source:    `threedgut_tracer/include/3dgut/kernels/slang/models/neuralHarmonicFeaturesParticle.slang`
- External CUDA ref: `/nv/dev/neural-harmonic-textures/gsplat/gsplat/cuda/csrc/RasterizeToPixelsFromWorldNHT3DGSBwd.cu`
  (sincos activation; do NOT copy the activation bwd verbatim — see
  "Caveats" in the forward-parity discussion: Slang's sincos sums into one
  channel, ref's keeps them separate).
