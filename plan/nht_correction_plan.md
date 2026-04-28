# NHT Correction Plan

## Goal

Close the correctness gaps found in the NHT branch review against the GSplat NHT reference, without changing SH baseline defaults.

## P0 - Correctness Bugs

### T1 - Match Forward Depth Gating In 3DGUT No-K-Buffer Backward

- File: `threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutKBufferRenderer.cuh`
- Problem: `evalBackwardNoKBuffer` calls `densityHit` but does not apply the same `hitT > ray.tMinMax.x && hitT < ray.tMinMax.y` gate used by forward.
- Change: add the depth-slab predicate around the backward hit processing path.
- Test:
  - Build/run a short NHT validation with default `k_buffer_size: 0`.
  - Add or run a targeted kernel smoke case where a Gaussian outside `ray.tMinMax` contributes no feature/density gradient.

### T2 - Use FeatureDecoder EMA For Final Test Evaluation

- Files: `threedgrut/trainer.py`, optionally `threedgrut/render.py`
- Problem: validation and checkpoint eval use EMA weights, but train-end `on_training_end()` test uses live decoder weights.
- Change: apply `feature_decoder.apply_ema_shadow()` around `Renderer.from_preloaded_model(...).render_all()` and restore after, mirroring validation and GSplat eval.
- Test:
  - Re-run checkpoint eval and train-end eval on the same checkpoint; metrics should agree within numerical noise.

### T3 - Do Not Apply EMA To Trainable Decoder On Resume

- File: `threedgrut/trainer.py`
- Problem: resume loads decoder EMA shadow and then copies EMA into live trainable parameters, while optimizer state still belongs to non-EMA weights.
- Change: load EMA shadow only; do not call `apply_ema_shadow()` in the resume path.
- Test:
  - Save a checkpoint with EMA, resume, and verify `feature_decoder.state_dict()` equals checkpoint `"module"` immediately after load.
  - Verify eval still uses EMA through the eval-only swap path.

## P1 - Metric And Config Semantics

### T4 - Fix `Renderer.render_all()` Extra-Metrics Guard

- File: `threedgrut/render.py`
- Problem: `compute_extra_metrics=False` omits SSIM/LPIPS criterions but `render_all()` always uses them.
- Change: either always construct metrics needed by the table, or guard SSIM/LPIPS/color-corrected metrics and output only PSNR when disabled.
- Test:
  - Run `Renderer.from_preloaded_model(..., compute_extra_metrics=False).render_all()` on a small checkpoint without `KeyError`.

### T5 - Correct Benchmark Result Table Units

- File: `plan/nht_reference_results.md`
- Problem: current 3DGRUT `2.455` / `2.489` values are `std_psnr`, not render time.
- Change: rename that column for 3DGRUT rows or replace with real `mean_inference_time_ms` from a timing-enabled eval.
- Test:
  - Confirm `metrics.json` and terminal table fields map one-to-one to the document columns.

### T6 - Add L2 Loss To Total Or Disable The Flag

- File: `threedgrut/trainer.py`
- Problem: `loss.use_l2` computes/logs L2 but does not affect `total_loss`.
- Change: add `lambda_l2 * loss_l2` to `total_loss`, or remove/mark unsupported if unused by intended configs.
- Test:
  - Unit/smoke check with `loss.use_l2=true`, `lambda_l1=lambda_ssim=lambda_opacity=lambda_scale=0`, verify nonzero `total_loss`.

### T7 - Split Decoder Weight Decay From Explicit Decoder Regularization

- Files: `configs/base_gs.yaml`, `threedgrut/trainer.py`, `threedgrut/model/feature_decoder.py`
- Problem: `nht_decoder.reg_weight` drives both Adam `weight_decay` and explicit `params^2` loss.
- Change: keep GSplat parity by using optimizer `weight_decay` only, or introduce separate config keys if both are desired.
- Test:
  - With default `reg_weight: 0.0`, behavior is unchanged.
  - With nonzero regularization, verify only the chosen mechanism contributes.

## P2 - Latent / Cleanup Items

### T8 - Make `rays_in_world_space` Consistent In Feature Decode

- File: `threedgrut/utils/render.py`
- Problem: tracer respects world-space rays, decoder always rotates directions by `T_to_world`.
- Change: if `gpu_batch.rays_in_world_space` is true, normalize `gpu_batch.rays_dir` directly.
- Test:
  - Create a small `Batch` with world-space rays and non-identity `T_to_world`; verify decoder direction input is not double-rotated.

### T9 - Remove Or Fully Reject Unsupported Bezier NHT Config

- File: `threedgrut/model/features.py`
- Problem: `interpolation_type` accepts `"bezier"` but `interpolation_support` and kernels do not support it.
- Change: raise a clear `NotImplementedError` for `"bezier"` at `interpolation_type`, or implement end-to-end later.
- Test:
  - Config with `model.nht_features.interpolation_type=bezier` fails early with a clear message.

### T10 - Clarify NHT Progressive Feature Bookkeeping

- Files: `threedgrut/model/model.py`, `threedgrut/trainer.py`, docs/comments
- Problem: `n_active_features` progresses for NHT but kernels do not use it to mask NHT feature dimensions.
- Change: disable progression for NHT or document it as unused and avoid misleading logs/exports.
- Test:
  - NHT training logs do not imply progressive feature activation unless implemented.

## P3 - Test Coverage

### T11 - Add NHT Smoke/Parity Tests

- Files: new tests or `validate.py`
- Problem: `validate.py` only checks that Python NHT symbols exist, not that CUDA NHT kernels work.
- Change:
  - Add a minimal NHT render/backward smoke test for 3DGUT.
  - Add shape checks for `particle_feature_dim`, `ray_feature_dim`, and sincos expansion.
  - Add a resume/EMA behavior check.
- Test:
  - Run the new tests in the `3dgrut-nht` environment.

## Suggested Order

1. T1, T2, T3: correctness and benchmark parity.
2. T4, T5, T6: metric/reporting correctness.
3. T7, T8, T9, T10: config semantics and latent bugs.
4. T11: regression coverage.

