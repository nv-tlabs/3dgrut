# TODO

## 3DGRT: half-precision particle features

`conf.render.particle_feature_half` is compiled into the 3DGRT kernel via `-DPARTICLE_FEATURE_HALF`
but the Python-side cast is missing. In `threedgrt_tracer/tracer.py`, `gaussians.get_features()`
must be cast to `.half()` before being passed to `_Autograd.apply` when the flag is set,
matching what 3DGUT already does.

See the `TODO` comment in `threedgrt_tracer/tracer.py`.

## 3DGRT: NHT support in CUDA path (`gaussianParticles.cuh`)

The NHT feature transform (`FEATURE_TRANSFORM_TYPE=1`) is implemented for the Slang path
(`gaussianParticles.slang`) but not yet in the CUDA path (`gaussianParticles.cuh`).
Full NHT support in 3DGRT requires extending `gaussianParticles.cuh` with the NHT
interpolation and activation logic currently only present in the Slang kernel.

## 3DGUT: refactor `evalBackwardNoKBuffer` to share path with k-buffer backward

`evalBackwardNoKBuffer` (`gutKBufferRenderer.cuh`) duplicates logic from the k-buffer backward
path. The two should be unified into a shared implementation to reduce code duplication and
ensure future fixes apply to both.
