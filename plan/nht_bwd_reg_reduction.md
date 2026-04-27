# Plan — NHT backward register-pressure reduction

## Context

`renderBackward` (NHT + PARTICLE_FEATURE_HALF=1, FEATURE_OUTPUT_HALF=1) is 60 ms
while fwd is 4 ms (15×). Two diagnostics established the root cause:

1. **A/B: commenting the body of `featuresIntegrateBwdToLocalGrad` changes nothing.**
   Commenting `featureLocalGradWarpReduceAndWrite` makes the kernel 2× faster.
   This is not an atomic-contention signal (reference
   `RasterizeToPixelsFromWorldNHT3DGSBwd.cu` uses the same 48-atomic pattern
   and is fast) — it is DCE freeing register pressure upstream.

2. **ptxas stats (sm_90):**

   | kernel           | regs/thread | spills | smem      | barriers |
   |------------------|-------------|--------|-----------|----------|
   | `renderBackward` | **214**     | 0      | 17 408 B  | 1        |
   | `render` (fwd)   | 64          | 0      | 17 408 B  | 1        |

   Actual `BlockSize = 16 × 16 = 256` (see `gutRendererParameters.h`), not 128
   as I first assumed. With 65 536 regs/SM on sm_90, reg cliffs at 256
   threads/block are:
   - 255 regs → 1 block/SM  (at 214 today → 1 block/SM, occupancy ~12.5 %)
   - 128 regs → 2 blocks/SM (25 %)
   -  85 regs → 3 blocks/SM (~37 %)
   -  64 regs → 4 blocks/SM (50 %)

   At 214 regs the kernel fits ~1 block/SM → effective occupancy ~12.5 %,
   memory latency cannot be hidden.
   (If `FINE_GRAINED_LOAD_BALANCING=true`, blocks are 128 thr — shifts the
   cliffs: 128r→4blk, 96r→5blk, 64r→8blk. Confirm which path is active.)

## Goal

Drive `renderBackward` regs/thread below **128** (ideally **≤ 96**) while
keeping gradients bit-equivalent to the current CUDA-integrated path
(compile-time toggle `NHT_FEATURES_BWD_LOCAL_GRAD_CUDA=1`).

Measurable success criteria:
- Primary:   `renderBackward` wall time reduced ≥ 25 % on the current scene.
- Secondary: regs/thread reported by ptxas ≤ 128.
- Parity:    feature/density gradient L2 vs pre-change reference < 1e-6 rel.

Non-goal: architectural rework of the renderer or Slang call surface.

## Task list (strictly ordered, each task independently reversible)

### T1 — Attribution of the register budget

Goal: measure how much each piece of live state costs in regs, so we know
which fix is worth pursuing. Pure diagnostic, no behavioral change.

- **T1.a**: build with `-Xptxas=-v -res-usage` permanently enabled behind an
  env var (`export NHT_PTXAS_VERBOSE=1`) via `setup_3dgut.py`. Archive
  the current 214-reg baseline.
- **T1.b**: temporarily gut `featureLocalGradWarpReduceAndWrite`'s body
  (keep signature, compile-time `#if 0` the shuffles + atomics). Rebuild.
  Record new regs + ms.
- **T1.c**: additionally gut our `featuresIntegrateBwdToLocalGrad` body.
  Rebuild. Record.
- **T1.d**: additionally replace the Slang `densityProcessHitBwdToBuffer`
  call with a no-op. Rebuild. Record.

Deliverable: a 5-row table (baseline + T1.a .. T1.d) of (regs, smem, ms).
Tells us exactly where the 214 regs are parked.

### T2 — Target fix 1: stage `featureLocalGrad[]` into `__shared__`

Hypothesis: moving the 48-float per-thread scratch out of registers and
into per-thread shmem slots will free up to 48 regs/thread, with negligible
runtime overhead (shmem LD/ST ≈ register throughput for small arrays).

- **T2.a** (implementation): add a `__shared__ float sFeatureLocalGrad[BlockSize][PARTICLE_FEATURE_DIM]`
  in the KBuffer renderer's bwd path. Change the caller site in
  `gutKBufferRenderer.cuh` to pass `sFeatureLocalGrad[tileThreadIdx]`
  (plus a clear of that row at top of each `j` iteration).
- **T2.b** (integrate): update the callee signature in
  `shRadiativeGaussianParticles.cuh::featuresIntegrateBwdToLocalGrad` and
  in `threedgut::nht::featuresIntegrateBwdToLocalGrad` (already takes a
  `float*`, so just documentation + assume shmem aliasing).
- **T2.c** (reduce): rewrite `featureLocalGradWarpReduceAndWrite` to reduce
  from shmem rather than thread-local registers. Verify `__shfl_xor_sync`
  still works (it does — operates on any register value; we first load
  shmem row into a single thread-private scalar per iteration).
- **T2.d** (smem budget check): new smem = `17 408 + BlockSize × 48 × 4`
  bytes. For `BlockSize=256` that is +49 152 B → 66 560 B/block. H100 has
  228 KB dynamic smem/SM → 3 blocks/SM by smem. Fine only if we hit ≥ 2
  blocks/SM by regs too (i.e. reg cliff ≤ 128). Record.
- **T2.e** (bench): rebuild, capture ptxas regs and `renderBackward` ms.
  Table row.
- **T2.f** (parity): run `validate.py` or equivalent; dump feature
  gradient buffer L2 vs baseline. Must be ≤ 1e-6 relative.

Expected: regs 214 → ~166 (–48). Probably still above 128 cliff; need T3.

### T3 — Target fix 2: `__launch_bounds__` + controlled spill

Hypothesis: if T2 alone does not cross the 128-reg cliff, force the
compiler to cap regs with `__launch_bounds__(BlockSize, minBlocksPerSM)`.
This may introduce local-memory spills, but occupancy gain beats spill
cost when kernel is latency-bound (our case).

- **T3.a**: apply `__launch_bounds__(256, 2)` to the `renderBackward`
  kernel entry in `gutRenderer.cuh`. Regs forced ≤ 128. Rebuild, record.
- **T3.b**: try `__launch_bounds__(256, 3)` (regs ≤ 85). Rebuild, record.
- **T3.c**: try `__launch_bounds__(256, 4)` (regs ≤ 64). Rebuild, record.
  (If `FINE_GRAINED_LOAD_BALANCING=true` the block size is 128; adjust
  the first arg accordingly.)
- **T3.d** (parity): only behavioral change is scheduling → gradients
  must be bit-identical. Confirm L2 = 0.

Table rows: regs, spill stores, spill loads, ms per configuration.

### T4 — Pick winner

Decision table from T2/T3 data:

| config         | regs | spills | ms    | parity | pick? |
|----------------|------|--------|-------|--------|-------|
| baseline       | 214  | 0      | 60    | ref    | —     |
| T2 (shmem)     | ?    | 0      | ?     | yes    | ?     |
| T3a (128,4)    | 128  | ?      | ?     | yes    | ?     |
| T3b (128,5)    | 96   | ?      | ?     | yes    | ?     |
| T2 + T3a       | ?    | ?      | ?     | yes    | ?     |
| T2 + T3b       | ?    | ?      | ?     | yes    | ?     |

Pick the lowest ms with spill stores low enough to not dominate L1 traffic.

### T5 — (conditional) Prefetch features into `__shared__` à la reference

Only if T4 falls short of the 25 % target. This is the
`RasterizeToPixelsFromWorldNHT3DGSBwd.cu` pattern: load the 48-float
feature block for all particles in the batch into shmem **once**, then
per-hit reads are shmem broadcasts. Cuts redundant global loads (currently
done twice per hit — inside `featuresFromBuffer` and again inside our
`featuresIntegrateBwdToLocalGrad`) and reduces register churn from callee
boundaries.

Bigger change: ~60 lines in `gutKBufferRenderer.cuh` inner batch loop,
plus new shmem size `BlockSize × 48 × sizeof(TFeatElem)` (= 12 KB @ fp16,
24 KB @ fp32). Defer until T4 signals it is needed.

### T6 — Validation & cleanup

- **T6.a**: final regs + ms comparison table.
- **T6.b**: full gradient parity on ≥ 2 training steps.
- **T6.c**: revert `NHT_PTXAS_VERBOSE` default (env-gated, stays available).
- **T6.d**: document the choice + knob(s) in `TODO_nht_cuda.md`.

## Test harness (used by every Tx)

One repeatable command invocation that captures both ptxas output and
kernel ms. Pseudo-code for `bench.sh`:

```
rm ~/.cache/torch_extensions/.../gutRenderer.cuda.o
NHT_PTXAS_VERBOSE=1 python validate.py --iters 50 --nsys ...
grep 'registers\|spill\|renderBackward' /tmp/build_and_run.log
```

Produces one row of the comparison table per Tx invocation.

## Confidence

- T1 (diagnostic): 95 %. Already have one data point, filling the matrix is mechanical.
- T2 alone buys 30–48 regs: 70 %.
- T2 alone reaches 128-reg cliff: 40 %.
- T3 forces the cliff and gains 25 %+: 60 %.
- Combined (T2+T3) reaches 50 % speedup: 50 %.

## Open questions (for your review before execution)

1. Is the validation harness OK as above, or do you have a preferred
   benchmarking script I should wire into `bench.sh`?
2. `BlockSize` — I assumed 128. Worth checking the actual tiling config
   used by the NHT path; the smem budget math depends on it.
3. Parity tolerance: 1e-6 relative is arbitrary. Tighter (bit-exact) is
   achievable for T2 and T3 since they do not change the math. Want me
   to require bit-exact?
4. If register pressure is mostly coming from the Slang-exported
   `densityProcessHitBwdToBuffer` (T1.d will tell), we may have to
   port that too. That is outside this plan — flag as follow-up if so.
