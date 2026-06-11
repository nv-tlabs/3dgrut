# 3DGRUT Export

Export trained Gaussian models to interchange formats, and transcode between
them. This document covers the general export workflows; PPISP
post-processing export has its own deep dive in
[`usd/post_processing/README.md`](usd/post_processing/README.md).

## Contents

- [Formats](#formats)
- [Requirements](#requirements)
- [Exporting at the end of training](#exporting-at-the-end-of-training)
- [Standalone USD export](#standalone-usd-export)
- [Transcoding between formats](#transcoding-between-formats)
- [Converting PLY files to USDZ](#converting-ply-files-to-usdz)
- [Adding a mesh to a USDZ file](#adding-a-mesh-to-a-usdz-file)
- [PPISP post-processing export](#ppisp-post-processing-export)

## Formats

| Format | Schema | Values | Use |
| --- | --- | --- | --- |
| **PLY** | point cloud | pre-activation | Interchange with 3DGS tooling |
| **USD (standard / lightfield)** | `UsdVol ParticleField3DGaussianSplat` | post-activation | Native USD Gaussian asset for any USD-aware renderer |
| **NuRec** | NuRec `Volume` USDZ | post-activation | Omniverse Kit 107.3 - 110.1 / Isaac Sim 5.0 - 6.0 |

> [!NOTE]
> While Isaac Sim 6.0 supports both the `ParticleField` (standard USD) schema
> and the NuRec USDZ output, NuRec is going to be deprecated and replaced by
> `ParticleField`. Prefer `ParticleField` for new assets. This is a beta
> feature and the workflows are likely to change in future versions.

## Requirements

USD and NuRec export depend on `usd-core` (the `pxr` module), which
`pyproject.toml` installs only on linux `x86_64`. On other platforms the
USD-based export paths and their tests are unavailable; PLY export has no
such constraint.

## Exporting at the end of training

Enable the `export_usd` config block to write a USD asset when training
finishes:

```bash
python train.py --config-name apps/colmap_3dgut.yaml path=data/mipnerf360/garden/ \
    out_dir=runs experiment_name=garden_3dgut dataset.downsample_factor=2 \
    export_usd.enabled=true
```

Relevant `export_usd` keys (see [`configs/base_gs.yaml`](../../configs/base_gs.yaml)
for the full set and defaults):

```yaml
export_usd:
  enabled: false
  path: ""
  format: standard          # standard (ParticleField3DGaussianSplat) | nurec
  half_precision: false
  export_cameras: true
  export_background: true
  apply_normalizing_transform: true
  sorting_mode_hint: cameraDistance   # zDepth | cameraDistance | rayHitDistance
  export_post_processing: true        # PPISP — see usd/post_processing/README.md
```

## Standalone USD export

Export a USD asset directly from a checkpoint:

```bash
python -m threedgrut.export.scripts.export_usd \
    --checkpoint path/to/checkpoint.pt \
    --output path/to/asset.usdz \
    --dataset path/to/dataset
```

The output extension selects the container: `.usdz` (packaged archive),
`.usda` (human-readable text), or `.usd`. Useful flags:

- `--format {standard,nurec}` — `standard` writes the
  `ParticleField3DGaussianSplat` schema; `nurec` writes the Omniverse NuRec
  format.
- `--half` / `--half-geometry` / `--half-features` — half-precision
  attributes for smaller files (LightField schema).
- `--no-cameras`, `--no-background`, `--no-transform` — skip camera export,
  background/environment export, or the normalizing transform.
- `--sorting-mode-hint {zDepth,cameraDistance,rayHitDistance}` — author the
  `ParticleField` sorting hint (use `rayHitDistance` for ray-tracing
  renderers that support ray-hit sorting).
- `--dataset` — dataset path for camera export, overriding the path stored
  in the checkpoint.
- `--no-usd-validate` — skip OpenUSD stage validation after standard
  export.

PPISP post-processing flags (`--ppisp-integration-mode`,
`--scene-radiance-scale`, ...) are documented in
[`usd/post_processing/README.md`](usd/post_processing/README.md).

Run `python -m threedgrut.export.scripts.export_usd --help` for the full
CLI.

## Transcoding between formats

`transcode.py` converts an existing export between PLY, USD (lightfield),
and NuRec without re-running training:

```bash
# PLY -> USD ParticleField
python -m threedgrut.export.scripts.transcode model.ply -o model.usdz --format lightfield

# NuRec USD -> USD ParticleField
python -m threedgrut.export.scripts.transcode nurec.usd -o lightfield.usdz --format lightfield

# PLY -> NuRec USDZ
python -m threedgrut.export.scripts.transcode model.ply -o model.usdz --format nurec
```

The input format is detected from the extension (`.ply` vs `.usd*`), and
USD inputs are further refined to `nurec` vs `lightfield` by inspecting the
stage. Output is chosen with `--format {ply,lightfield,nurec}`. Useful
flags:

- `--max-sh-degree INT` — max SH degree for PLY input.
- `--half` / `--half-geometry` / `--half-features` — half-precision output
  (LightField).
- `--apply-coordinate-transform` — apply the source-to-target coordinate
  transform.
- `--render-order-hint {zDepth,cameraDistance,rayHitDistance}` — sorting
  hint (lightfield output only; ignored for `ply`/`nurec`).
- `--no-copy-source-prims` — do not copy non-Gaussian prims (cameras, etc.)
  from a USD source. `--copy-source-include-gaussians` also merges the
  source `/World/Gaussians` prim (skipped by default).
- `--no-usd-validate` — skip OpenUSD stage validation.

## Converting PLY files to USDZ

Convert existing Gaussian data in PLY format (e.g. from 3DGS) to NuRec
USDZ:

```bash
python -m threedgrut.export.scripts.ply_to_usd path/to/model.ply \
    --output_file path/to/output.usdz
```

`--output_file` defaults to the input path with a `.usdz` extension. The
resulting USDZ does not include a mesh; if you need one (e.g. for collision
geometry), use the next step.

## Adding a mesh to a USDZ file

Add a mesh (PLY or USD) into an existing USDZ — useful for assets with
physics properties such as collision geometry:

```bash
python -m threedgrut.export.scripts.add_mesh_to_usdz \
    --input_usdz path/to/input.usdz \
    --output_usdz path/to/output.usdz \
    --mesh_ply path/to/mesh.ply \
    --set_collision
```

Provide the mesh with `--mesh_ply` (converted to `mesh.usd`, both packaged)
or `--mesh_usd`. Optional flags:

- `--set_collision` — enable collision on mesh prims.
- `--set_invisible` — make mesh prims invisible.
- `--referencing_usd` — which USD file in the package to modify (default:
  auto-detect the one with a Volume prim).

## PPISP post-processing export

When a checkpoint contains a supported PPISP module, USD export includes
post-processing by default (`export_usd.export_post_processing=true`). PPISP
can be authored as runtime Omniverse SPG shaders (`spg-runtime`) or folded
into the Gaussian SH coefficients (`sh-optimized`), with an optional
per-camera controller path.

The full rationale, CLI/config reference, authored USD surface, traps and
troubleshooting live in [`usd/post_processing/README.md`](usd/post_processing/README.md).
