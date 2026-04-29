# Image Comparison

Viser based image comparison viewer for either two specific images or two folders of matching image names.

## Usage

Compare two images:

```bash
python tools/image_comparison/image_comparison.py --images path/to/a.png path/to/b.png
```

Compare two folders:

```bash
python tools/image_comparison/image_comparison.py --folders path/to/folder_a path/to/folder_b
```

Optional arguments:

```bash
python tools/image_comparison/image_comparison.py --folders path/to/folder_a path/to/folder_b --port 8080 --target_fps 20
```

Serve on all network interfaces for another host:

```bash
python tools/image_comparison/image_comparison.py --folders path/to/folder_a path/to/folder_b --host 0.0.0.0 --port 8080
```

Then open `http://<server-ip>:8080` from the other host. If direct access is blocked, use SSH forwarding:

```bash
ssh -L 8080:localhost:8080 user@server-host
```

## Viewer Modes

- `Display Mode = fit_largest_dimension`: scales the image so one dimension fills the viewport while preserving the image aspect ratio. The other dimension is smaller than or equal to the viewport. This is the default.
- `Display Mode = fit`: stretches the image to fill both viewport dimensions.
- `slider`: displays both images in the same frame, split by a vertical or horizontal slider. The split can be changed from the `Slider Position` GUI control.
- `checkerboard`: alternates images with a checkerboard mask.
- `diff`: displays a selectable difference map with a `JET` colormap and a scale slider.

When folder mode is used, images are matched by file name. Duplicate file names inside one folder are rejected so the comparison target is unambiguous.
Use `Previous Image` and `Next Image` to cycle through matched image pairs.

## Metrics

The `Metrics` panel displays scalar `PSNR`, `SSIM`, `LPIPS`, and `FLIP` values for the selected image pair. `PSNR` and `SSIM` are computed automatically. Use `Compute LPIPS / FLIP` to compute the heavier perceptual metrics on demand.

The `Diff Metric` dropdown supports:

- `l1`: per-pixel mean absolute RGB difference.
- `l2`: per-pixel RGB root mean squared difference.
- `psnr`: per-pixel PSNR-derived error, where lower PSNR is brighter.
- `ssim`: local `1 - SSIM` dissimilarity.
- `lpips`: scalar LPIPS displayed as a uniform heatmap.
- `flip`: FLIP error map when `flip-evaluator` is installed.

`LPIPS` depends on `torchmetrics` and its model weights. `FLIP` depends on the `flip-evaluator` package, which provides the `flip_evaluator` Python module. Do not install the unrelated `flip` package.
