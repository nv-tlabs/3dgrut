#!/usr/bin/env python3
"""Utilities for NHT Step 2 render-only parity checks.

This tool is intentionally separate from production training/rendering code.
It provides the glue needed to evaluate one 3DGRUT NHT checkpoint through the
reference gsplat/NHT eval path.
"""

from __future__ import annotations

import argparse
import math
import sys
import statistics
from pathlib import Path
from typing import Any


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise SystemExit("PyTorch is required for checkpoint conversion.") from exc
    return torch


def _to_plain(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return value.item()
    except AttributeError:
        return value


def _get_nested(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _decoder_config(ckpt: dict[str, Any]) -> dict[str, Any]:
    conf = ckpt.get("config")
    arch = ckpt.get("feature_decoder", {}).get("arch", {})

    dir_encoding = _get_nested(conf, "model.nht_decoder.dir_encoding", "SphericalHarmonics")
    if str(dir_encoding).lower() in {"sphericalharmonics", "sh"}:
        view_encoding_type = "sh"
    elif str(dir_encoding).lower() in {"frequency", "fourier"}:
        view_encoding_type = "fourier"
    else:
        raise ValueError(f"Unsupported 3DGRUT NHT dir_encoding={dir_encoding!r}")

    feature_dim = int(ckpt.get("particle_feature_dim", ckpt["features"].shape[-1]))
    return {
        "feature_dim": feature_dim,
        "enable_view_encoding": True,
        "view_encoding_type": view_encoding_type,
        "mlp_hidden_dim": int(
            _get_nested(conf, "model.nht_decoder.hidden_dim", arch.get("hidden_dim", 128))
        ),
        "mlp_num_layers": int(
            _get_nested(conf, "model.nht_decoder.num_layers", arch.get("num_layers", 3))
        ),
        "sh_degree": int(_get_nested(conf, "model.nht_decoder.dir_encoding_degree", 3)),
        "sh_scale": float(_get_nested(conf, "model.nht_decoder.sh_scale", arch.get("sh_scale", 3.0))),
        "fourier_num_freqs": int(_get_nested(conf, "model.nht_decoder.dir_encoding_degree", 4)),
        "primitive_type": "3dgs",
        "center_ray_encoding": bool(_get_nested(conf, "model.nht_decoder.center_ray_encoding", False)),
        "auxiliary_output_dim": 0,
    }


def convert_checkpoint(args: argparse.Namespace) -> None:
    torch = _load_torch()
    src = Path(args.grut_ckpt)
    dst = Path(args.out_ckpt)
    dst.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    required = ["positions", "rotation", "scale", "density", "features", "feature_decoder"]
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise SystemExit(f"{src} is not a 3DGRUT NHT checkpoint; missing {missing}")

    module = ckpt["feature_decoder"].get("module", {})
    ema = ckpt["feature_decoder"].get("ema", {})
    if "network.params" not in module:
        raise SystemExit("3DGRUT checkpoint missing feature_decoder.module['network.params']")

    step = args.step
    if step is None:
        global_step = int(ckpt.get("global_step", 0))
        step = max(global_step - 1, 0) if global_step else 0

    def maybe_half(t):
        return t.detach().cpu().half() if args.fp16 else t.detach().cpu()

    density = ckpt["density"].detach().cpu()
    if density.ndim == 2 and density.shape[-1] == 1:
        density = density[:, 0]

    out = {
        "step": int(step),
        "splats": {
            "features": maybe_half(ckpt["features"]),
            "means": ckpt["positions"].detach().cpu(),
            "opacities": density,
            "quats": ckpt["rotation"].detach().cpu(),
            "scales": ckpt["scale"].detach().cpu(),
        },
        "deferred_module": {"backbone.params": maybe_half(module["network.params"])},
        "deferred_module_config": _decoder_config(ckpt),
    }
    if "network.params" in ema:
        out["deferred_ema"] = {"backbone.params": maybe_half(ema["network.params"])}

    torch.save(out, dst)
    print(f"wrote {dst}")
    print(f"step: {out['step']}")
    print("splats:")
    for key, value in out["splats"].items():
        print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
    print("deferred_module_config:")
    for key, value in out["deferred_module_config"].items():
        print(f"  {key}: {value}")


def saved_png_compare(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise SystemExit("Pillow and numpy are required for PNG comparison.") from exc

    current = sorted(Path(args.grut_renders).glob("*.png"))
    reference = sorted(Path(args.ref_renders).glob(args.ref_glob))
    if not current or not reference:
        raise SystemExit("No PNGs found for comparison.")
    n = min(len(current), len(reference))
    if len(current) != len(reference):
        print(f"warning: comparing first {n} frames; counts are {len(current)} and {len(reference)}")

    def read_rgb(path: Path):
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    def score(a, b):
        mse = float(np.mean((a - b) ** 2))
        psnr = -10.0 * math.log10(max(mse, 1e-12))
        mae = float(np.mean(np.abs(a - b)))
        return psnr, mae

    by_half: dict[str, list[tuple[float, float]]] = {"left": [], "right": []}
    for cur_path, ref_path in zip(current[:n], reference[:n]):
        cur = read_rgb(cur_path)
        ref = read_rgb(ref_path)
        if ref.shape[0] == cur.shape[0] and ref.shape[1] == cur.shape[1] * 2:
            width = cur.shape[1]
            halves = {"left": ref[:, :width], "right": ref[:, width:]}
        elif ref.shape == cur.shape:
            halves = {"right": ref}
        else:
            raise SystemExit(f"Shape mismatch: {cur_path} {cur.shape} vs {ref_path} {ref.shape}")
        for half, img in halves.items():
            by_half.setdefault(half, []).append(score(cur, img))

    for half, values in by_half.items():
        if not values:
            continue
        psnrs = [v[0] for v in values]
        maes = [v[1] for v in values]
        print(
            f"{half}: count={len(values)} "
            f"psnr_mean={statistics.mean(psnrs):.6f} "
            f"psnr_min={min(psnrs):.6f} "
            f"psnr_max={max(psnrs):.6f} "
            f"mae_mean={statistics.mean(maes):.6f}"
        )


def _ray_dirs_from_grut_batch(torch, gpu_batch: Any, center_ray_encoding: bool) -> tuple[Any, Any]:
    R = gpu_batch.T_to_world[:, :3, :3]
    rays_dir_cam = gpu_batch.rays_dir
    if center_ray_encoding:
        B, H, W, _ = rays_dir_cam.shape
        center_ray_world = torch.nn.functional.normalize(R[:, :, 2], dim=-1)
        rays_dir_world = center_ray_world.view(B, 1, 1, 3).expand(B, H, W, 3)
    else:
        rays_dir_world = torch.einsum("bij,bhwj->bhwi", R, rays_dir_cam)
        rays_dir_world = torch.nn.functional.normalize(rays_dir_world, dim=-1)
    return rays_dir_world, (rays_dir_world * 3.0 + 1.0) * 0.5


def _detach_cpu(tensor: Any) -> Any:
    return tensor.detach().float().cpu()


def _squeeze_batch(tensor: Any) -> Any:
    if tensor.ndim >= 1 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def _print_dump_summary(path: Path, dump: dict[str, Any]) -> None:
    print(f"wrote {path}")
    for key, value in dump.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
        elif isinstance(value, (str, int, float, bool)) or value is None:
            print(f"  {key}: {value}")


def dump_grut_frame(args: argparse.Namespace) -> None:
    torch = _load_torch()
    from threedgrut.render import Renderer
    from threedgrut.utils.render import apply_background, apply_feature_decoder

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    renderer = Renderer.from_checkpoint(
        args.grut_ckpt,
        out_dir=args.tmp_out_dir,
        path=args.data_dir,
        save_gt=True,
        computes_extra_metrics=False,
    )
    renderer.model.eval()
    if renderer.feature_decoder is None:
        raise SystemExit("Checkpoint did not load a NHT feature decoder.")
    renderer.feature_decoder.eval()

    if args.frame_index < 0 or args.frame_index >= len(renderer.dataset):
        raise SystemExit(f"Frame index {args.frame_index} is out of range for {len(renderer.dataset)} frames.")
    batch = torch.utils.data.default_collate([renderer.dataset[args.frame_index]])

    with torch.no_grad():
        gpu_batch = renderer.dataset.get_gpu_batch_with_intrinsics(batch)
        raw_outputs = renderer.model(gpu_batch)
        raw_features = raw_outputs["pred_features"]
        alpha = raw_outputs["pred_opacity"]
        center_ray_encoding = bool(getattr(renderer.conf.model.nht_decoder, "center_ray_encoding", False))
        ray_dirs_world, ray_dirs_mlp = _ray_dirs_from_grut_batch(torch, gpu_batch, center_ray_encoding)

        decoded_outputs = dict(raw_outputs)
        decoded_outputs = apply_feature_decoder(
            renderer.feature_decoder,
            decoded_outputs,
            gpu_batch,
            training=False,
            center_ray_encoding=center_ray_encoding,
        )
        decoded_rgb = decoded_outputs["pred_features"]
        final_outputs = apply_background(renderer.model.background, dict(decoded_outputs), gpu_batch, training=False)
        final_rgb = final_outputs["pred_features"]

    dump = {
        "format": "grut_frame_dump_v1",
        "frame_index": int(args.frame_index),
        "global_step": int(renderer.global_step),
        "center_ray_encoding": center_ray_encoding,
        "features": _detach_cpu(_squeeze_batch(raw_features)),
        "alpha": _detach_cpu(_squeeze_batch(alpha)).squeeze(-1),
        "ray_dirs_world": _detach_cpu(_squeeze_batch(ray_dirs_world)),
        "ray_dirs_mlp": _detach_cpu(_squeeze_batch(ray_dirs_mlp)),
        "decoded_rgb": _detach_cpu(_squeeze_batch(decoded_rgb)),
        "final_rgb": _detach_cpu(_squeeze_batch(final_rgb)),
        "gt": _detach_cpu(_squeeze_batch(gpu_batch.rgb_gt)),
    }
    torch.save(dump, out)
    _print_dump_summary(out, dump)


def _add_reference_paths(nht_root: Path) -> None:
    examples = nht_root / "gsplat" / "examples"
    gsplat_root = nht_root / "gsplat"
    for path in (examples, gsplat_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def dump_ref_frame(args: argparse.Namespace) -> None:
    torch = _load_torch()
    import torch.nn.functional as F

    nht_root = Path(args.nht_root).resolve()
    _add_reference_paths(nht_root)

    from datasets.colmap import Dataset, Parser
    from gsplat import NHTParams, rasterization
    from gsplat.nht.deferred_shader import DeferredShaderModule

    device = torch.device(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ref_ckpt, map_location=device, weights_only=False)
    dm = DeferredShaderModule(**ckpt["deferred_module_config"]).to(device)
    shader_state = ckpt.get("deferred_ema") or ckpt["deferred_module"]
    dm.load_state_dict(shader_state)
    dm.eval()

    parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=True,
        test_every=args.test_every,
        load_exposure=False,
        native_images_factor=False,
    )
    valset = Dataset(parser, split="val")
    if args.frame_index >= len(valset):
        raise SystemExit(f"Frame index {args.frame_index} is out of range for {len(valset)} val frames.")
    data = valset[args.frame_index]

    pixels = data["image"].to(device) / 255.0
    height, width = pixels.shape[:2]
    camtoworlds = data["camtoworld"].to(device).unsqueeze(0)
    Ks = data["K"].to(device).unsqueeze(0)

    splats = {key: value.to(device) for key, value in ckpt["splats"].items()}
    features = splats["features"]
    if features.dtype != torch.float16 and args.features_half:
        features = features.half()

    with torch.no_grad():
        render_data, alpha, _ = rasterization(
            means=splats["means"],
            quats=F.normalize(splats["quats"], dim=-1),
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=features,
            sh_degree=None,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            tile_size=args.tile_size,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode=args.rasterize_mode,
            render_mode="RGB",
            distributed=False,
            camera_model="pinhole",
            with_ut=True,
            with_eval3d=True,
            nht_params=NHTParams(
                center_ray_mode=bool(ckpt["deferred_module_config"].get("center_ray_encoding", False)),
                ray_dir_scale=dm.ray_dir_scale,
            ),
        )
        decoded_rgb, extras = dm(render_data)

    encoded_dim = int(dm.encoded_dim)
    ray_dirs_mlp = render_data[..., encoded_dim : encoded_dim + 3]
    ray_dirs_world = (ray_dirs_mlp * 2.0 - 1.0) / float(dm.ray_dir_scale)

    dump = {
        "format": "ref_frame_dump_v1",
        "frame_index": int(args.frame_index),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "center_ray_encoding": bool(ckpt["deferred_module_config"].get("center_ray_encoding", False)),
        "encoded_dim": encoded_dim,
        "features": _detach_cpu(render_data[0, ..., :encoded_dim]),
        "alpha": _detach_cpu(alpha[0, ..., 0]),
        "ray_dirs_world": _detach_cpu(ray_dirs_world[0]),
        "ray_dirs_mlp": _detach_cpu(ray_dirs_mlp[0]),
        "decoded_rgb": _detach_cpu(decoded_rgb[0]),
        "final_rgb": _detach_cpu(decoded_rgb[0]),
        "gt": _detach_cpu(pixels),
    }
    if extras is not None:
        dump["extras"] = _detach_cpu(extras[0])
    torch.save(dump, out)
    _print_dump_summary(out, dump)


def _score_tensors(torch, a: Any, b: Any) -> dict[str, float]:
    a = a.float()
    b = b.float()
    diff = a - b
    mse = torch.mean(diff * diff).item()
    mae = torch.mean(torch.abs(diff)).item()
    max_abs = torch.max(torch.abs(diff)).item()
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    return {"psnr": psnr, "mae": mae, "max_abs": max_abs, "mse": mse}


def _exact_tensor_stats(torch, a: Any, b: Any) -> dict[str, Any]:
    exact = bool(torch.equal(a, b))
    neq = torch.ne(a, b)
    num_diff = int(torch.count_nonzero(neq).item())
    total = int(a.numel())
    return {
        "exact": exact,
        "num_diff": num_diff,
        "total": total,
        "frac_diff": (num_diff / total) if total else 0.0,
    }


def compare_dumps(args: argparse.Namespace) -> None:
    torch = _load_torch()
    left = torch.load(args.grut_dump, map_location="cpu", weights_only=False)
    right = torch.load(args.ref_dump, map_location="cpu", weights_only=False)

    print(f"left:  {args.grut_dump} ({left.get('format')})")
    print(f"right: {args.ref_dump} ({right.get('format')})")

    for key in ["gt", "alpha", "features", "ray_dirs_mlp", "ray_dirs_world", "decoded_rgb", "final_rgb"]:
        if key not in left or key not in right:
            continue
        if tuple(left[key].shape) != tuple(right[key].shape):
            print(f"{key}: shape mismatch {tuple(left[key].shape)} vs {tuple(right[key].shape)}")
            continue
        exact = _exact_tensor_stats(torch, left[key], right[key])
        scores = _score_tensors(torch, left[key], right[key])
        print(
            f"{key}: exact={exact['exact']} "
            f"diff={exact['num_diff']}/{exact['total']} ({exact['frac_diff']:.6%}) "
            f"psnr={scores['psnr']:.6f} "
            f"mae={scores['mae']:.8f} max_abs={scores['max_abs']:.8f}"
        )


def print_commands(args: argparse.Namespace) -> None:
    root = Path(args.repo_root).resolve()
    nht_root = root / "thirdparty" / "neural-harmonic-textures"
    converted = Path(args.converted_ckpt).resolve()
    data_dir = Path(args.data_dir).resolve()
    grut_ckpt = Path(args.grut_ckpt).resolve()
    grut_out = Path(args.grut_out).resolve()
    ref_out = Path(args.ref_out).resolve()

    print("3DGRUT render command:")
    print(
        "CUDA_VISIBLE_DEVICES=0 python render.py "
        f"--checkpoint {grut_ckpt} --path {data_dir} --out-dir {grut_out}"
    )
    print()
    print("Reference gsplat/NHT eval command for the converted 3DGRUT checkpoint:")
    print(
        f"cd {nht_root} && "
        "CUDA_VISIBLE_DEVICES=0 python gsplat/examples/simple_trainer_nht.py default "
        "--disable_viewer --disable_video "
        f"--data_dir {data_dir} --data_factor {args.data_factor} "
        f"--result_dir {ref_out} --strategy.cap-max {args.cap_max} "
        "--render_traj_path ellipse "
        f"--ckpt {converted}"
    )
    print()
    print("After it runs, compare:")
    print(
        f"python tools/nht_step2_parity.py saved-png-compare "
        f"--grut-renders {grut_out / 'ours_30000' / 'renders'} "
        f"--ref-renders {ref_out / 'renders'} --ref-glob 'val_step*.png'"
    )


def preflight(_: argparse.Namespace) -> None:
    torch = _load_torch()
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        print(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
    for module in ["omegaconf", "PIL", "numpy"]:
        try:
            __import__(module)
            print(f"{module}: ok")
        except Exception as exc:  # noqa: BLE001 - preflight diagnostic
            print(f"{module}: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert", help="convert a 3DGRUT NHT checkpoint to reference gsplat/NHT format")
    p.add_argument("--grut-ckpt", required=True)
    p.add_argument("--out-ckpt", required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--fp16", action="store_true", help="write feature/MLP buffers as fp16 like reference checkpoints")
    p.set_defaults(func=convert_checkpoint)

    p = sub.add_parser("saved-png-compare", help="compare saved 3DGRUT renders to reference side-by-side PNGs")
    p.add_argument("--grut-renders", required=True)
    p.add_argument("--ref-renders", required=True)
    p.add_argument("--ref-glob", default="val_step29999_*.png")
    p.set_defaults(func=saved_png_compare)

    p = sub.add_parser("dump-grut-frame", help="dump one 3DGRUT-rendered validation frame before/after NHT decode")
    p.add_argument("--grut-ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--tmp-out-dir", default="results/nht_step2_parity/tmp_grut_dump")
    p.set_defaults(func=dump_grut_frame)

    p = sub.add_parser("dump-ref-frame", help="dump one reference gsplat/NHT validation frame before/after NHT decode")
    p.add_argument("--ref-ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nht-root", default="thirdparty/neural-harmonic-textures")
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--data-factor", type=int, default=4)
    p.add_argument("--test-every", type=int, default=8)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rasterize-mode", default="classic", choices=["classic", "antialiased"])
    p.add_argument("--features-half", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=dump_ref_frame)

    p = sub.add_parser("compare-dumps", help="compare 3DGRUT and reference one-frame tensor dumps")
    p.add_argument("--grut-dump", required=True)
    p.add_argument("--ref-dump", required=True)
    p.set_defaults(func=compare_dumps)

    p = sub.add_parser("commands", help="print commands for rerendering both sides")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--grut-ckpt", required=True)
    p.add_argument("--converted-ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--grut-out", required=True)
    p.add_argument("--ref-out", required=True)
    p.add_argument("--data-factor", type=int, default=4)
    p.add_argument("--cap-max", type=int, default=1_000_000)
    p.set_defaults(func=print_commands)

    p = sub.add_parser("preflight", help="print local runtime readiness")
    p.set_defaults(func=preflight)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
