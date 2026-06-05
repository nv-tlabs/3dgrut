# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pxr import Gf, Sdf, Usd, UsdGeom

from threedgrut.export.usd.ppisp_spg import (
    get_ppisp_auto_spg_files,
    get_ppisp_spg_files,
)
from threedgrut.export.usd.writers.ppisp_controller_weights import (
    EXPECTED_CONTROLLER_WEIGHTS_LEN,
)
from threedgrut.export.usd.writers.ppisp_controller_writer import (
    CONTROLLER_FEATURES_RENDER_VAR,
    CONTROLLER_PARAMS_RENDER_VAR,
    CONTROLLER_POOL_PRIM_NAME_TEMPLATE,
    CONTROLLER_WEIGHTS_CAMERA_ATTR,
    EMBEDDED_CONTROLLER_WEIGHTS_MARKER,
    EMBEDDED_CONTROLLER_WEIGHTS_SYMBOL,
    PPISP_AUTO_PRIM_NAME,
)
from threedgrut.export.usd.writers.ppisp_controller_writer import (
    PPISP_TILE_COUNT_INPUT_NAMES as AUTO_PPISP_TILE_COUNT_INPUT_NAMES,
)
from threedgrut.export.usd.writers.ppisp_controller_writer import (
    add_ppisp_auto_shader_to_render_product,
    get_ppisp_embedded_controller_spg_files,
)
from threedgrut.export.usd.writers.ppisp_writer import (
    PPISP_ATTR_NAMESPACE,
    PPISP_CAMERA_SUFFIX,
    PPISP_TILE_COUNT_INPUT_NAMES,
    add_ppisp_shader_to_render_product,
    add_ppisp_to_all_render_products,
)


def _make_controller() -> nn.Module:
    class Controller(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=1),
                nn.MaxPool2d(kernel_size=3, stride=3),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=1),
                nn.AdaptiveAvgPool2d((5, 5)),
            )
            self.mlp_trunk = nn.Sequential(
                nn.Linear(1601, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.exposure_head = nn.Linear(128, 1)
            self.color_head = nn.Linear(128, 8)

    controller = Controller()
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for param in controller.parameters():
            param.copy_(torch.empty_like(param).normal_(0.0, 0.01, generator=generator))
    return controller


class _FakePPISP(nn.Module):
    def __init__(self, *, use_controller: bool = True) -> None:
        super().__init__()
        self.num_cameras = 1
        self.exposure_params = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
        self.color_params = torch.arange(24, dtype=torch.float32).reshape(3, 8) * 0.01
        self.vignetting_params = torch.zeros((1, 3, 5), dtype=torch.float32)
        self.crf_params = torch.zeros((1, 3, 4), dtype=torch.float32)
        self.controllers = nn.ModuleList([_make_controller()]) if use_controller else nn.ModuleList()


def _make_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Cameras")
    UsdGeom.Camera.Define(stage, "/World/Cameras/camera_0")
    UsdGeom.Scope.Define(stage, "/Render")
    product = stage.DefinePrim("/Render/camera_0", "RenderProduct")
    product.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(8, 6))
    product.CreateRelationship("camera").SetTargets([Sdf.Path("/World/Cameras/camera_0")])
    hdr = stage.DefinePrim("/Render/camera_0/HdrColor", "RenderVar")
    hdr.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("HdrColor")
    product.CreateRelationship("orderedVars").SetTargets([Sdf.Path("/Render/camera_0/HdrColor")])
    return stage


def test_static_ppisp_sidecars_are_cuda() -> None:
    names = {entry.filename for entry in get_ppisp_spg_files()}
    assert names == {"ppisp_usd_spg.cu", "ppisp_usd_spg.cu.lua", "ppisp_usd_spg.usda"}
    assert not any(name.endswith(".slang") for name in names)


def test_auto_ppisp_sidecars_are_cuda() -> None:
    names = {entry.filename for entry in get_ppisp_auto_spg_files()}
    assert names == {"ppisp_usd_spg_auto.cu", "ppisp_usd_spg_auto.cu.lua", "ppisp_usd_spg_auto.usda"}
    assert not any(name.endswith(".slang") for name in names)


def test_static_ppisp_shader_references_cuda_sidecar() -> None:
    stage = _make_stage()
    add_ppisp_shader_to_render_product(
        stage=stage,
        render_product_path="/Render/camera_0",
        camera_index=0,
        ppisp=_FakePPISP(use_controller=False),
        frame_indices=[0, 1, 2],
    )
    shader = stage.GetPrimAtPath("/Render/camera_0/PPISP")
    assert shader.GetAttribute("info:spg:sourceAsset").Get().path == "ppisp_usd_spg.cu"
    assert shader.GetAttribute("info:spg:sourceAsset:subIdentifier").Get() == "ppispProcess"
    for name in PPISP_TILE_COUNT_INPUT_NAMES:
        assert shader.GetAttribute(f"inputs:{name}").Get() == 1
    ldr = stage.GetPrimAtPath("/Render/camera_0/LdrColor")
    assert ldr.GetAttribute("omni:rtx:aov").GetConnections() == [Sdf.Path("/Render/camera_0/PPISP.outputs:PPISPColor")]


def test_embedded_controller_sidecar_contains_weights_without_weight_input() -> None:
    ppisp = _FakePPISP(use_controller=True)
    sidecars = get_ppisp_embedded_controller_spg_files(ppisp, [0])
    by_name = {entry.filename: entry.serialized for entry in sidecars}
    assert set(by_name) == {"ppisp_controller_0.cu", "ppisp_controller_0.cu.lua"}
    cuda_source = by_name["ppisp_controller_0.cu"].decode("utf-8")
    lua_source = by_name["ppisp_controller_0.cu.lua"].decode("utf-8")
    assert EMBEDDED_CONTROLLER_WEIGHTS_SYMBOL in cuda_source
    assert EMBEDDED_CONTROLLER_WEIGHTS_MARKER not in cuda_source
    assert str(EXPECTED_CONTROLLER_WEIGHTS_LEN) in cuda_source
    assert 'inputs["weights"]' not in lua_source


def test_auto_ppisp_controller_graph_uses_embedded_cuda_no_weight_attr() -> None:
    stage = _make_stage()
    ppisp = _FakePPISP(use_controller=True)
    add_ppisp_auto_shader_to_render_product(
        stage=stage,
        render_product_path="/Render/camera_0",
        camera_index=0,
        ppisp=ppisp,
        controller=ppisp.controllers[0],
    )
    pool = stage.GetPrimAtPath("/Render/camera_0/PPISPControllerPool_0")
    controller = stage.GetPrimAtPath("/Render/camera_0/PPISPController_0")
    auto = stage.GetPrimAtPath(f"/Render/camera_0/{PPISP_AUTO_PRIM_NAME}")
    assert pool.GetAttribute("info:spg:sourceAsset").Get().path == "ppisp_controller_0.cu"
    assert controller.GetAttribute("info:spg:sourceAsset").Get().path == "ppisp_controller_0.cu"
    assert auto.GetAttribute("info:spg:sourceAsset").Get().path == "ppisp_usd_spg_auto.cu"
    assert not controller.GetAttribute("inputs:weights").IsValid()
    assert pool.GetAttribute("inputs:responsivity").Get() == 1.0
    for prim in (pool, controller, auto):
        for name in AUTO_PPISP_TILE_COUNT_INPUT_NAMES:
            assert prim.GetAttribute(f"inputs:{name}").Get() == 1
    assert stage.GetPrimAtPath(f"/Render/camera_0/{CONTROLLER_FEATURES_RENDER_VAR}").IsValid()
    assert stage.GetPrimAtPath(f"/Render/camera_0/{CONTROLLER_PARAMS_RENDER_VAR}").IsValid()


def test_auto_ppisp_controller_graph_authors_controller_pool_responsivity() -> None:
    stage = _make_stage()
    ppisp = _FakePPISP(use_controller=True)
    add_ppisp_auto_shader_to_render_product(
        stage=stage,
        render_product_path="/Render/camera_0",
        camera_index=0,
        ppisp=ppisp,
        controller=ppisp.controllers[0],
        responsivity=2.5,
    )
    pool = stage.GetPrimAtPath(f"/Render/camera_0/{CONTROLLER_POOL_PRIM_NAME_TEMPLATE.format(camera_index=0)}")
    auto = stage.GetPrimAtPath(f"/Render/camera_0/{PPISP_AUTO_PRIM_NAME}")
    assert pool.GetAttribute("inputs:responsivity").Get() == 2.5
    assert auto.GetAttribute("inputs:responsivity").Get() == 2.5


def test_ppisp_camera_shim_is_sibling_of_source_camera() -> None:
    stage = _make_stage()
    add_ppisp_to_all_render_products(
        stage=stage,
        ppisp=_FakePPISP(use_controller=False),
        camera_names=["camera_0"],
        camera_frame_mapping={"camera_0": [0, 1, 2]},
    )
    product = stage.GetPrimAtPath("/Render/camera_0")
    ppisp_camera_path = Sdf.Path(f"/World/Cameras/camera_0{PPISP_CAMERA_SUFFIX}")
    assert product.GetRelationship("camera").GetTargets() == [ppisp_camera_path]
    ppisp_camera = stage.GetPrimAtPath(ppisp_camera_path)
    assert ppisp_camera.IsValid()
    assert not stage.GetPrimAtPath(f"/Render/camera_0/camera_0{PPISP_CAMERA_SUFFIX}").IsValid()
    assert ppisp_camera.GetAttribute(f"{PPISP_ATTR_NAMESPACE}responsivity").Get() == 1.0
    assert ppisp_camera.GetAttribute(f"{PPISP_ATTR_NAMESPACE}exposureOffset").GetTimeSamples() == [0.0, 1.0, 2.0]
    assert ppisp_camera.GetAttribute(f"{PPISP_ATTR_NAMESPACE}colorLatentBlue").IsValid()
    assert ppisp_camera.GetAttribute(f"{PPISP_ATTR_NAMESPACE}vignettingCenterR").IsValid()


def test_controller_export_authors_ppisp_camera_controller_weights() -> None:
    stage = _make_stage()
    add_ppisp_to_all_render_products(
        stage=stage,
        ppisp=_FakePPISP(use_controller=True),
        camera_names=["camera_0"],
        camera_frame_mapping={"camera_0": [0, 1, 2]},
        use_controller=True,
    )
    ppisp_camera = stage.GetPrimAtPath(f"/World/Cameras/camera_0{PPISP_CAMERA_SUFFIX}")
    weights = ppisp_camera.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{CONTROLLER_WEIGHTS_CAMERA_ATTR}").Get()
    assert len(weights) == EXPECTED_CONTROLLER_WEIGHTS_LEN


def _requires_cuda_ppisp() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for exported PPISP equivalence tests")
    try:
        import ppisp  # noqa: F401
    except ImportError:
        pytest.skip("ppisp package is required for exported PPISP equivalence tests")


def _randomize_ppisp_params(ppisp: nn.Module, *, seed: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        ppisp.exposure_params.copy_(torch.empty_like(ppisp.exposure_params).uniform_(-0.35, 0.35, generator=generator))
        ppisp.color_params.copy_(torch.empty_like(ppisp.color_params).normal_(0.0, 0.35, generator=generator))
        ppisp.vignetting_params[..., :2].copy_(
            torch.empty_like(ppisp.vignetting_params[..., :2]).normal_(0.0, 0.04, generator=generator)
        )
        ppisp.vignetting_params[..., 2:].copy_(
            torch.empty_like(ppisp.vignetting_params[..., 2:]).uniform_(-0.35, 0.02, generator=generator)
        )
        ppisp.crf_params.add_(torch.empty_like(ppisp.crf_params).normal_(0.0, 0.08, generator=generator))


def _randomize_controller(
    controller: nn.Module,
    *,
    seed: int,
    weight_scale: float = 0.03,
    bias_scale: float = 0.03,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name, param in controller.named_parameters():
            scale = weight_scale if "weight" in name else bias_scale
            param.copy_(torch.empty_like(param).normal_(0.0, scale, generator=generator))


def _pixel_coords(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1)


def _make_gaussian_radiance_scene(width: int, height: int, *, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    image = torch.zeros((height, width, 3), device=device, dtype=torch.float32)

    for _ in range(90):
        cx = torch.rand((), generator=generator, device=device) * float(width - 1)
        cy = torch.rand((), generator=generator, device=device) * float(height - 1)
        sx = 1.5 + torch.rand((), generator=generator, device=device) * 10.0
        sy = 1.5 + torch.rand((), generator=generator, device=device) * 10.0
        amp = 0.03 + torch.rand((), generator=generator, device=device) * 0.55
        color = 0.25 + torch.rand((3,), generator=generator, device=device) * 1.5
        blob = torch.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
        image = image + blob.unsqueeze(-1) * amp * color

    wave = 1.0 + 0.18 * torch.sin(x * 0.31) + 0.12 * torch.cos(y * 0.23) + 0.06 * torch.sin((x + y) * 0.17)
    return torch.clamp_min(image * wave.unsqueeze(-1) + 0.02, 0.0)


def _assert_exported_ldr_close(actual: torch.Tensor, expected_hdr_float: torch.Tensor) -> None:
    expected_u8 = torch.floor(expected_hdr_float.clamp(0.0, 1.0) * 255.0) / 255.0
    diff = (actual - expected_u8).abs()
    assert float(diff.mean().item()) <= 1.0 / 255.0
    assert float(diff.max().item()) <= 3.0 / 255.0 + 1.0e-6


def test_exported_embedded_cuda_controller_matches_torch_controller() -> None:
    _requires_cuda_ppisp()
    from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

    from threedgrut.export.tests.ppisp_cuda_controller_runtime import (
        ExportedEmbeddedCudaController,
    )

    device = torch.device("cuda")
    ppisp = PPISP(num_cameras=1, num_frames=1, config=PPISPConfig(use_controller=True)).to(device).eval()
    _randomize_controller(ppisp.controllers[0], seed=11)
    source = get_ppisp_embedded_controller_spg_files(ppisp, [0])[0].serialized
    runtime = ExportedEmbeddedCudaController(source, device=device)
    hdr = torch.rand((31, 37, 3), device=device, dtype=torch.float32) * 0.85
    prior = torch.tensor([0.17], device=device)
    with torch.no_grad():
        torch_exposure, torch_color = ppisp.controllers[0](hdr, prior)
        cuda_exposure, cuda_color = runtime(hdr, prior)
    torch.testing.assert_close(cuda_exposure, torch_exposure, rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(cuda_color, torch_color, rtol=2.0e-5, atol=2.0e-5)


def test_exported_embedded_cuda_controller_responsivity_matches_prescaled_hdr() -> None:
    _requires_cuda_ppisp()
    from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

    from threedgrut.export.tests.ppisp_cuda_controller_runtime import (
        ExportedEmbeddedCudaController,
    )

    device = torch.device("cuda")
    ppisp = PPISP(num_cameras=1, num_frames=1, config=PPISPConfig(use_controller=True)).to(device).eval()
    # Use an input-sensitive controller so non-unit responsivity changes the
    # prediction instead of passing only through near-zero weights.
    _randomize_controller(ppisp.controllers[0], seed=11, weight_scale=0.15, bias_scale=0.005)
    source = get_ppisp_embedded_controller_spg_files(ppisp, [0])[0].serialized
    runtime = ExportedEmbeddedCudaController(source, device=device)
    hdr = _make_gaussian_radiance_scene(73, 67, seed=101, device=device)
    prior = torch.tensor([0.23], device=device, dtype=torch.float32)

    responsivity = 1.7
    scaled_exposure, scaled_color = runtime(hdr, prior, responsivity=responsivity)
    prescaled_exposure, prescaled_color = runtime(hdr * responsivity, prior)

    torch.testing.assert_close(scaled_exposure, prescaled_exposure, rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(scaled_color, prescaled_color, rtol=2.0e-5, atol=2.0e-5)

    base_exposure, _ = runtime(hdr, prior)
    assert float((scaled_exposure - base_exposure).abs().item()) > 1.0e-4


def test_exported_embedded_cuda_controller_tiled_params_match_per_tile() -> None:
    _requires_cuda_ppisp()
    from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

    from threedgrut.export.tests.ppisp_cuda_controller_runtime import (
        ExportedEmbeddedCudaController,
    )

    device = torch.device("cuda")
    ppisp = PPISP(num_cameras=1, num_frames=1, config=PPISPConfig(use_controller=True)).to(device).eval()
    _randomize_controller(ppisp.controllers[0], seed=13)
    source = get_ppisp_embedded_controller_spg_files(ppisp, [0])[0].serialized
    runtime = ExportedEmbeddedCudaController(source, device=device)
    prior = torch.tensor([0.17], device=device)
    hdr0 = _make_gaussian_radiance_scene(39, 33, seed=210, device=device)
    hdr1 = _make_gaussian_radiance_scene(39, 33, seed=211, device=device)
    atlas = torch.cat((hdr0, hdr1), dim=1)

    exp0, color0 = runtime(hdr0, prior)
    exp1, color1 = runtime(hdr1, prior)
    tiled = runtime.run_tiled_params(atlas, prior, tile_count_x=2, tile_count_y=1).reshape(2, 9)

    torch.testing.assert_close(tiled[0], torch.cat((exp0.reshape(1), color0.reshape(-1))), rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(tiled[1], torch.cat((exp1.reshape(1), color1.reshape(-1))), rtol=2.0e-5, atol=2.0e-5)


def test_exported_static_cuda_ppisp_matches_torch_ppisp() -> None:
    _requires_cuda_ppisp()
    from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

    from threedgrut.export.tests.ppisp_cuda_ppisp_runtime import (
        ExportedCudaPPISP,
        pack_static_ppisp_params,
    )

    device = torch.device("cuda")
    ppisp = PPISP(num_cameras=1, num_frames=1, config=PPISPConfig(use_controller=False)).to(device).eval()
    _randomize_ppisp_params(ppisp, seed=23)
    stage = _make_stage()
    shader_prim = add_ppisp_shader_to_render_product(
        stage=stage,
        render_product_path="/Render/camera_0",
        camera_index=0,
        ppisp=ppisp,
        frame_indices=[0],
        fixed_frame_index=0,
    )
    params = pack_static_ppisp_params(shader_prim).to(device=device)
    hdr = _make_gaussian_radiance_scene(65, 59, seed=202, device=device)
    coords = _pixel_coords(hdr.shape[0], hdr.shape[1], device=device)
    with torch.no_grad():
        expected = ppisp(
            hdr,
            coords,
            resolution=(hdr.shape[1], hdr.shape[0]),
            camera_idx=0,
            frame_idx=0,
        )
        actual = ExportedCudaPPISP(device=device).run_static(hdr, params)
    _assert_exported_ldr_close(actual, expected)


def test_exported_static_cuda_ppisp_tiled_atlas_matches_per_tile_untiled() -> None:
    _requires_cuda_ppisp()
    from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

    from threedgrut.export.tests.ppisp_cuda_ppisp_runtime import (
        ExportedCudaPPISP,
        pack_static_ppisp_params,
    )

    device = torch.device("cuda")
    ppisp = PPISP(num_cameras=1, num_frames=1, config=PPISPConfig(use_controller=False)).to(device).eval()
    _randomize_ppisp_params(ppisp, seed=25)
    stage = _make_stage()
    shader_prim = add_ppisp_shader_to_render_product(
        stage=stage,
        render_product_path="/Render/camera_0",
        camera_index=0,
        ppisp=ppisp,
        frame_indices=[0],
        fixed_frame_index=0,
    )
    params = pack_static_ppisp_params(shader_prim).to(device=device)
    hdr = _make_gaussian_radiance_scene(48, 40, seed=404, device=device)
    runtime = ExportedCudaPPISP(device=device)

    width = hdr.shape[1]
    single = runtime.run_static(hdr, params)
    atlas = torch.cat((hdr, hdr), dim=1)
    tiled = runtime.run_static(atlas, params, tile_count_x=2, tile_count_y=1)

    torch.testing.assert_close(tiled[:, :width], single, rtol=0.0, atol=0.0)
    torch.testing.assert_close(tiled[:, width:], single, rtol=0.0, atol=0.0)
    untiled_atlas = runtime.run_static(atlas, params)
    assert not torch.allclose(untiled_atlas[:, :width], single, rtol=0.0, atol=1.0 / 255.0)


def test_exported_auto_cuda_ppisp_matches_torch_ppisp() -> None:
    _requires_cuda_ppisp()
    from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

    from threedgrut.export.tests.ppisp_cuda_controller_runtime import (
        ExportedEmbeddedCudaController,
    )
    from threedgrut.export.tests.ppisp_cuda_ppisp_runtime import (
        ExportedCudaPPISP,
        pack_auto_ppisp_params,
    )

    device = torch.device("cuda")
    ppisp = (
        PPISP(
            num_cameras=1,
            num_frames=1,
            config=PPISPConfig(use_controller=True, controller_activation_ratio=0.0),
        )
        .to(device)
        .eval()
    )
    _randomize_ppisp_params(ppisp, seed=31)
    _randomize_controller(ppisp.controllers[0], seed=37)
    stage = _make_stage()
    auto_prim = add_ppisp_auto_shader_to_render_product(
        stage=stage,
        render_product_path="/Render/camera_0",
        camera_index=0,
        ppisp=ppisp,
        controller=ppisp.controllers[0],
        prior_exposure=0.19,
    )
    params = pack_auto_ppisp_params(auto_prim).to(device=device)
    hdr = _make_gaussian_radiance_scene(71, 62, seed=303, device=device)
    prior = torch.tensor([0.19], device=device, dtype=torch.float32)
    coords = _pixel_coords(hdr.shape[0], hdr.shape[1], device=device)

    controller_source = get_ppisp_embedded_controller_spg_files(ppisp, [0])[0].serialized
    controller_runtime = ExportedEmbeddedCudaController(controller_source, device=device)
    exposure, color = controller_runtime(hdr, prior)
    controller_params = torch.cat((exposure.reshape(1), color.reshape(-1)), dim=0)

    with torch.no_grad():
        expected = ppisp(
            hdr,
            coords,
            resolution=(hdr.shape[1], hdr.shape[0]),
            camera_idx=0,
            frame_idx=-1,
            exposure_prior=prior,
        )
        actual = ExportedCudaPPISP(device=device).run_auto(hdr, controller_params, params)
    _assert_exported_ldr_close(actual, expected)


def test_exported_auto_cuda_ppisp_tiled_atlas_selects_per_tile_controller_params() -> None:
    _requires_cuda_ppisp()
    from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

    from threedgrut.export.tests.ppisp_cuda_ppisp_runtime import (
        ExportedCudaPPISP,
        pack_auto_ppisp_params,
    )

    device = torch.device("cuda")
    ppisp = (
        PPISP(
            num_cameras=1,
            num_frames=1,
            config=PPISPConfig(use_controller=True, controller_activation_ratio=0.0),
        )
        .to(device)
        .eval()
    )
    _randomize_ppisp_params(ppisp, seed=33)
    stage = _make_stage()
    auto_prim = add_ppisp_auto_shader_to_render_product(
        stage=stage,
        render_product_path="/Render/camera_0",
        camera_index=0,
        ppisp=ppisp,
        controller=ppisp.controllers[0],
    )
    params = pack_auto_ppisp_params(auto_prim).to(device=device)
    runtime = ExportedCudaPPISP(device=device)

    hdr = _make_gaussian_radiance_scene(44, 38, seed=505, device=device)
    cp0 = torch.tensor([0.30, 0.10, -0.05, 0.20, 0.02, -0.12, 0.07, 0.15, -0.09], device=device)
    cp1 = torch.tensor([-0.40, -0.18, 0.22, -0.03, 0.11, 0.25, -0.14, 0.04, 0.19], device=device)

    width = hdr.shape[1]
    single0 = runtime.run_auto(hdr, cp0, params)
    single1 = runtime.run_auto(hdr, cp1, params)
    atlas = torch.cat((hdr, hdr), dim=1)
    tiled = runtime.run_auto(atlas, torch.cat((cp0, cp1), dim=0), params, tile_count_x=2, tile_count_y=1)

    torch.testing.assert_close(tiled[:, :width], single0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(tiled[:, width:], single1, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    os.environ.get("THREEDGRUT_RUN_PPISP_EXPORTED_CUDA_PERF") != "1",
    reason="set THREEDGRUT_RUN_PPISP_EXPORTED_CUDA_PERF=1 to run exported CUDA performance sweep",
)
def test_exported_cuda_ppisp_performance_smoke() -> None:
    _requires_cuda_ppisp()
    from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

    from threedgrut.export.tests.ppisp_cuda_controller_runtime import (
        ExportedEmbeddedCudaController,
    )
    from threedgrut.export.tests.ppisp_cuda_ppisp_runtime import (
        ExportedCudaPPISP,
        pack_auto_ppisp_params,
        pack_static_ppisp_params,
    )

    device = torch.device("cuda")
    static_ppisp = PPISP(num_cameras=1, num_frames=1, config=PPISPConfig(use_controller=False)).to(device).eval()
    stage = _make_stage()
    shader_prim = add_ppisp_shader_to_render_product(
        stage=stage,
        render_product_path="/Render/camera_0",
        camera_index=0,
        ppisp=static_ppisp,
        frame_indices=[0],
        fixed_frame_index=0,
    )
    static_params = pack_static_ppisp_params(shader_prim).to(device=device)

    auto_ppisp = (
        PPISP(
            num_cameras=1,
            num_frames=1,
            config=PPISPConfig(use_controller=True, controller_activation_ratio=0.0),
        )
        .to(device)
        .eval()
    )
    auto_stage = _make_stage()
    auto_prim = add_ppisp_auto_shader_to_render_product(
        stage=auto_stage,
        render_product_path="/Render/camera_0",
        camera_index=0,
        ppisp=auto_ppisp,
        controller=auto_ppisp.controllers[0],
        prior_exposure=0.19,
    )
    auto_params = pack_auto_ppisp_params(auto_prim).to(device=device)
    controller_source = get_ppisp_embedded_controller_spg_files(auto_ppisp, [0])[0].serialized

    hdr = torch.rand((128, 192, 3), device=device, dtype=torch.float32)
    prior = torch.tensor([0.19], device=device, dtype=torch.float32)
    image_runtime = ExportedCudaPPISP(device=device)
    controller_runtime = ExportedEmbeddedCudaController(controller_source, device=device)
    exposure, color = controller_runtime(hdr, prior)
    controller_params = torch.cat((exposure.reshape(1), color.reshape(-1)), dim=0)

    calls = (
        lambda: image_runtime.run_static(hdr, static_params),
        lambda: controller_runtime(hdr, prior),
        lambda: image_runtime.run_auto(hdr, controller_params, auto_params),
    )

    for call in calls:
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            call()
        end.record()
        torch.cuda.synchronize()
        assert start.elapsed_time(end) > 0.0
