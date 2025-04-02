# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import copy
import torch
import os
from typing import List, Optional, Dict
from enum import IntEnum
from pathlib import Path
from dataclasses import dataclass
from kaolin.render.camera import Camera, generate_centered_pixel_coords, generate_pinhole_rays
from threedgrut.utils.logger import logger
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.model.background import BackgroundColor
from playground.tracer import Tracer
from playground.utils.mesh_io import load_mesh, load_materials, load_missing_material_info, create_procedural_mesh
from playground.utils.depth_of_field import DepthOfField
from playground.utils.video_out import VideoRecorder
from playground.utils.spp import SPP
from playground.utils.kaolin_future.transform import ObjectTransform
from playground.utils.kaolin_future.conversions import polyscope_from_kaolin_camera, polyscope_to_kaolin_camera
from playground.utils.kaolin_future.fisheye import generate_fisheye_rays
from playground.utils.kaolin_future.interpolated_cameras import camera_path_generator


#################################
##       --- Common ---        ##
#################################

@dataclass
class RayPack:
    rays_ori: torch.FloatTensor
    rays_dir: torch.FloatTensor
    pixel_x: Optional[torch.IntTensor] = None
    pixel_y: Optional[torch.IntTensor] = None
    mask: Optional[torch.BoolTensor] = None

    def split(self, size=None) -> List[RayPack]:
        if size is None:
            return [self]
        assert self.rays_ori.ndim == 2 and self.rays_dir.ndim == 2, 'Only 1D ray packs can be split'
        rays_orig = torch.split(self.rays_ori, size, dim=0)
        rays_dir = torch.split(self.rays_dir, size, dim=0)
        return [RayPack(ray_ori, ray_dir) for ray_ori, ray_dir in zip(rays_orig, rays_dir)]


@dataclass
class PBRMaterial:
    material_id: int
    diffuse_map: Optional[torch.Tensor] = None  # (H, W, 4)
    emissive_map: Optional[torch.Tensor] = None  # (H, W, 4)
    metallic_roughness_map: Optional[torch.Tensor] = None  # (H, W, 2)
    normal_map: Optional[torch.Tensor] = None  # (H, W, 4)
    diffuse_factor: torch.Tensor = None  # (4,)
    emissive_factor: torch.Tensor = None  # (3,)
    metallic_factor: float = 0.0
    roughness_factor: float = 0.0
    alpha_mode: int = 0
    alpha_cutoff: float = 0.5
    transmission_factor: float = 0.0
    ior: float = 1.0


#################################
##     --- OPTIX STRUCTS ---   ##
#################################


class OptixPlaygroundRenderOptions(IntEnum):
    NONE = 0
    SMOOTH_NORMALS = 1
    DISABLE_GAUSSIAN_TRACING = 2
    DISABLE_PBR_TEXTURES = 4


class OptixPrimitiveTypes(IntEnum):
    NONE = 0
    MIRROR = 1
    GLASS = 2
    DIFFUSE = 3

    @classmethod
    def names(cls):
        return ['None', 'Mirror', 'Glass', 'Diffuse Mesh']


@dataclass
class OptixPrimitive:
    geometry_type: str = None
    vertices: torch.Tensor = None
    triangles: torch.Tensor = None
    vertex_normals: torch.Tensor = None
    has_tangents: torch.Tensor = None
    vertex_tangents: torch.Tensor = None
    material_uv: Optional[torch.Tensor] = None
    material_id: Optional[torch.Tensor] = None
    primitive_type: Optional[OptixPrimitiveTypes] = None
    primitive_type_tensor: torch.Tensor = None

    # Mirrors
    reflectance_scatter: torch.Tensor = None
    # Glass
    refractive_index: Optional[float] = None
    refractive_index_tensor: torch.Tensor = None

    transform: ObjectTransform() = None

    @classmethod
    def stack(cls, primitives):
        device = primitives[0].vertices.device
        vertices = torch.cat([p.vertices for p in primitives], dim=0)
        v_offset = torch.tensor([0] + [p.vertices.shape[0] for p in primitives[:-1]], device=device)
        v_offset = torch.cumsum(v_offset, dim=0)
        triangles = torch.cat([p.triangles + offset for p, offset in zip(primitives, v_offset)], dim=0)

        return OptixPrimitive(
            vertices=vertices.float(),
            triangles=triangles.int(),
            vertex_normals=torch.cat([p.vertex_normals for p in primitives], dim=0).float(),
            has_tangents=torch.cat([p.has_tangents for p in primitives], dim=0).bool(),
            vertex_tangents=torch.cat([p.vertex_tangents for p in primitives], dim=0).float(),
            material_uv=torch.cat([p.material_uv for p in primitives if p.material_uv is not None], dim=0).float(),
            material_id=torch.cat([p.material_id for p in primitives if p.material_id is not None], dim=0).int(),
            primitive_type_tensor=torch.cat([p.primitive_type_tensor for p in primitives], dim=0).int(),
            reflectance_scatter=torch.cat([p.reflectance_scatter for p in primitives], dim=0).float(),
            refractive_index_tensor=torch.cat([p.refractive_index_tensor for p in primitives], dim=0).float()
        )

    def apply_transform(self):
        model_matrix = self.transform.model_matrix()
        rs_comp = model_matrix[None, :3, :3]
        t_comp = model_matrix[None, :3, 3:]
        transformed_verts = (rs_comp @ self.vertices[:, :, None] + t_comp).squeeze(2)

        normal_matrix = self.transform.rotation_matrix()[None, :3, :3]
        transformed_normals = (normal_matrix @ self.vertex_normals[:, :, None]).squeeze(2)
        transformed_normals = torch.nn.functional.normalize(transformed_normals)

        transformed_tangents = (normal_matrix @ self.vertex_tangents[:, :, None]).squeeze(2)
        transformed_tangents = torch.nn.functional.normalize(transformed_tangents)

        return OptixPrimitive(
            vertices=transformed_verts,
            triangles=self.triangles,
            vertex_normals=transformed_normals,
            vertex_tangents=transformed_tangents,
            has_tangents=self.has_tangents,
            material_uv=self.material_uv,
            material_id=self.material_id,
            primitive_type=self.primitive_type,
            primitive_type_tensor=self.primitive_type_tensor,
            reflectance_scatter=self.reflectance_scatter,
            refractive_index=self.refractive_index,
            refractive_index_tensor=self.refractive_index_tensor,
            transform=ObjectTransform(device=self.transform.device)
        )


class Primitives:
    SUPPORTED_MESH_EXTENSIONS = ['.obj', '.glb']
    DEFAULT_REFRACTIVE_INDEX = 1.33
    SCALE_OF_NEW_MESH_TO_SMALL_SCENE = 0.5    # Mesh will be this percent of the scene on longest axis

    def __init__(self, tracer, mesh_assets_folder, enable_envmap=False, use_envmap_as_background=False,
                 scene_scale=None):
        # str -> str ; shape name to filename + extension
        self.assets = self.register_available_assets(assets_folder=mesh_assets_folder)
        self.tracer = tracer
        self.enabled = True
        self.objects = dict()
        self.use_smooth_normals = True
        self.disable_gaussian_tracing = False
        self.disable_pbr_textures = False
        self.force_white_bg = False
        self.enable_envmap = enable_envmap
        self.use_envmap_as_background = use_envmap_as_background

        if scene_scale is None:
            self.scene_scale = torch.tensor([1.0, 1.0, 1.0], device='cpu')
        else:
            self.scene_scale = scene_scale.cpu()

        self.stacked_fields = None
        self.dirty = True

        self.instance_counter = dict()  # Counts number of primitives of each geometry

        device = 'cuda'

        self.registered_materials = self.register_default_materials(device)  # str -> PBRMaterial

    def register_available_assets(self, assets_folder):
        available_assets = {Path(asset).stem.capitalize(): os.path.join(assets_folder, asset)
                            for asset in os.listdir(assets_folder)
                            if Path(asset).suffix in Primitives.SUPPORTED_MESH_EXTENSIONS}
        # Procedural shapes are added manually
        available_assets['Quad'] = None
        return available_assets # i.e. {MeshName: /path/to/mesh.glb}

    def register_default_materials(self, device):
        checkboard_res = 512
        checkboard_square = 20
        checkboard_texture = torch.tensor([0.25, 0.25, 0.25, 1.0],
                                          device=device, dtype=torch.float32).repeat(checkboard_res, checkboard_res, 1)
        for i in range(checkboard_res // checkboard_square):
            for j in range(checkboard_res // checkboard_square):
                start_x = (2 * i + j % 2) * checkboard_square
                end_x = min((2 * i + 1 + j % 2) * checkboard_square, checkboard_res)
                start_y = j * checkboard_square
                end_y = min((j + 1) * checkboard_square, checkboard_res)
                checkboard_texture[start_y:end_y, start_x:end_x, :3] = 0.5
        default_materials = dict(
            solid=PBRMaterial(
                material_id=0,
                diffuse_map=torch.tensor([130 / 255.0, 193 / 255.0, 255 / 255.0, 1.0],
                                         device=device, dtype=torch.float32).expand(2, 2, 4),
                diffuse_factor=torch.ones(4, device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.0,
                transmission_factor=0.0,
                ior=1.0
            ),
            checkboard=PBRMaterial(
                material_id=1,
                diffuse_map=checkboard_texture.contiguous(),
                diffuse_factor=torch.ones(4, device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.0,
                transmission_factor=0.0,
                ior=1.0
            )
        )
        return default_materials

    def set_mesh_scale_to_scene(self, mesh, transform):
        """
        Uses heuristics to scale the mesh so it appears nice:
        1) mesh rescaled to unit size
        2) if the scene is small, the mesh is rescaled to SCALE_OF_NEW_MESH_TO_SMALL_SCENE of the scene scale
        """
        mesh_scale = ((mesh.vertices.max(dim=0)[0] - mesh.vertices.min(dim=0)[0]).cpu()).to(transform.device)
        transform.scale(1.0 / mesh_scale.max())
        if self.scene_scale.max() > 5.0:    # Don't scale for large scenes
            return
        adjusted_scale = self.SCALE_OF_NEW_MESH_TO_SMALL_SCENE * self.scene_scale.to(transform.device)
        largest_axis_scale = adjusted_scale.max()
        transform.scale(largest_axis_scale)

    def add_primitive(self, geometry_type: str, primitive_type: OptixPrimitiveTypes, device):
        if geometry_type not in self.instance_counter:
            self.instance_counter[geometry_type] = 1
        else:
            self.instance_counter[geometry_type] += 1
        name = f"{geometry_type} {self.instance_counter[geometry_type]}"

        mesh = self.create_geometry(geometry_type, device)

        # Generate tangents mas, if available
        num_verts = len(mesh.vertices)
        num_faces = len(mesh.faces)
        has_tangents = torch.ones([num_verts, 1], device=device, dtype=torch.bool) \
            if mesh.vertex_tangents is not None \
            else torch.zeros([num_verts, 1], device=device, dtype=torch.bool)
        # Create identity transform and set scale to scene size
        transform = ObjectTransform(device=device)
        self.set_mesh_scale_to_scene(mesh, transform)
        # Face attributes
        prim_type_tensor = mesh.faces.new_full(size=(num_faces,), fill_value=primitive_type.value)
        reflectance_scatter = mesh.faces.new_zeros(size=(num_faces,))
        refractive_index = Primitives.DEFAULT_REFRACTIVE_INDEX
        refractive_index_tensor = mesh.faces.new_full(size=(num_faces,), fill_value=refractive_index)

        self.objects[name] = OptixPrimitive(
            geometry_type=geometry_type,
            vertices=mesh.vertices.float(),
            triangles=mesh.faces.int(),
            vertex_normals=mesh.vertex_normals.float(),
            has_tangents=has_tangents.bool(),
            vertex_tangents=mesh.vertex_tangents.float(),
            material_uv=mesh.face_uvs.float(),
            material_id=mesh.material_assignments.unsqueeze(1).int(),
            primitive_type=primitive_type,
            primitive_type_tensor=prim_type_tensor.int(),
            reflectance_scatter=reflectance_scatter.float(),
            refractive_index=refractive_index,
            refractive_index_tensor=refractive_index_tensor.float(),
            transform=transform
        )

    def remove_primitive(self, name: str):
        del self.objects[name]
        self.rebuild_bvh_if_needed(True, True)

    def duplicate_primitive(self, name: str):
        prim = self.objects[name]
        geometry_type = prim.geometry_type
        self.instance_counter[prim.geometry_type] += 1
        name = f"{geometry_type} {self.instance_counter[geometry_type]}"
        self.objects[name] = copy.deepcopy(prim)
        self.rebuild_bvh_if_needed(True, True)

    def register_materials(self, materials, model_name: str):
        """ Registers list of material dictionaries.
        """
        mat_idx_to_mat_id = torch.full([len(materials)], -1)
        for mat_idx, mat in enumerate(materials):
            material_name = f'{model_name}${mat["material_name"]}'
            if material_name not in self.registered_materials:
                self.registered_materials[material_name] = PBRMaterial(
                    material_id=len(self.registered_materials),
                    diffuse_map=mat['diffuse_map'],
                    emissive_map=mat['emissive_map'],
                    metallic_roughness_map=mat['metallic_roughness_map'],
                    normal_map=mat['normal_map'],
                    diffuse_factor=mat['diffuse_factor'],
                    emissive_factor=mat['emissive_factor'],
                    metallic_factor=mat['metallic_factor'],
                    roughness_factor=mat['roughness_factor'],
                    alpha_mode=mat['alpha_mode'],
                    alpha_cutoff=mat['alpha_cutoff'],
                    transmission_factor=mat['transmission_factor'],
                    ior=mat['ior']
                )
            mat_idx_to_mat_id[mat_idx] = self.registered_materials[material_name].material_id
        return mat_idx_to_mat_id

    def create_geometry(self, geometry_type: str, device):
        match geometry_type:
            case 'Quad':
                MS = 1.0
                MZ = 2.5
                v0 = [-MS, -MS, MZ]
                v1 = [-MS, +MS, MZ]
                v2 = [+MS, -MS, MZ]
                v3 = [+MS, +MS, MZ]
                faces = torch.tensor([[0, 1, 2], [2, 1, 3]])
                vertex_uvs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
                mesh = create_procedural_mesh(
                    vertices=torch.tensor([v0, v1, v2, v3]),
                    faces=faces,
                    face_uvs=vertex_uvs[faces].contiguous(), # (F, 3, 2)
                    device=device
                )
            case _:
                mesh_path = self.assets[geometry_type]
                mesh = load_mesh(mesh_path, device)
                materials = load_materials(mesh, device)
                if len(materials) > 0:
                    load_missing_material_info(mesh_path, materials, device)
                    material_index_mapping = self.register_materials(materials=materials, model_name=geometry_type)
                    # Update material assignments to match playground material registry
                    material_index_mapping = material_index_mapping.to(device=device)
                    material_id = mesh.material_assignments.to(device=device, dtype=torch.long)
                    mesh.material_assignments = material_index_mapping[material_id].int()
        # Always use default material, if no materials were specified
        mesh.material_assignments = torch.max(mesh.material_assignments, torch.zeros_like(mesh.material_assignments))
        return mesh

    def recompute_stacked_buffers(self):
        objects = [p.apply_transform() for p in self.objects.values()]
        # Recompute primitive type tensor
        for obj in objects:
            f = obj.triangles
            num_faces = f.shape[0]
            obj.primitive_type_tensor = f.new_full(size=(num_faces,), fill_value=obj.primitive_type.value)
            obj.refractive_index_tensor = f.new_full(size=(num_faces,),
                                                     fill_value=obj.refractive_index, dtype=torch.float)

        # Stack fields again
        self.stacked_fields = None
        if self.has_visible_objects():
            self.stacked_fields = OptixPrimitive.stack([p for p in objects if p.primitive_type != OptixPrimitiveTypes.NONE])

    def has_visible_objects(self):
        return len([p for p in self.objects.values() if p.primitive_type != OptixPrimitiveTypes.NONE]) > 0

    @torch.cuda.nvtx.range("rebuild_bvh (prim)")
    def rebuild_bvh_if_needed(self, force=False, rebuild=True):
        if self.dirty or force:
            if self.has_visible_objects():
                self.recompute_stacked_buffers()
                self.tracer.build_mesh_acc(
                    mesh_vertices=self.stacked_fields.vertices,
                    mesh_faces=self.stacked_fields.triangles,
                    rebuild=rebuild,
                    allow_update=True
                )
            else:
                self.tracer.build_mesh_acc(
                    mesh_vertices=torch.zeros([3, 3], dtype=torch.float, device='cuda'),
                    mesh_faces=torch.zeros([1, 3], dtype=torch.int, device='cuda'),
                    rebuild=True,
                    allow_update=True
                )
        self.dirty = False

#################################
##       --- Renderer      --- ##
#################################

class Engine3DGRUT:
    """
    An interface to the core functionality of rendering 3dgrt with secondary ray effects & mesh primitives.
    Important methods:
        - render_pass(): renders a single frame pass. Intended to be called from interactive gui viewers,
            where FPS should be maintained. Repeated calls to render_pass() will add additional details to
            the frame if multipass rendering effects are on (i.e. antialiasing, depth of field).
            Once all passes have been rendered, calling this function again will result in returning
            cached frames.
        - render(): renders a complete frame, possibly consisting of multi-passes. Intended for offline,
            high quality renderings.
        - invalidate_materials(): if any of the mesh materials in the scene have changed, viewers should
            mark the materials as invalid, to signal the engine they should be resynced to the gpu.
        - is_dirty(): returns true if the canvas state have changed since the last pass was rendered.
    Important members:
        - scene_mog: a reference to the gaussians model viewed within the scene.
        - primitives: a component managing all mesh primitives in the scene.
        - video_recorder: a component used to render videos of camera trajectories from within the scene
        -
    """
    DEFAULT_DEVICE = torch.device('cuda')
    AVAILABLE_CAMERAS = ['Pinhole', 'Fisheye']
    ANTIALIASING_MODES = ['4x MSAA', '8x MSAA', '16x MSAA', 'Quasi-Random (Sobol)']

    def __init__(self, gs_object, mesh_assets_folder, default_config):
        self.scene_mog, self.scene_name = self.load_3dgrt_object(gs_object, config_name=default_config)
        self.tracer = Tracer(self.scene_mog.conf)
        device = self.scene_mog.device

        self.envmap = None  # Currently disabled
        self.frame_id = 0

        # -- Outwards facing, these are the useful settings to configure --
        """ Type of camera used to render the scene """
        self.camera_type = 'Pinhole'
        """ Camera field of view """
        self.camera_fov = 45.0
        """ Toggles depth of field on / off """
        self.use_depth_of_field = False
        """ Component managing the depth of field settings in the scene """
        self.depth_of_field = DepthOfField(aperture_size=0.01, focus_z=1.0)
        """ Toggles antialiasing on / off """
        self.use_spp = True
        """ Currently set antialiasing mode """
        self.antialiasing_mode = '4x MSAA'
        """ Component managing the antialiasing settings in the scene """
        self.spp = SPP(mode='msaa', spp=4, device=device)
        """ Gamma correction factor """
        self.gamma_correction = 1.0
        """ Maximum number of PBR material bounces (transmissions & refractions, reflections) """
        self.max_pbr_bounces = 15
        """ If enabled, will use the optix denoiser as post-processing """
        self.use_optix_denoiser = True

        scene_scale = self.scene_mog.positions.max(dim=0)[0] - self.scene_mog.positions.min(dim=0)[0]
        self.primitives = Primitives(
            tracer=self.tracer,
            mesh_assets_folder=mesh_assets_folder,
            enable_envmap=self.envmap is not None,
            use_envmap_as_background=self.envmap is not None,
            scene_scale=scene_scale
        )
        self.primitives.add_primitive(geometry_type='Sphere', primitive_type=OptixPrimitiveTypes.GLASS, device=device)
        self.rebuild_bvh(self.scene_mog)
        if self.envmap is not None:
            self.primitives.force_white_bg = False

        self.last_state = dict(
            camera=None,
            rgb=None,
            opacity=None
        )

        self.video_recorder = VideoRecorder(renderer=self)

        """ When this flag is toggled on, the state of the materials have changed they need to be re-uploaded to device
        """
        self.is_materials_dirty = False

    def _accumulate_to_buffer(self, prev_frames, new_frame, num_frames_accumulated, gamma, batch_size=1):
        prev_frames = torch.pow(prev_frames, gamma)
        buffer = ((prev_frames * num_frames_accumulated) + new_frame) / (num_frames_accumulated + batch_size)
        buffer = torch.pow(buffer, 1.0 / gamma)
        return buffer

    @torch.cuda.nvtx.range("_render_depth_of_field_buffer")
    def _render_depth_of_field_buffer(self, rb, camera, rays):
        if self.use_depth_of_field and self.depth_of_field.has_more_to_accumulate():
            # Store current spp index
            i = self.depth_of_field.spp_accumulated_for_frame
            extrinsics_R = camera.R.squeeze(0).to(dtype=rays.rays_ori.dtype)
            dof_rays_ori, dof_rays_dir = self.depth_of_field(extrinsics_R, rays)
            if not self.primitives.enabled or not self.primitives.has_visible_objects():
                dof_rb = self.scene_mog.trace(rays_o=dof_rays_ori, rays_d=dof_rays_dir)
            else:
                dof_rb = self._render_playground_hybrid(dof_rays_ori, dof_rays_dir)

            rb['rgb'] = self._accumulate_to_buffer(rb['rgb'], dof_rb['pred_rgb'], i, self.gamma_correction)
            rb['opacity'] = (rb['opacity'] * i + dof_rb['pred_opacity']) / (i + 1)

    def _render_spp_buffer(self, rb, rays):
        if self.use_spp and self.spp.has_more_to_accumulate():
            # Store current spp index
            i = self.spp.spp_accumulated_for_frame

            if not self.primitives.enabled or not self.primitives.has_visible_objects():
                spp_rb = self.scene_mog.trace(rays_o=rays.rays_ori, rays_d=rays.rays_dir)
            else:
                spp_rb = self._render_playground_hybrid(rays.rays_ori, rays.rays_dir)
            batch_rgb = spp_rb['pred_rgb'].sum(dim=0).unsqueeze(0)
            rb['rgb'] = self._accumulate_to_buffer(rb['rgb'], batch_rgb, i, self.gamma_correction,
                                                   batch_size=self.spp.batch_size)
            rb['opacity'] = (rb['opacity'] * i + spp_rb['pred_opacity']) / (i + self.spp.batch_size)

    @torch.cuda.nvtx.range(f"playground._render_playground_hybrid")
    def _render_playground_hybrid(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> Dict[str, torch.Tensor]:
        mog = self.scene_mog
        playground_render_opts = 0
        if self.primitives.use_smooth_normals:
            playground_render_opts |= OptixPlaygroundRenderOptions.SMOOTH_NORMALS
        if self.primitives.disable_gaussian_tracing:
            playground_render_opts |= OptixPlaygroundRenderOptions.DISABLE_GAUSSIAN_TRACING
        if self.primitives.disable_pbr_textures:
            playground_render_opts |= OptixPlaygroundRenderOptions.DISABLE_PBR_TEXTURES

        self.primitives.rebuild_bvh_if_needed()

        envmap = self.envmap
        if self.primitives.force_white_bg:
            background_color = torch.ones(3)
            envmap = None
        elif isinstance(mog.background, BackgroundColor):
            background_color = mog.background.color
        else:
            background_color = torch.zeros(3)

        rendered_results = self.tracer.render_playground(
            gaussians=mog,
            ray_o=rays_o,
            ray_d=rays_d,
            playground_opts=playground_render_opts,
            mesh_faces=self.primitives.stacked_fields.triangles,
            vertex_normals=self.primitives.stacked_fields.vertex_normals,
            vertex_tangents=self.primitives.stacked_fields.vertex_tangents,
            vertex_tangents_mask=self.primitives.stacked_fields.has_tangents,
            primitive_type=self.primitives.stacked_fields.primitive_type_tensor[:, None],
            frame_id=self.frame_id,
            ray_max_t=None,
            material_uv=self.primitives.stacked_fields.material_uv,
            material_id=self.primitives.stacked_fields.material_id,
            materials=sorted(self.primitives.registered_materials.values(), key=lambda mat: mat.material_id),
            is_sync_materials=self.is_materials_dirty,
            refractive_index=self.primitives.stacked_fields.refractive_index_tensor[:, None],
            background_color=background_color,
            envmap=envmap,
            enable_envmap=self.primitives.enable_envmap,
            use_envmap_as_background=self.primitives.use_envmap_as_background,
            max_pbr_bounces=self.max_pbr_bounces
        )

        pred_rgb = rendered_results['pred_rgb']
        pred_opacity = rendered_results['pred_opacity']

        if envmap is None or not self.primitives.use_envmap_as_background:
            if self.primitives.force_white_bg:
                pred_rgb += (1.0 - pred_opacity)
            else:
                poses = torch.tensor([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]
                ], dtype=torch.float32)
                pred_rgb, pred_opacity = mog.background(
                    poses.contiguous(),
                    rendered_results['last_ray_d'].contiguous(),
                    pred_rgb,
                    pred_opacity,
                    False
                )

        # Mark materials as uploaded
        self.is_materials_dirty = False

        # Advance frame id (for i.e., random number generator) and avoid int32 overflow
        self.frame_id = self.frame_id + self.spp.batch_size if self.frame_id <= (2 ** 31 - 1) else 0

        pred_rgb = torch.clamp(pred_rgb, 0.0, 1.0)  # Make sure image pixels are in valid range

        rendered_results['pred_rgb'] = pred_rgb
        return rendered_results

    @torch.cuda.nvtx.range("render_pass")
    @torch.no_grad()
    def render_pass(self, camera: Camera, is_first_pass: bool):
        """
        Renders a single pass of the current frame from the camera point of view.
        render_pass() is suitable for online rendering situations, where interactivity is preferred.
        If is_first_pass is true, the first pass excluding any antialiasing and depth-of-field effects will be rendered.
        If is_first_pass is false and multi-pass effects are enabled, additional effects will be gradually rendered
        every time this function is called.
        This function returns the last cached frame if no more passes remain.
        """
        # Rendering 3dgrut requires camera to run on cuda device -- avoid crashing
        if camera.device.type == 'cpu':
            camera = camera.cuda()

        is_use_spp = not is_first_pass and not self.use_depth_of_field and self.use_spp
        rays = self.raygen(camera, use_spp=is_use_spp)

        if is_first_pass:
            if not self.primitives.enabled or not self.primitives.has_visible_objects():
                rb = self.scene_mog.trace(rays_o=rays.rays_ori, rays_d=rays.rays_dir)
            else:
                rb = self._render_playground_hybrid(rays.rays_ori, rays.rays_dir)

            rb = dict(rgb=rb['pred_rgb'], opacity=rb['pred_opacity'])
            rb['rgb'] = torch.pow(rb['rgb'], 1.0 / self.gamma_correction)
            rb['rgb'] = rb['rgb'].mean(dim=0).unsqueeze(0)
            rb['opacity'] = rb['opacity'].mean(dim=0).unsqueeze(0)
            self.spp.reset_accumulation()
            self.depth_of_field.reset_accumulation()
        else:
            # Render accumulated effects, i.e. depth of field
            rb = dict(rgb=self.last_state['rgb_buffer'], opacity=self.last_state['opacity'])
            if self.use_depth_of_field:
                self._render_depth_of_field_buffer(rb, camera, rays)
            elif self.use_spp:
                self._render_spp_buffer(rb, rays)

        # Keep a noisy version of the accumulated rgb buffer so we don't repeat denoising per frame
        rb['rgb_buffer'] = rb['rgb']
        if self.use_optix_denoiser:
            rb['rgb'] = self.tracer.denoise(rb['rgb'])

        if rays.mask is not None:  # mask is for masking away pixels out of view for, i.e. fisheye
            mask = rays.mask[None, :, :, 0]
            rb['rgb'][mask] = 0.0
            rb['rgb_buffer'][mask] = 0.0
            rb['opacity'][mask] = 0.0

        self.cache_last_state(camera=camera, renderbuffers=rb, canvas_size=[camera.height, camera.width])
        return rb

    @torch.cuda.nvtx.range("render")
    def render(self, camera: Camera) -> Dict[str, torch.Tensor]:
        """
        Renders a single frame, possibly consisting of multiple passes if any visual effects are toggled on.
        render() is suitable for offline rendering situations, where high quality is preferred.

        By default, 3dgrt requires only a single pass to render.
        Toggling on antialiasing and depth of field may require additional samples which require additional passes.

        The returned dictionary of rendered buffers always includes the 'rgb' and 'opacity' buffers.
        """
        renderbuffers = self.render_pass(camera, is_first_pass=True)
        while self.has_progressive_effects_to_render():
            renderbuffers = self.render_pass(camera, is_first_pass=False)
        return renderbuffers

    def invalidate_materials_on_gpu(self):
        """ Marks the materials on GPU as out of date.
        Materials and textures will be uploaded to the GPU during the next rendering pass.
        """
        self.is_materials_dirty = True

    @torch.cuda.nvtx.range("load_3dgrt_object")
    def load_3dgrt_object(self, object_path, config_name='apps/colmap_3dgrt.yaml'):
        """
        Loads a 3dgrt object model from object_path.
        If object is in .ingp, .ply format, the model will be initialized with a config loaded from config_name.
        """
        def load_default_config():
            from hydra.compose import compose
            from hydra.initialize import initialize
            with initialize(version_base=None, config_path='../configs'):
                conf = compose(config_name=config_name)
            return conf

        if object_path.endswith('.pt'):
            checkpoint = torch.load(object_path)
            conf = checkpoint["config"]
            if conf.render['method'] != '3dgrt':
                conf = load_default_config()
            model = MixtureOfGaussians(conf)
            model.init_from_checkpoint(checkpoint, setup_optimizer=False)
            object_name = conf.experiment_name
        elif object_path.endswith('.ingp'):
            conf = load_default_config()
            model = MixtureOfGaussians(conf)
            model.init_from_ingp(object_path, init_model=False)
            object_name = Path(object_path).stem
        elif object_path.endswith('.ply'):
            conf = load_default_config()
            model = MixtureOfGaussians(conf)
            model.init_from_ply(object_path, init_model=False)
            object_name = Path(object_path).stem
        else:
            raise ValueError(f"Unknown object type: {object_path}")

        if object_name is None or len(object_name) == 0:
            object_name = Path(object_path).stem    # Fallback to pick object name from path, if none specified

        model.build_acc(rebuild=True)

        return model, object_name

    @torch.cuda.nvtx.range("rebuild_bvh (mog)")
    def rebuild_bvh(self, scene_mog):
        """ Rebuilds all BVHs used by playground:
        1. 3dgrt BVH of proxy shapes encapsulating gaussian particles
        2. Mesh BVH holding faces of primitives use in the playground
        """
        rebuild = True
        self.tracer.build_gs_acc(gaussians=scene_mog, rebuild=rebuild)
        self.primitives.rebuild_bvh_if_needed()

    def did_camera_change(self, camera) -> bool:
        current_view_matrix = camera.view_matrix()
        cached_camera_matrix = self.last_state.get('camera')
        is_camera_changed = cached_camera_matrix is not None and (cached_camera_matrix != current_view_matrix).any()
        return is_camera_changed

    def has_cached_buffers(self) -> bool:
        return self.last_state.get('rgb') is not None and self.last_state.get('opacity') is not None

    def has_progressive_effects_to_render(self) -> bool:
        has_dof_buffers_to_render = self.use_depth_of_field and \
                                    self.depth_of_field.spp_accumulated_for_frame <= self.depth_of_field.spp
        has_spp_buffers_to_render = not self.use_depth_of_field and \
                                    self.use_spp and self.spp.spp_accumulated_for_frame <= self.spp.spp
        return has_dof_buffers_to_render or has_spp_buffers_to_render

    def is_dirty(self, camera):
        """ Returns true if the state of the scene have changed since last time the canvas was rendered. """
        # Force dirty flag is on
        if self.is_materials_dirty:
            return True
        if self.did_camera_change(camera):
            return True
        if not self.has_cached_buffers():
            return True
        return False

    def cache_last_state(self, camera, renderbuffers, canvas_size):
        self.last_state['canvas_size'] = canvas_size
        self.last_state['camera'] = copy.deepcopy(camera.view_matrix())
        self.last_state['rgb'] = renderbuffers['rgb']
        self.last_state['rgb_buffer'] = renderbuffers['rgb_buffer']
        self.last_state['opacity'] = renderbuffers['opacity']

    def _raygen_pinhole(self, camera, jitter=None) -> RayPack:
        pixel_y, pixel_x = generate_centered_pixel_coords(camera.width, camera.height, device=camera.device)
        if jitter is not None:
            jitter = jitter.to(device=pixel_x.device)
            pixel_x += jitter[:, :, 0]
            pixel_y += jitter[:, :, 1]
        ray_grid = [pixel_y, pixel_x]
        rays_o, rays_d = generate_pinhole_rays(camera, coords_grid=ray_grid)

        return RayPack(
            rays_ori=rays_o.reshape(1, camera.height, camera.width, 3).float(),
            rays_dir=rays_d.reshape(1, camera.height, camera.width, 3).float(),
            pixel_x=torch.round(pixel_x - 0.5).squeeze(-1),
            pixel_y=torch.round(pixel_y - 0.5).squeeze(-1)
        )

    @torch.cuda.nvtx.range("_raygen_fisheye")
    def _raygen_fisheye(self, camera, jitter) -> RayPack:
        pixel_y, pixel_x = generate_centered_pixel_coords(
            camera.width, camera.height, device=camera.device
        )
        if jitter is not None:
            jitter = jitter.to(device=pixel_x.device)
            pixel_x += jitter[:, :, 0]
            pixel_y += jitter[:, :, 1]
        ray_grid = [pixel_y, pixel_x]
        rays_o, rays_d, mask = generate_fisheye_rays(camera, ray_grid)

        return RayPack(
            rays_ori=rays_o.reshape(1, camera.height, camera.width, 3).float(),
            rays_dir=rays_d.reshape(1, camera.height, camera.width, 3).float(),
            pixel_x=torch.round(pixel_x - 0.5).squeeze(-1),
            pixel_y=torch.round(pixel_y - 0.5).squeeze(-1),
            mask=mask.reshape(camera.height, camera.width, 1)
        )

    def raygen(self, camera, use_spp=False) -> RayPack:
        ray_batch_size = 1 if not use_spp else self.spp.batch_size
        rays = []
        for _ in range(ray_batch_size):
            jitter = self.spp(camera.height, camera.width) if use_spp and self.spp is not None else None
            if self.camera_type == 'Pinhole':
                next_rays = self._raygen_pinhole(camera, jitter)
            elif self.camera_type == 'Fisheye':
                next_rays = self._raygen_fisheye(camera, jitter)
            else:
                raise ValueError(f"Unknown camera type: {self.camera_type}")
            rays.append(next_rays)
        return RayPack(
            mask=rays[0].mask,
            pixel_x=rays[0].pixel_x,
            pixel_y=rays[0].pixel_y,
            rays_ori=torch.cat([r.rays_ori for r in rays], dim=0),
            rays_dir=torch.cat([r.rays_dir for r in rays], dim=0)
        )
