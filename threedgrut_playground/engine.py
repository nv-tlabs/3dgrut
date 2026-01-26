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
import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import kaolin
import torch
from kaolin.render.camera import (
    Camera,
    generate_centered_pixel_coords,
    generate_pinhole_rays,
)

from threedgrut.model.background import BackgroundColor
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger
from threedgrut_playground.tracer import Tracer
from threedgrut_playground.utils.depth_of_field import DepthOfField
from threedgrut_playground.utils.environment import Environment
from threedgrut_playground.utils.kaolin_future.fisheye import generate_fisheye_rays
from threedgrut_playground.utils.kaolin_future.transform import ObjectTransform
from threedgrut_playground.utils.mesh_io import (
    create_procedural_mesh,
    create_quad_mesh,
    load_materials,
    load_mesh,
    load_missing_material_info,
)
from threedgrut_playground.utils.spp import SPP
from threedgrut_playground.utils.video_out import VideoRecorder

#################################
##       --- Common ---        ##
#################################


@dataclass
class RayPack:
    """A container for ray tracing data representing a batch of rays.

    This class encapsulates all necessary information for ray tracing operations,
    including ray origins, directions, and optional pixel coordinates and masks.

    Attributes:
        rays_ori (torch.FloatTensor): Ray origin points of shape (B, H, W, 3) or (N, 3),
            where B is batch size, H and W are image dimensions, and N is number of rays.
            Values are in world coordinates.

        rays_dir (torch.FloatTensor): Ray direction vectors of shape (B, H, W, 3) or (N, 3),
            matching rays_ori shape. Directions should be normalized.

        pixel_x (Optional[torch.IntTensor]): X-coordinates of pixel samples corresponding to rays,
            shape (H, W). Used for mapping rays back to image space.

        pixel_y (Optional[torch.IntTensor]): Y-coordinates of pixel samples corresponding to rays,
            shape (H, W). Used for mapping rays back to image space.

        mask (Optional[torch.BoolTensor]): Boolean mask of shape (H, W, 1) indicating valid
            rays. Typically used for fisheye cameras where not all pixels have valid rays.
    """

    rays_ori: torch.FloatTensor
    rays_dir: torch.FloatTensor
    pixel_x: Optional[torch.IntTensor] = None
    pixel_y: Optional[torch.IntTensor] = None
    mask: Optional[torch.BoolTensor] = None

    def split(self, size: Optional[int] = None) -> List[RayPack]:
        """Splits a batch of rays into smaller batches for processing."""
        if size is None:
            return [self]
        assert self.rays_ori.ndim == 2 and self.rays_dir.ndim == 2, "Only 1D ray packs can be split"
        rays_orig = torch.split(self.rays_ori, size, dim=0)
        rays_dir = torch.split(self.rays_dir, size, dim=0)
        return [RayPack(ray_ori, ray_dir) for ray_ori, ray_dir in zip(rays_orig, rays_dir)]


@dataclass
class PBRMaterial:
    """A Physically Based Rendering (PBR) material representation, managed within the engine and passed to the
    path tracer. This class implements the metallic-roughness PBR material model, supporting both
    texture maps and constant factors. It follows the glTF 2.0 PBR specification
    with additional support for transmission and IOR (Index of Refraction).
    """

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
    """Render options to control the path tracer."""

    NONE = 0
    SMOOTH_NORMALS = 1  # Use smooth interpolated normals for shading
    DISABLE_GAUSSIAN_TRACING = 2  # Disable Gaussian volumetric tracing
    DISABLE_PBR_TEXTURES = 4  # Disable PBR textures


class OptixPrimitiveTypes(IntEnum):
    """Types of mesh primitives that can be rendered."""

    NONE = 0
    MIRROR = 1
    GLASS = 2
    DIFFUSE = 3
    PBR = 4

    @classmethod
    def names(cls):
        return ["None", "Mirror", "Glass", "Diffuse Mesh", "PBR Mesh"]


@dataclass
class OptixPrimitive:
    """Holds the data for a single mesh primitive to be rendered, or a stack of multiple primitives.
    That includes the geometry, material, and transform.
    Materials are managed within the engine and referenced by material_id.
    """

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
    def stack(cls, primitives: List[OptixPrimitive]) -> OptixPrimitive:
        """Stack a list of primitives into a single primitive.
        That is - all fields are concatenated along the first dimension.
        """
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
            refractive_index_tensor=torch.cat([p.refractive_index_tensor for p in primitives], dim=0).float(),
        )

    def apply_transform(self):
        """Apply the primitive's transform to the geometry."""
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
            transform=ObjectTransform(device=self.transform.device),
        )


def set_mesh_scale_to_scene(
    scene_scale: torch.Tensor,
    mesh: kaolin.rep.SurfaceMesh,
    transform: ObjectTransform,
    scale_of_new_mesh_to_small_scene=0.5,
) -> None:
    """Automatically scales a mesh to fit appropriately within the scene.

    This function is a heuristic that applies a two-step scaling process:
    1. Normalizes mesh to unit size by dividing by its largest dimension
    2. For small scenes (max dimension ≤ 5.0), scales the mesh to a proportion
       of the scene size

    Args:
        scene_scale (torch.Tensor): Overall scene dimensions of shape (3,)
        mesh (kaolin.rep.SurfaceMesh): Mesh to be scaled
        transform (ObjectTransform): Transform object to apply scaling to
        scale_of_new_mesh_to_small_scene (float, optional): For small scenes,
            mesh will be scaled to this fraction of scene size. Defaults to 0.5.

    Note:
        - Large scenes (max dimension > 5.0) only get unit normalization
        - Small scenes get additional scaling relative to scene size
        - All scaling is uniform (preserves mesh proportions)
        - Scaling is applied through the transform object
    """
    mesh_scale = ((mesh.vertices.max(dim=0)[0] - mesh.vertices.min(dim=0)[0]).cpu()).to(transform.device)
    transform.scale(1.0 / mesh_scale.max())
    if scene_scale.max() > 5.0:  # Don't scale for large scenes
        return
    adjusted_scale = scale_of_new_mesh_to_small_scene * scene_scale.to(transform.device)
    largest_axis_scale = adjusted_scale.max()
    transform.scale(largest_axis_scale)


class Primitives:
    """Manages mesh primitives and their materials within the 3DGRUT rendering engine.

    This class handles the lifecycle of mesh objects in the scene, including loading,
    transformation, material assignment, and BVH acceleration structure management.

    Viewers may reference this class to add primitives to the scene, or modify existing ones.

    Example usage:
    ```
        engine.primitives.add_primitive(geometry_type='Sphere', primitive_type=OptixPrimitiveTypes.GLASS, device=device)
        engine.primitives.remove_primitive(name='Sphere 1')
        for material_name, material in engine.primitives.registered_materials.items():
            print(material_name, material)
    ```
    """

    SUPPORTED_MESH_EXTENSIONS = [".obj", ".glb", ".gltf"]  # Supported mesh file formats
    DEFAULT_REFRACTIVE_INDEX = 1.33  # Default IOR for transparent materials
    PROCEDURAL_SHAPES: Dict[str, Callable[[torch.device], kaolin.rep.SurfaceMesh]] = dict(
        Quad=create_quad_mesh
    )  # Supported procedural shapes -> constructor function

    def __init__(
        self,
        tracer: Tracer,
        mesh_assets_folder: str,
        scene_scale: Optional[torch.Tensor] = None,
        mesh_autoscale_func: Callable[
            [torch.Tensor, kaolin.rep.SurfaceMesh, ObjectTransform], None
        ] = set_mesh_scale_to_scene,
        device: Optional[torch.device] = None,
    ):
        """Initialize the primitives manager for mesh rendering.

        Args:
            tracer (Tracer): OptixTracer instance for ray tracing, should reference the same tracer as the engine.
            mesh_assets_folder (str): Directory containing mesh asset files to load from.
            scene_scale (Optional[torch.Tensor], optional): Scene dimensions of shape (3,).
                Defaults to [1.0, 1.0, 1.0].
            mesh_autoscale_func (Callable, optional): Function to autoscale meshes when new primitives are added.
                Signature: (scene_scale, mesh, transform) -> None.
                The scene scale and mesh are used to decide how to manipulate the transform of the mesh.
                Defaults to set_mesh_scale_to_scene.
            device (Optional[torch.device], optional): Device for tensor operations.
        """

        """ Mapping of mesh names to file paths: str -> str ; shape name to filename + extension """
        self.assets: Dict[str, str] = self.register_available_assets(assets_folder=mesh_assets_folder)
        """ Reference to the OptixTracer instance """
        self.tracer = tracer
        """ Whether mesh primitive rendering is enabled """
        self.enabled: bool = True
        """ Holds existing primitives in the scene """
        self.objects: Dict[str, OptixPrimitive] = dict()
        """ Whether to use smooth normal interpolation """
        self.use_smooth_normals: bool = True
        """ Whether to skip PBR texture lookups """
        self.disable_pbr_textures: bool = False
        """ Overall scene scale factor, shape (3,) """
        if scene_scale is None:
            self.scene_scale: torch.Tensor = torch.tensor([1.0, 1.0, 1.0], device="cpu")
        else:
            self.scene_scale: torch.Tensor = scene_scale.cpu()
        self.stacked_fields = None
        """ Whether the mesh state have changed and BVH needs rebuilding """
        self.dirty: bool = True
        """ Counts number of primitives of each geometry type """
        self.instance_counter: Dict[str, int] = dict()
        """ Holds all available materials loaded so far with any of the meshes, or pre-registered procedurally with the scene. """
        self.registered_materials: Dict[str, PBRMaterial] = self.register_default_materials(device)
        """ Mesh autoscale function to use when placing a new mesh in the scene"""
        self.mesh_autoscale_func = mesh_autoscale_func

    def register_available_assets(self, assets_folder: str) -> Dict[str, Optional[str]]:
        """Scans directory for supported mesh files and builds asset registry.
        Finds all supported mesh files in the given directory and maps their
        capitalized names to full file paths.

        Args:
            assets_folder (str): Directory containing mesh asset files

        Returns:
            Dict[str, str]: Mapping of capitalized mesh names to file paths,
                with procedural shapes mapping to None
        """
        available_assets = {
            Path(asset).stem.capitalize(): os.path.join(assets_folder, asset)
            for asset in os.listdir(assets_folder)
            if Path(asset).suffix in Primitives.SUPPORTED_MESH_EXTENSIONS
        }
        # Procedural shapes are added manually
        for shape in Primitives.PROCEDURAL_SHAPES.keys():
            available_assets[shape] = None
        return available_assets  # i.e. {MeshName: /path/to/mesh.glb}

    def register_default_materials(self, device) -> Dict[str, PBRMaterial]:
        """Registers default procedural materials which always load with the engine."""
        checkboard_res = 512
        checkboard_square = 20
        checkboard_texture = torch.tensor([0.25, 0.25, 0.25, 1.0], device=device, dtype=torch.float32).repeat(
            checkboard_res, checkboard_res, 1
        )
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
                diffuse_map=torch.tensor(
                    [130 / 255.0, 193 / 255.0, 255 / 255.0, 1.0], device=device, dtype=torch.float32
                ).expand(2, 2, 4),
                diffuse_factor=torch.ones(4, device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.0,
                transmission_factor=0.0,
                ior=1.0,
            ),
            checkboard=PBRMaterial(
                material_id=1,
                diffuse_map=checkboard_texture.contiguous(),
                diffuse_factor=torch.ones(4, device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.0,
                transmission_factor=0.0,
                ior=1.0,
            ),
            brushed_copper=PBRMaterial(
                material_id=2,
                diffuse_factor=torch.tensor([0.95, 0.64, 0.54, 1.0], device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=1.0,
                roughness_factor=0.5,
                transmission_factor=0.0,
                ior=1.1,
            ),
            blue_glass=PBRMaterial(
                material_id=3,
                diffuse_factor=torch.tensor([0.1, 0.2, 0.8, 0.8], device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.1,
                transmission_factor=0.95,
                ior=1.52,
            ),
            jade=PBRMaterial(
                material_id=4,
                diffuse_factor=torch.tensor([0.2, 0.8, 0.5, 0.9], device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.3,
                transmission_factor=0.4,
                ior=1.66,
            ),
            polished_marble=PBRMaterial(
                material_id=5,
                diffuse_factor=torch.tensor([0.9, 0.9, 0.95, 1.0], device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.1,
                transmission_factor=0.0,
                ior=1.6,
            ),
            diamond=PBRMaterial(
                material_id=6,
                diffuse_factor=torch.tensor([0.98, 0.98, 0.98, 0.2], device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.02,
                transmission_factor=0.99,
                ior=2.42,
            ),
            rose_gold=PBRMaterial(
                material_id=7,
                diffuse_factor=torch.tensor([0.92, 0.72, 0.75, 1.0], device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=1.0,
                roughness_factor=0.15,
                transmission_factor=0.0,
                ior=1.1,
            ),
            luminous_yellow=PBRMaterial(
                material_id=8,
                diffuse_factor=torch.tensor([0.2, 0.9, 0.3, 1.0], device=device, dtype=torch.float32),
                emissive_factor=torch.tensor([0.8, 0.8, 0.4], device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.7,
                transmission_factor=0.0,
                ior=1.0,
            ),
            ruby_red=PBRMaterial(
                material_id=9,
                diffuse_factor=torch.tensor([0.9, 0.1, 0.2, 0.9], device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.1,
                transmission_factor=0.3,
                ior=1.76,  # Ruby-like IOR
            ),
            blue_plastic=PBRMaterial(
                material_id=10,
                diffuse_factor=torch.tensor([0.1, 0.2, 0.8, 1.0], device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.4,
                transmission_factor=0.0,
                ior=1.45,
            ),
            oak_wood=PBRMaterial(
                material_id=11,
                diffuse_factor=torch.tensor([0.65, 0.5, 0.35, 1.0], device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.75,
                transmission_factor=0.0,
                ior=1.3,
            ),
            black_rubber=PBRMaterial(
                material_id=12,
                diffuse_factor=torch.tensor([0.1, 0.1, 0.1, 1.0], device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.9,
                transmission_factor=0.0,
                ior=1.5,
            ),
        )
        return default_materials

    def add_primitive(self, geometry_type: str, primitive_type: OptixPrimitiveTypes, device) -> None:
        """Creates a mesh from geometry type, sets up its materials and transforms,
        and adds it to the scene with automatic scaling.

        The new primitive instance is added to the scene with an auto-generated numbered name.

        Args:
            geometry_type (str): Type of geometry to create ('Quad', or any loaded asset from assets folder like 'Sphere').
            primitive_type (OptixPrimitiveTypes): Material type (MIRROR, GLASS, etc.)
            device: Device to create tensors on (i.e. 'cuda')

        Note:
            - Automatically generates unique name as "{geometry_type} {count}"
            - Scales mesh to fit scene using self.mesh_autoscale_func, if specified
            - Sets default refractive index for transparent materials
        """
        if geometry_type not in self.instance_counter:
            self.instance_counter[geometry_type] = 1
        else:
            self.instance_counter[geometry_type] += 1
        name = f"{geometry_type} {self.instance_counter[geometry_type]}"

        mesh = self.create_geometry(geometry_type, device)

        # Generate tangents mas, if available
        num_verts = len(mesh.vertices)
        num_faces = len(mesh.faces)
        is_precomputed_tangents = mesh.has_attribute("vertex_tangents") and mesh.vertex_tangents is not None
        has_tangents = (
            torch.ones([num_verts, 1], device=device, dtype=torch.bool)
            if is_precomputed_tangents
            else torch.zeros([num_verts, 1], device=device, dtype=torch.bool)
        )
        vertex_tangents = (
            mesh.vertex_tangents
            if is_precomputed_tangents
            else torch.zeros([num_verts, 3], device=device, dtype=torch.float)
        )

        # Create identity transform and set scale to scene size
        transform = ObjectTransform(device=device)
        if self.mesh_autoscale_func is not None:
            self.mesh_autoscale_func(self.scene_scale, mesh, transform)  # i.e. set_mesh_scale_to_scene()
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
            vertex_tangents=vertex_tangents.float(),
            material_uv=mesh.face_uvs.float(),
            material_id=mesh.material_assignments.unsqueeze(1).int(),
            primitive_type=primitive_type,
            primitive_type_tensor=prim_type_tensor.int(),
            reflectance_scatter=reflectance_scatter.float(),
            refractive_index=refractive_index,
            refractive_index_tensor=refractive_index_tensor.float(),
            transform=transform,
        )

    def remove_primitive(self, name: str) -> None:
        """Removes a primitive from the scene and updates the BVH.

        Args:
            name (str): Name of primitive to remove
        """
        del self.objects[name]
        self.rebuild_bvh_if_needed(True, True)

    def duplicate_primitive(self, name: str) -> None:
        """Creates a deep copy of an existing primitive with a new name.
        Forces BVH rebuild after duplication.

        Args:
            name (str): Name of primitive instance to duplicate
        """
        prim = self.objects[name]
        geometry_type = prim.geometry_type
        self.instance_counter[prim.geometry_type] += 1
        name = f"{geometry_type} {self.instance_counter[geometry_type]}"
        self.objects[name] = copy.deepcopy(prim)
        self.rebuild_bvh_if_needed(True, True)

    def register_materials(self, materials: List[Dict], model_name: str) -> torch.Tensor:
        """Registers new PBR materials and creates index mapping for material assignments in current scene registry.
        e.g: the returned tensor will map the material indices of the given materials to the material IDs in the current scene registry.

        Args:
            materials (List[Dict]): List of material property dictionaries
            model_name (str): Prefix for material names

        Returns:
            torch.Tensor: Mapping from material indices to registered material IDs.
                Shape (N,) where N is number of materials.

        Note:
            Material names are formatted as '{model_name}${material_name}'
        """
        mat_idx_to_mat_id = torch.full([len(materials)], -1)
        for mat_idx, mat in enumerate(materials):
            material_name = f'{model_name}${mat["material_name"]}'
            if material_name not in self.registered_materials:
                self.registered_materials[material_name] = PBRMaterial(
                    material_id=len(self.registered_materials),
                    diffuse_map=mat["diffuse_map"],
                    emissive_map=mat["emissive_map"],
                    metallic_roughness_map=mat["metallic_roughness_map"],
                    normal_map=mat["normal_map"],
                    diffuse_factor=mat["diffuse_factor"],
                    emissive_factor=mat["emissive_factor"],
                    metallic_factor=mat["metallic_factor"],
                    roughness_factor=mat["roughness_factor"],
                    alpha_mode=mat["alpha_mode"],
                    alpha_cutoff=mat["alpha_cutoff"],
                    transmission_factor=mat["transmission_factor"],
                    ior=mat["ior"],
                )
            mat_idx_to_mat_id[mat_idx] = self.registered_materials[material_name].material_id
        return mat_idx_to_mat_id

    def create_geometry(self, geometry_type: str, device) -> kaolin.rep.SurfaceMesh:
        """Creates mesh geometry either procedurally or from file.

        Args:
            geometry_type (str): Type of geometry to create ('Quad', or any loaded asset from assets folder like 'Sphere').
            device: Device to create tensors on

        Returns:
            Mesh object with:
                - vertices: Vertex positions
                - faces: Triangle indices
                - face_uvs: UV coordinates
                - material_assignments: Material IDs
                - vertex_normals: Normal vectors
                - vertex_tangents: Tangent vectors (if available)

        Note:
            - Handles both procedural shapes and loaded meshes
            - Automatically loads and registers materials for mesh files
            - Falls back to default material if none specified
        """
        if geometry_type in Primitives.PROCEDURAL_SHAPES:
            constructor = Primitives.PROCEDURAL_SHAPES[geometry_type]
            mesh = constructor(device)
        else:
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

    def recompute_stacked_buffers(self) -> None:
        """Updates internal buffers for all visible primitives.
        Applies transforms and updates per-face properties before
        stacking all visible primitives into combined buffers.
        """
        objects = [p.apply_transform() for p in self.objects.values()]
        # Recompute primitive type tensor
        for obj in objects:
            f = obj.triangles
            num_faces = f.shape[0]
            obj.primitive_type_tensor = f.new_full(size=(num_faces,), fill_value=obj.primitive_type.value)
            obj.refractive_index_tensor = f.new_full(
                size=(num_faces,), fill_value=obj.refractive_index, dtype=torch.float
            )

        # Stack fields again
        self.stacked_fields = None
        if self.has_visible_objects():
            self.stacked_fields = OptixPrimitive.stack(
                [p for p in objects if p.primitive_type != OptixPrimitiveTypes.NONE]
            )

    def has_visible_objects(self) -> bool:
        """Checks if scene contains any visible primitives.

        Returns:
            bool: True if any primitives have non-NONE type
        """
        return len([p for p in self.objects.values() if p.primitive_type != OptixPrimitiveTypes.NONE]) > 0

    @torch.cuda.nvtx.range("rebuild_bvh (prim)")
    def rebuild_bvh_if_needed(self, force: bool = False, rebuild: bool = True) -> None:
        """Rebuilds mesh primitives BVH acceleration structure if needed.
        Creates empty BVH if no visible objects.

        Args:
            force (bool): Force rebuild regardless of engine "dirty" state
            rebuild (bool): Whether to rebuild or update existing BVH
        """
        if self.dirty or force:
            if self.has_visible_objects():
                self.recompute_stacked_buffers()
                self.tracer.build_mesh_acc(
                    mesh_vertices=self.stacked_fields.vertices,
                    mesh_faces=self.stacked_fields.triangles,
                    rebuild=rebuild,
                    allow_update=True,
                )
            else:
                self.tracer.build_mesh_acc(
                    mesh_vertices=torch.zeros([3, 3], dtype=torch.float, device="cuda"),
                    mesh_faces=torch.zeros([1, 3], dtype=torch.int, device="cuda"),
                    rebuild=True,
                    allow_update=True,
                )
        self.dirty = False


#################################
##       --- Renderer      --- ##
#################################


class Engine3DGRUT:
    """An interface to the core functionality of rendering pretrained 3DGRUT scenes with secondary ray effects &
    mesh primitives. This engine supports loading pretrained scenes from:
        - 3D Gaussian Ray Tracing (3DGRT)
        - 3D Gaussian Unscented Transform (3DGUT)

    Scenes are loaded as converted 3DGRT scenes, rendered with a hybrid gaussian-mesh renderer,
    supporting advanced effects like PBR materials, depth of field, and antialiasing.

    Key Features:
        - Hybrid rendering of Gaussian splats and triangle meshes
        - PBR material system with support for textures and transparency
        - Multiple camera models (Pinhole, Fisheye)
        - Progressive rendering with antialiasing (MSAA, Sobol)
        - Depth of field effects
        - Environment mapping
        - OptixDenoiser support
        - Video recording capabilities

    Main Components:
        - scene_mog: 3D Gaussian model, representing the scene
        - primitives: Mesh primitive manager, for adding / removing / modifying mesh objects in the scene
        - video_recorder: Camera path recording utility, for recording videos of camera trajectories
        - depth_of_field: Depth of field effect manager
        - spp: Antialiasing effect manager

    Useful Settings:
        - camera_type: 'Pinhole' or 'Fisheye'
        - camera_fov: Field of view in degrees
        - use_depth_of_field: Enable/disable DoF effect
        - use_spp: Enable/disable antialiasing
        - antialiasing_mode: '4x MSAA', '8x MSAA', '16x MSAA', or 'Quasi-Random'
        - gamma_correction: Output gamma value
        - max_pbr_bounces: Maximum ray bounces for PBR effects
        - use_optix_denoiser: Enable/disable denoising

    Important methods:
        - render_pass(): renders a single frame pass. Intended to be called from interactive gui viewers,
            where higher FPS should be maintained. Repeated calls to render_pass() will add additional details to
            the frame if multipass rendering effects are on (i.e. antialiasing, depth of field).
            Once all passes have been rendered, calling this function again will result in returning
            cached frames.
        - render(): renders a complete frame, possibly consisting of multiple rendering passes.
            Intended for offline, high quality renderings.
        - invalidate_materials(): if any of the mesh materials in the scene have changed, viewers should
            mark the materials as invalid, to signal the engine they should be resynced to the gpu.
        - is_dirty(): returns true if the canvas state have changed since the last pass was rendered.

    Example:
        ```
        # Initialize engine with a 3D Gaussian model and mesh assets
        engine = Engine3DGRUT(
            gs_object="path/to/model.pt",
            mesh_assets_folder="path/to/assets",
            default_config="apps/colmap_3dgrt.yaml"
        )

        # Configure rendering settings
        engine.camera_type = 'Pinhole'
        engine.camera_fov = 45.0
        engine.use_spp = True
        engine.antialiasing_mode = '8x MSAA'

        # Add a glass sphere to the scene
        engine.primitives.add_primitive(
            geometry_type='Sphere',
            primitive_type=OptixPrimitiveTypes.GLASS,
            device='cuda'
        )

        # Render a frame
        camera = Camera(...)  # Configure camera parameters
        framebuffer = engine.render(camera)  # Full quality render

        # Or for interactive rendering:
        framebuffer = engine.render_pass(camera, is_first_pass=True)
        while engine.has_progressive_effects_to_render():
            framebuffer = engine.render_pass(camera, is_first_pass=False)

        # Rendered pixels are stored in framebuffer['rgb']
        ```
    """

    AVAILABLE_CAMERAS = ["Pinhole", "Fisheye"]
    ANTIALIASING_MODES = ["4x MSAA", "8x MSAA", "16x MSAA", "Quasi-Random (Sobol)"]

    def __init__(
        self, gs_object: str, mesh_assets_folder: str, default_config: str, envmap_assets_folder: Optional[str] = None
    ):
        self.scene_mog, self.scene_name = self.load_3dgrt_object(gs_object, config_name=default_config)
        self.tracer = Tracer(self.scene_mog.conf)
        self.device = self.scene_mog.device

        self.frame_id = 0
        """ Environment definitions, such as envmap, tonemapping used, and forced background color. """
        self.environment = Environment(envmap_assets_folder, device=self.device)

        # -- Outwards facing, these are the useful settings to configure --
        """ Type of camera used to render the scene """
        self.camera_type = "Pinhole"
        """ Camera field of view """
        self.camera_fov = 45.0
        """ Toggles depth of field on / off """
        self.use_depth_of_field = False
        """ Component managing the depth of field settings in the scene """
        self.depth_of_field = DepthOfField(aperture_size=0.01, focus_z=1.0)
        """ Toggles antialiasing on / off """
        self.use_spp = True
        """ Currently set antialiasing mode """
        self.antialiasing_mode = "4x MSAA"
        """ Component managing the antialiasing settings in the scene """
        self.spp = SPP(mode="msaa", spp=4, device=self.device)
        """ Gamma correction factor """
        self.gamma_correction = 1.0
        """ Maximum number of PBR material bounces (transmissions & refractions, reflections) """
        self.max_pbr_bounces = 15
        """ If enabled, will use the optix denoiser as post-processing """
        self.use_optix_denoiser = True
        """ Enables / disables gaussian rendering """
        self.disable_gaussian_tracing = False

        scene_scale = self.scene_mog.positions.max(dim=0)[0] - self.scene_mog.positions.min(dim=0)[0]
        self.primitives = Primitives(
            tracer=self.tracer, mesh_assets_folder=mesh_assets_folder, scene_scale=scene_scale, device=self.device
        )
        self.primitives.add_primitive(
            geometry_type="Sphere", primitive_type=OptixPrimitiveTypes.GLASS, device=self.device
        )
        self.rebuild_bvh(self.scene_mog)

        self.last_state = dict(camera=None, rgb=None, opacity=None)

        self.video_recorder = VideoRecorder(renderer=self)

        """ When this flag is toggled on, the state of the materials have changed they need to be re-uploaded to device
        """
        self.is_materials_dirty = False

    def _accumulate_to_buffer(self, prev_frames, new_frame, num_frames_accumulated, gamma, batch_size=1):
        """Accumulate a new frame to the buffer, using the previous frames and the current frame."""
        prev_frames = torch.pow(prev_frames, gamma)
        new_frame = self.environment.tonemap(new_frame)
        buffer = ((prev_frames * num_frames_accumulated) + new_frame) / (num_frames_accumulated + batch_size)
        buffer = torch.pow(buffer, 1.0 / gamma)
        return buffer

    @torch.cuda.nvtx.range("_render_depth_of_field_buffer")
    def _render_depth_of_field_buffer(self, rb, camera, rays):
        """Render the depth of field buffer, accumulating the previous frames and the current frame."""
        if self.use_depth_of_field and self.depth_of_field.has_more_to_accumulate():
            # Store current spp index
            i = self.depth_of_field.spp_accumulated_for_frame
            extrinsics_R = camera.R.squeeze(0).to(dtype=rays.rays_ori.dtype)
            dof_rays_ori, dof_rays_dir = self.depth_of_field(extrinsics_R, rays)
            # Disable path tracer under certain settings, useful for comparing with the vanilla volumetric tracer
            if not self.primitives.enabled or not self.primitives.has_visible_objects():
                if self.disable_gaussian_tracing:
                    dof_rb = dict(
                        pred_rgb=torch.zeros_like(rays.rays_ori),
                        pred_opacity=torch.zeros_like(rays.rays_ori[:, :, 0:1]),
                    )
                else:
                    dof_rb = self.scene_mog.trace(rays_o=dof_rays_ori, rays_d=dof_rays_dir)
            else:
                dof_rb = self._render_playground_hybrid(dof_rays_ori, dof_rays_dir)

            rb["rgb"] = self._accumulate_to_buffer(rb["rgb"], dof_rb["pred_rgb"], i, self.gamma_correction)
            rb["opacity"] = (rb["opacity"] * i + dof_rb["pred_opacity"]) / (i + 1)

    def _render_spp_buffer(self, rb, rays):
        """Render the SPP buffer, accumulating the previous frames and the current frame."""
        if self.use_spp and self.spp.has_more_to_accumulate():
            # Store current spp index
            i = self.spp.spp_accumulated_for_frame
            # Disable path tracer under certain settings, useful for comparing with the vanilla volumetric tracer
            if not self.primitives.enabled or not self.primitives.has_visible_objects():
                if self.disable_gaussian_tracing:
                    spp_rb = dict(
                        pred_rgb=torch.zeros_like(rays.rays_ori),
                        pred_opacity=torch.zeros_like(rays.rays_ori[:, :, 0:1]),
                    )
                else:
                    spp_rb = self.scene_mog.trace(rays_o=rays.rays_ori, rays_d=rays.rays_dir)
            else:
                spp_rb = self._render_playground_hybrid(rays.rays_ori, rays.rays_dir)
            batch_rgb = spp_rb["pred_rgb"].sum(dim=0).unsqueeze(0)
            rb["rgb"] = self._accumulate_to_buffer(
                rb["rgb"], batch_rgb, i, self.gamma_correction, batch_size=self.spp.batch_size
            )
            rb["opacity"] = (rb["opacity"] * i + spp_rb["pred_opacity"]) / (i + self.spp.batch_size)

    @torch.cuda.nvtx.range(f"playground._render_playground_hybrid")
    def _render_playground_hybrid(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Internal method for hybrid rendering of Gaussians and mesh primitives.
        Performs ray tracing through the scene, handling both 3D Gaussians and mesh
        primitives with PBR materials. Supports environment mapping and background
        handling.

        Args:
            rays_o (torch.Tensor): Ray origins of shape (B, H, W, 3)
            rays_d (torch.Tensor): Ray directions of shape (B, H, W, 3)

        Returns:
            Dict[str, torch.Tensor]: Rendering results containing:
                - 'pred_rgb': RGB colors of shape (B, H, W, 3), range [0, 1]
                - 'pred_opacity': Opacity values of shape (B, H, W, 1), range [0, 1]
                - 'last_ray_d': Final ray directions for background computation
                - Additional buffers from tracer.render_playground()

        Note:
            - Updates internal frame_id counter for random number generation
            - Handles material synchronization with GPU
            - Applies background color based on scene configuration
            - Clamps final RGB values to valid range
        """
        mog = self.scene_mog
        playground_render_opts = 0
        if self.primitives.use_smooth_normals:
            playground_render_opts |= OptixPlaygroundRenderOptions.SMOOTH_NORMALS
        if self.disable_gaussian_tracing:
            playground_render_opts |= OptixPlaygroundRenderOptions.DISABLE_GAUSSIAN_TRACING
        if self.primitives.disable_pbr_textures:
            playground_render_opts |= OptixPlaygroundRenderOptions.DISABLE_PBR_TEXTURES

        self.primitives.rebuild_bvh_if_needed()

        # !! Background color only affects RGB channels (not alpha) !!
        # force_white_bg: pixels are blended with white according to remaining alpha
        # mog.background is solid color: pixels are blended with background color of mog model
        # otherwise: no background color will be blended (== pixels are blended with black).
        envmap = self.environment.get_envmap()
        envmap_offset = self.environment.get_envmap_offset()

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
            envmap=envmap,
            envmap_offset=envmap_offset,
            max_pbr_bounces=self.max_pbr_bounces,
        )

        pred_rgb = rendered_results["pred_rgb"]
        pred_opacity = rendered_results["pred_opacity"]

        # If no envmap is used for background, saturate the color channels by blending the mog background
        if envmap is None or not self.environment.is_ignore_envmap():
            poses = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)
            pred_rgb, pred_opacity = mog.background(
                poses.contiguous(), rendered_results["last_ray_d"].contiguous(), pred_rgb, pred_opacity, False
            )

        # Mark materials as uploaded
        self.is_materials_dirty = False

        # Advance frame id (for i.e., random number generator) and avoid int32 overflow
        self.frame_id = self.frame_id + self.spp.batch_size if self.frame_id <= (2**31 - 1) else 0

        pred_rgb = torch.clamp(pred_rgb, 0.0, 1.0)  # Make sure image pixels are in valid range

        rendered_results["pred_rgb"] = pred_rgb
        return rendered_results

    @torch.cuda.nvtx.range("render_pass")
    @torch.no_grad()
    def render_pass(self, camera: Camera, is_first_pass: bool) -> Dict[str, torch.Tensor]:
        """Renders a single frame pass from the provided camera view, with optional progressive effects.
        This method is designed for interactive/real-time rendering scenarios where frame rate is prioritized
        over immediate full quality. It manages an internal state for progressive rendering effects.

        A rendering pass in this context is a single frame of the rendered image, where each pixel is rendered with
        a single ray sample.

        The rendering process happens in multiple passes when certain effects are enabled:
        1. First pass (is_first_pass=True): Renders base image without antialiasing or depth of field
        2. Subsequent passes (is_first_pass=False): Gradually adds quality improvements if enabled:
        - Antialiasing: Accumulates samples based on self.antialiasing_mode
        - Depth of Field: Accumulates samples based on self.depth_of_field settings

        By calling this function repeatedly, viewers can remain interactive while gradually improving quality
        as additional passes are rendered.
        Each repeated call returns the accumulated frame due to the most recent pass.
        This function returns the last cached frame if no more passes remain.

        Args:
            camera (Camera): Camera object defining the view to render from
            is_first_pass (bool): Whether this is the first pass for a new frame

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'rgb': Tensor of shape (1, H, W, 3) with values in [0, 1]
                - 'opacity': Tensor of shape (1, H, W, 1) with values in [0, 1]
                - 'rgb_buffer': Raw buffer before denoising, same shape as 'rgb'. Used for blending passes internally.

        Example:
            To enable progressive effects:
            ```
            engine = Engine3DGRUT(...)

            # Enable antialiasing
            engine.use_spp = True
            engine.antialiasing_mode = '8x MSAA'  # Options: '4x MSAA', '8x MSAA', '16x MSAA', 'Quasi-Random (Sobol)'

            # Enable depth of field
            engine.use_depth_of_field = True
            engine.depth_of_field.aperture_size = 0.01
            engine.depth_of_field.focus_z = 1.0

            # Render progressively
            rb = engine.render_pass(camera, is_first_pass=True)
            while engine.has_progressive_effects_to_render():
                rb = engine.render_pass(camera, is_first_pass=False)

            output_rgb = rb['rgb]
            ```
        """
        # Rendering 3DGRUT requires camera to run on cuda device -- avoid crashing
        if camera.device.type == "cpu":
            camera = camera.cuda()

        is_use_spp = not is_first_pass and not self.use_depth_of_field and self.use_spp
        rays = self.raygen(camera, use_spp=is_use_spp)

        if is_first_pass:
            # Disable path tracer under certain settings, useful for comparing with the vanilla volumetric tracer
            if not self.primitives.enabled or not self.primitives.has_visible_objects():
                if self.disable_gaussian_tracing:
                    rb = dict(
                        pred_rgb=torch.zeros_like(rays.rays_ori),
                        pred_opacity=torch.zeros_like(rays.rays_ori[:, :, 0:1]),
                    )
                else:
                    rb = self.scene_mog.trace(rays_o=rays.rays_ori, rays_d=rays.rays_dir)
            else:
                rb = self._render_playground_hybrid(rays.rays_ori, rays.rays_dir)

            rb = dict(rgb=rb["pred_rgb"], opacity=rb["pred_opacity"])
            rb["rgb"] = self.environment.tonemap(rb["rgb"])
            rb["rgb"] = torch.pow(rb["rgb"], 1.0 / self.gamma_correction)
            rb["rgb"] = rb["rgb"].mean(dim=0).unsqueeze(0)
            rb["opacity"] = rb["opacity"].mean(dim=0).unsqueeze(0)
            self.spp.reset_accumulation()
            self.depth_of_field.reset_accumulation()
        else:
            # Render accumulated effects, i.e. depth of field
            rb = dict(rgb=self.last_state["rgb_buffer"], opacity=self.last_state["opacity"])
            if self.use_depth_of_field:
                self._render_depth_of_field_buffer(rb, camera, rays)
            elif self.use_spp:
                self._render_spp_buffer(rb, rays)

        # Keep a noisy version of the accumulated rgb buffer so we don't repeat denoising per frame
        rb["rgb_buffer"] = rb["rgb"]
        if self.use_optix_denoiser:
            rb["rgb"] = self.tracer.denoise(rb["rgb"])

        if rays.mask is not None:  # mask is for masking away pixels out of view for, i.e. fisheye
            mask = rays.mask[None, :, :, 0]
            rb["rgb"][mask] = 0.0
            rb["rgb_buffer"][mask] = 0.0
            rb["opacity"][mask] = 0.0

        self._cache_last_state(camera=camera, renderbuffers=rb, canvas_size=[camera.height, camera.width])
        return rb

    @torch.cuda.nvtx.range("render")
    def render(self, camera: Camera) -> Dict[str, torch.Tensor]:
        """Renders a complete frame with all enabled visual effects.
        e.g. high-quality rendering method that ensures all progressive effects (antialiasing, depth of field)
        are fully computed.
        By default, 3dgrt requires only a single pass to render.
        Toggling on visual effects like antialiasing and depth of field may require additional samples which
        require additional passes. This method is best suited for offline rendering.

        Args:
            camera (Camera): Camera object defining the view to render

        Returns:
            Dict[str, torch.Tensor]: Rendering results containing:
                - 'rgb': Final image of shape (1, H, W, 3), range [0, 1]
                - 'opacity': Opacity buffer of shape (1, H, W, 1), range [0, 1]
                - 'rgb_buffer': Raw buffer before denoising, same shape as 'rgb'. Used for blending passes internally.

        Example:
            To enable progressive effects before rendering:
            ```
            engine.use_spp = True  # Enable antialiasing
            engine.use_depth_of_field = True  # Enable depth of field
            result = engine.render(camera)  # Will compute all passes
            ```
        """
        renderbuffers = self.render_pass(camera, is_first_pass=True)
        while self.has_progressive_effects_to_render():
            renderbuffers = self.render_pass(camera, is_first_pass=False)
        return renderbuffers

    def invalidate_materials_on_gpu(self) -> None:
        """Marks the materials on GPU as out of date.
        Materials and textures will be uploaded to the GPU with the next rendering pass.
        """
        self.is_materials_dirty = True

    @torch.cuda.nvtx.range("load_3dgrt_object")
    def load_3dgrt_object(
        self, object_path: str, config_name: str = "apps/colmap_3dgrt.yaml"
    ) -> Tuple[MixtureOfGaussians, str]:
        """Loads a pretrained 3D Gaussian model from various supported file formats.

        Supports loading from:
        - .pt: PyTorch checkpoint with model and config
        - .ingp: Instant-NGP format
        - .ply: Point cloud format

        For .ingp and .ply formats, initializes model using provided config.
        For .pt format, uses embedded config if valid, otherwise falls back to provided config.

        Args:
            object_path (str): Path to model file (.pt, .ingp, or .ply)
            config_name (str, optional): Path to hydra config file, relative to configs directory.
                Defaults to 'apps/colmap_3dgrt.yaml'.

        Returns:
            Tuple[MixtureOfGaussians, str]: Tuple containing:
                - Initialized 3D Gaussian model
                - Object name (from config or file stem)

        Raises:
            ValueError: If file extension is not supported

        Note:
            - Automatically builds acceleration structure after loading
            - Does not set up optimizer for checkpoint loading
            - Falls back to filename stem if no object name specified
        """

        def load_default_config():
            from hydra.compose import compose
            from hydra.initialize import initialize

            with initialize(version_base=None, config_path="../configs"):
                conf = compose(config_name=config_name)
            return conf

        if object_path.endswith(".pt"):
            checkpoint = torch.load(object_path, weights_only=False)
            conf = checkpoint["config"]
            if conf.render["method"] != "3dgrt":
                conf = load_default_config()
            model = MixtureOfGaussians(conf)
            model.init_from_checkpoint(checkpoint, setup_optimizer=False)
            object_name = conf.experiment_name
        elif object_path.endswith(".ingp"):
            conf = load_default_config()
            model = MixtureOfGaussians(conf)
            model.init_from_ingp(object_path, init_model=False)
            object_name = Path(object_path).stem
        elif object_path.endswith(".ply"):
            conf = load_default_config()
            model = MixtureOfGaussians(conf)
            model.init_from_ply(object_path, init_model=False)
            object_name = Path(object_path).stem
        else:
            raise ValueError(f"Unknown object type: {object_path}")

        if object_name is None or len(object_name) == 0:
            object_name = Path(object_path).stem  # Fallback to pick object name from path, if none specified

        model.build_acc(rebuild=True)

        return model, object_name

    @torch.cuda.nvtx.range("rebuild_bvh (mog)")
    def rebuild_bvh(self, scene_mog: MixtureOfGaussians) -> None:
        """Rebuilds all Bounding Volume Hierarchies (BVHs) used by the 3DGRUT engine.

        The engine calls this function internally.
        Users should only call this function directly in case they perform further modifications to the scene.

        This method rebuilds two types of BVH structures:
        1. 3dgrt BVH of proxy shapes encapsulating gaussian particles
        2. Mesh BVH holding faces of primitives used in the playground

        BVH should be rebuilt when:
        - The scene is first loaded
        - The scene geometry changes (gaussians moved/added/removed)
        - Primitives are added, removed, or transformed

        Args:
            scene_mog (MixtureOfGaussians): The 3D Gaussian model whose BVH needs to be rebuilt
        """
        rebuild = True
        self.tracer.build_gs_acc(gaussians=scene_mog, rebuild=rebuild)
        self.primitives.rebuild_bvh_if_needed()

    def did_camera_change(self, camera: Camera) -> bool:
        """Checks if the camera view matrix has changed from the last cached state.

        Compares the current camera's view matrix against the cached camera matrix
        from the last render. Used to determine if a new render is needed.

        Args:
            camera (Camera): Current camera to check against cached state

        Returns:
            bool: True if camera has moved or rotated since last render,
                  False if camera is unchanged or no previous state exists
        """
        current_view_matrix = camera.view_matrix()
        cached_camera_matrix = self.last_state.get("camera")
        is_camera_changed = cached_camera_matrix is not None and (cached_camera_matrix != current_view_matrix).any()
        return is_camera_changed

    def has_cached_buffers(self) -> bool:
        """Checks if cached buffers exist for the last rendered frame.
        Normally, buffers of the last rendered frame are cached in the engine state at the end of the render pass.

        Returns:
            bool: True if cached buffers exist, False otherwise
        """
        return self.last_state.get("rgb") is not None and self.last_state.get("opacity") is not None

    def has_progressive_effects_to_render(self) -> bool:
        """Checks if additional progressive rendering passes are needed.

        Determines whether enabled progressive effects (i.e. antialiasing, depth of field)
        require more samples to complete. Used to control progressive rendering loops.

        Returns:
            bool: True if either:
                  - Depth of field is enabled and needs more samples
                  - Antialiasing is enabled and needs more samples
                  False if all enabled effects are fully sampled
        """
        has_dof_buffers_to_render = (
            self.use_depth_of_field and self.depth_of_field.spp_accumulated_for_frame <= self.depth_of_field.spp
        )
        has_spp_buffers_to_render = (
            not self.use_depth_of_field and self.use_spp and self.spp.spp_accumulated_for_frame <= self.spp.spp
        )
        return has_dof_buffers_to_render or has_spp_buffers_to_render

    def is_dirty(self, camera: Camera) -> bool:
        """Returns whether the scene state has changed since the last canvas render:
        - Camera has moved
        - Materials have been modified
        - No cached buffers exist (e.g. this is the first frame being rendered)

        This method is not used internally by the engine, but is intended to be used by viewers
        to determine if they need to re-render the scene.

        Args:
            camera (Camera): Current camera to compare against cached state

        Returns:
            bool: True if scene needs re-rendering, False if cached framebuffers can be used

        Example:
            # Viewer code
            def on_frame_render(engine, camera):
                is_first_pass = engine.is_dirty(camera)
                if not is_first_pass and not engine.has_progressive_effects_to_render():
                    return engine.last_state['rgb'], engine.last_state['opacity']

                buffers = engine.render_pass(camera, is_first_pass)
                return buffers['rgb'], buffers['opacity']
        """
        # Force dirty flag is on
        if self.is_materials_dirty:
            return True
        if self.did_camera_change(camera):
            return True
        if not self.has_cached_buffers():
            return True
        return False

    def _cache_last_state(self, camera: Camera, renderbuffers: Dict[str, torch.Tensor], canvas_size: Tuple[int, int]):
        """Caches the last rendered state for comparison during future renders.

        Args:
            camera (Camera): Current camera view matrix
            renderbuffers (dict): Mapping of buffer name -> buffers of recently rendered frame
            canvas_size (tuple): Canvas size (width, height)
        """
        self.last_state["canvas_size"] = canvas_size
        self.last_state["camera"] = copy.deepcopy(camera.view_matrix())
        self.last_state["rgb"] = renderbuffers["rgb"]
        self.last_state["rgb_buffer"] = renderbuffers["rgb_buffer"]
        self.last_state["opacity"] = renderbuffers["opacity"]

    def _raygen_pinhole(self, camera: Camera, jitter: Optional[torch.Tensor] = None) -> RayPack:
        """Generates ray origins and directions for pinhole camera model.

        Creates rays for each pixel in the image plane using the pinhole camera model.
        Optionally applies jitter for antialiasing.

        Args:
            camera (Camera): Camera parameters including intrinsics and pose
            jitter (Optional[torch.Tensor]): Per-pixel offset of shape (H, W, 2) in range [-0.5, 0.5].
                Used for antialiasing. Defaults to None.

        Returns:
            RayPack: Contains:
                - rays_ori: Ray origins of shape (1, H, W, 3)
                - rays_dir: Ray directions of shape (1, H, W, 3)
                - pixel_x/y: Integer pixel coordinates
        """
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
            pixel_y=torch.round(pixel_y - 0.5).squeeze(-1),
        )

    @torch.cuda.nvtx.range("_raygen_fisheye")
    def _raygen_fisheye(self, camera: Camera, jitter: Optional[torch.Tensor] = None) -> RayPack:
        """Generates ray origins and directions for a perfect fisheye camera model.

        Creates rays for each pixel using the fisheye camera model. Includes
        mask for valid rays since not all pixels map to valid directions in
        equirectangular fisheye projection.

        Args:
            camera (Camera): Camera parameters including intrinsics and pose
            jitter (Optional[torch.Tensor]): Per-pixel offset of shape (H, W, 2).
                Used for antialiasing. Defaults to None.

        Returns:
            RayPack: Contains:
                - rays_ori: Ray origins of shape (1, H, W, 3)
                - rays_dir: Ray directions of shape (1, H, W, 3)
                - pixel_x/y: Integer pixel coordinates
                - mask: Boolean mask of valid rays (H, W, 1)
        """
        pixel_y, pixel_x = generate_centered_pixel_coords(camera.width, camera.height, device=camera.device)
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
            mask=mask.reshape(camera.height, camera.width, 1),
        )

    def raygen(self, camera: Camera, use_spp: bool = False) -> RayPack:
        """Generates rays for rendering based on current camera type.

        Creates ray batch for supported camera models.
        Handles antialiasing by generating multiple jittered ray samples
        when SPP (Samples Per Pixel) is enabled.

        Args:
            camera (Camera): Camera parameters including intrinsics and pose
            use_spp (bool): Whether to generate multiple samples per pixel
                for antialiasing. Defaults to False.

        Returns:
            RayPack: Contains batched rays, where batch size is either 1 or
                self.spp.batch_size if use_spp is True.
        """
        ray_batch_size = 1 if not use_spp else self.spp.batch_size
        rays = []
        for _ in range(ray_batch_size):
            jitter = self.spp(camera.height, camera.width) if use_spp and self.spp is not None else None
            if self.camera_type == "Pinhole":
                next_rays = self._raygen_pinhole(camera, jitter)
            elif self.camera_type == "Fisheye":
                next_rays = self._raygen_fisheye(camera, jitter)
            else:
                raise ValueError(f"Unknown camera type: {self.camera_type}")
            rays.append(next_rays)
        return RayPack(
            mask=rays[0].mask,
            pixel_x=rays[0].pixel_x,
            pixel_y=rays[0].pixel_y,
            rays_ori=torch.cat([r.rays_ori for r in rays], dim=0),
            rays_dir=torch.cat([r.rays_dir for r in rays], dim=0),
        )
