#!/usr/bin/env python3
# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
Script to add a mesh (PLY or USD) into an existing USDZ file.

The mesh reference and proxy relationship are added in the volume/gauss USD
layer so that the composition matches the expected NuRec USDZ layout:

  volume.usda / gauss.usda:
    /World
      /World/volume (or /World/gauss)   ← Volume prim, proxy -> </World/mesh>
      /World/mesh                        ← over, references @mesh.usd@

  default.usda (root layer):
    /World (Xform, defaultPrim)
      over "volume" (references = @volume.usda@)   ← or gauss.usda

  mesh.usd:
    defaultPrim = "mesh"
    /mesh (UsdGeom.Mesh)

After composition the Volume's proxy rel (</World/mesh>) gets remapped
through the reference so it correctly resolves to the mesh geometry.
"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt


def _ply_to_usd_mesh(ply_path: Path, usd_path: Path) -> None:
    """
    Convert a triangle mesh PLY file to a USD file containing a single UsdGeom.Mesh.

    The prim is named "/mesh" with defaultPrim = "mesh" to match the
    convention used in NuRec USDZ packages (proxy = </World/mesh>).
    """
    from plyfile import PlyData

    ply_data = PlyData.read(str(ply_path))
    vertex_data = ply_data["vertex"]
    vertices = np.column_stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]]
    ).astype(np.float32)

    face_data = ply_data["face"]
    triangles = np.vstack(face_data["vertex_indices"]).astype(np.int32)
    if triangles.shape[1] != 3:
        raise ValueError(f"PLY mesh faces are not triangles (got {triangles.shape[1]} vertices per face): {ply_path}")
    if len(triangles) == 0:
        raise ValueError(f"PLY mesh has no triangles: {ply_path}")

    stage = Usd.Stage.CreateInMemory()
    mesh_prim = UsdGeom.Mesh.Define(stage, "/mesh")
    mesh_prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh_prim.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32))
    )
    mesh_prim.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(triangles.ravel())
    )
    stage.GetRootLayer().defaultPrim = "mesh"
    stage.GetRootLayer().Export(str(usd_path))
    print(f"  Converted PLY -> USD: {usd_path.name} ({len(vertices)} verts, {len(triangles)} faces)")


def _find_volume_prim(stage):
    """Find the first Volume prim under /World in the stage."""
    world = stage.GetPrimAtPath("/World")
    if not world:
        return None
    for prim in Usd.PrimRange(world):
        if prim.GetTypeName() == "Volume":
            return prim
    return None


def add_mesh_to_usdz(
    input_usdz,
    output_usdz,
    mesh_usd_path=None,
    mesh_ply_path=None,
    referencing_usd=None,
    set_collision=True,
    set_invisible=False,
):
    """
    Add the specified mesh into the USDZ.

    Either mesh_usd_path or mesh_ply_path must be set.

    - mesh_usd_path: copy the USD file as mesh.usd into the package.
    - mesh_ply_path: copy PLY as mesh.ply, convert to mesh.usd, put both in package.

    The script modifies the volume/gauss USD layer to:
    1. Add ``over "mesh" (references = @mesh.usd@)`` under /World
    2. Set the Volume prim's ``proxy`` relationship to </World/mesh>

    Args:
        input_usdz: Path to input USDZ file.
        output_usdz: Path to output USDZ file.
        mesh_usd_path: Optional path to a mesh USD file to add.
        mesh_ply_path: Optional path to a mesh PLY file; converted to USD then added.
        referencing_usd: Which USD file in the package to modify. If None, auto-detected
                         as the file containing a Volume prim (gauss.usda / volume.usda).
        set_collision: If True, enable collision on mesh prims in mesh.usd.
        set_invisible: If True, make mesh prims in mesh.usd invisible.
    """
    input_path = Path(input_usdz)
    if not input_path.exists():
        raise FileNotFoundError(f"Input USDZ file not found: {input_usdz}")

    if mesh_ply_path is not None and mesh_usd_path is not None:
        raise ValueError("Provide only one of mesh_ply_path or mesh_usd_path.")
    if mesh_ply_path is None and mesh_usd_path is None:
        raise ValueError("Provide either mesh_ply_path or mesh_usd_path.")

    use_ply = mesh_ply_path is not None
    if use_ply:
        mesh_ply_path = Path(mesh_ply_path)
        if not mesh_ply_path.exists():
            raise FileNotFoundError(f"Mesh PLY file not found: {mesh_ply_path}")
    else:
        mesh_usd_path = Path(mesh_usd_path)
        if not mesh_usd_path.exists():
            raise FileNotFoundError(f"Mesh USD file not found: {mesh_usd_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Using temporary directory: {temp_path}")

        # Step 1: Unzip the USDZ file
        print("Step 1: Unzipping USDZ file...")
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(temp_path)

        print("Files found in USDZ:")
        for f in temp_path.rglob("*"):
            if f.is_file():
                print(f"  - {f.relative_to(temp_path)}")

        # Step 2: Add mesh files
        print("Step 2: Adding mesh into package...")
        if use_ply:
            dest_ply = temp_path / "mesh.ply"
            dest_usd = temp_path / "mesh.usd"
            shutil.copy2(mesh_ply_path, dest_ply)
            print("  Copied mesh PLY to: mesh.ply")
            _ply_to_usd_mesh(mesh_ply_path, dest_usd)
        else:
            dest_usd = temp_path / "mesh.usd"
            shutil.copy2(mesh_usd_path, dest_usd)
            print("  Copied mesh USD to: mesh.usd")

        # Step 3: Find the volume/gauss USD and add mesh reference + proxy
        if referencing_usd:
            ref_usd_path = temp_path / referencing_usd
            if not ref_usd_path.exists():
                raise RuntimeError(
                    f"'{referencing_usd}' not found in the USDZ package. "
                    f"Available: {[str(f.relative_to(temp_path)) for f in temp_path.rglob('*') if f.is_file()]}"
                )
        else:
            ref_usd_path = None
            for f in temp_path.iterdir():
                if f.suffix.lower() in (".usd", ".usda") and f.name != "default.usda":
                    stage_candidate = Usd.Stage.Open(str(f))
                    if stage_candidate and _find_volume_prim(stage_candidate):
                        ref_usd_path = f
                        break
            if not ref_usd_path:
                raise RuntimeError(
                    "Could not find a USD file with a Volume prim. "
                    "Use --referencing_usd to specify it explicitly."
                )

        referencing_usd = ref_usd_path.name
        print(f"Step 3: Modifying {referencing_usd} ...")
 
        stage = Usd.Stage.Open(str(ref_usd_path))
        if not stage:
            raise RuntimeError(f"Failed to open USD file: {ref_usd_path}")

        # 3a: Add  over "mesh" (references = @mesh.usd@)  under /World
        mesh_prim_path = "/World/mesh"
        prim = stage.GetPrimAtPath(mesh_prim_path)
        if not prim:
            prim = stage.OverridePrim(mesh_prim_path)
        prim.GetReferences().AddReference(assetPath="mesh.usd")
        print(f"  Added {mesh_prim_path} with reference to mesh.usd")

        # 3b: Set proxy relationship on the Volume prim -> </World/mesh>
        volume_prim = _find_volume_prim(stage)
        if volume_prim:
            proxy_target = Sdf.Path("/World/mesh")
            proxy_rel = volume_prim.GetRelationship("proxy")
            if not proxy_rel:
                proxy_rel = volume_prim.CreateRelationship("proxy", custom=True)
            proxy_rel.SetTargets([proxy_target])
            print(f"  Set {volume_prim.GetPath()}.proxy -> {proxy_target}")
        else:
            print("  Warning: No Volume prim found, skipping proxy relationship")

        stage.Save()
        print(f"  Saved: {referencing_usd}")

        # 3c: Set Mesh properties on USD file
        if set_collision or set_invisible:
            print("Step 3c: Setting mesh properties on mesh.usd ...")
            mesh_stage = Usd.Stage.Open(str(dest_usd))
            if not mesh_stage:
                raise RuntimeError(f"Failed to open mesh USD file: {dest_usd}")

            mesh_prims = [p for p in mesh_stage.Traverse() if p.IsA(UsdGeom.Mesh)]
            if not mesh_prims:
                print("  Warning: No mesh prims found in mesh.usd")
            else:
                print(f"  Found {len(mesh_prims)} mesh prim(s)")
                for mp in mesh_prims:
                    if set_collision:
                        UsdPhysics.CollisionAPI.Apply(mp).CreateCollisionEnabledAttr(True)
                        UsdPhysics.MeshCollisionAPI.Apply(mp).CreateApproximationAttr().Set("none")
                        print(f"  Enabled collision on {mp.GetPath()}")
                    if set_invisible:
                        UsdGeom.Imageable(mp).MakeInvisible()
                        print(f"  Made {mp.GetPath()} invisible")
                mesh_stage.Save()
                print(f"  Saved mesh properties to: {dest_usd.name}")

        # Step 4: Create new USDZ (preserve original file order, append new files)
        print("Step 4: Creating new USDZ file...")
        output_path = Path(output_usdz)
        if output_path.exists():
            output_path.unlink()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(input_path, "r") as orig_zip:
            file_order = orig_zip.namelist()

        current_files = {
            str(f.relative_to(temp_path))
            for f in temp_path.rglob("*")
            if f.is_file()
        }
        new_files = current_files - set(file_order)
        if new_files:
            print(f"New files added: {sorted(new_files)}")
            file_order.extend(sorted(new_files))

        print(f"Write order: {file_order}")

        with zipfile.ZipFile(
            output_path, "w", compression=zipfile.ZIP_STORED
        ) as zip_file:
            for filename in file_order:
                file_path = temp_path / filename
                if file_path.exists():
                    print(f"  Adding: {filename}")
                    with open(file_path, "rb") as f:
                        zip_file.writestr(filename, f.read())
                else:
                    print(f"  Warning: {filename} not found in temp directory")

        print(f"Successfully created: {output_path}")
        print(f"File size: {output_path.stat().st_size} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Add a mesh into a USDZ file (input: mesh USD or mesh PLY)"
    )
    parser.add_argument("--input_usdz", required=True, help="Input USDZ file path")
    parser.add_argument("--output_usdz", required=True, help="Output USDZ file path")
    mesh_group = parser.add_mutually_exclusive_group(required=True)
    mesh_group.add_argument(
        "--mesh_usd",
        help="Path to a mesh USD file to add as mesh.usd",
    )
    mesh_group.add_argument(
        "--mesh_ply",
        help="Path to a mesh PLY file; converts to mesh.usd and adds both mesh.ply and mesh.usd",
    )
    parser.add_argument(
        "--referencing_usd",
        default=None,
        help="Which USD file in the package to modify (default: auto-detect the one with a Volume prim)",
    )
    parser.add_argument(
        "--set_collision", action="store_true", help="Enable collision on mesh prims"
    )
    parser.add_argument(
        "--set_invisible", action="store_true", help="Make mesh prims invisible"
    )

    args = parser.parse_args()

    try:
        add_mesh_to_usdz(
            args.input_usdz,
            args.output_usdz,
            mesh_usd_path=args.mesh_usd,
            mesh_ply_path=args.mesh_ply,
            referencing_usd=args.referencing_usd,
            set_collision=args.set_collision,
            set_invisible=args.set_invisible,
        )
        print("USDZ processing completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
