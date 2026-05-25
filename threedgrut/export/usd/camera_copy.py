# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Copy prims from a source USD stage into another USD export stage."""

import logging
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Collection, Iterator, List, Optional, Set, Tuple

from pxr import Sdf, Usd, UsdGeom, UsdUtils

from threedgrut.export.usd.stage_utils import NamedSerialized

logger = logging.getLogger(__name__)

UsdStagePathPair = Tuple[Path, Path]

# Default: do not duplicate source Gaussian roots (large); the transcode target
# writes a fresh Gaussian payload in its own flavor.
_DEFAULT_SKIP_SUBTREES = (
    Sdf.Path("/World/Gaussians"),
    Sdf.Path("/World/gaussians"),
    Sdf.Path("/World/gauss"),
)


def _path_is_under_skipped(src_path: Sdf.Path, skip_roots: Collection[Sdf.Path]) -> bool:
    for root in skip_roots:
        if src_path == root:
            return True
        # Children of root (e.g. /World/Gaussians/gaussians)
        prefix = str(root) + "/"
        if str(src_path).startswith(prefix):
            return True
    return False


def _copy_prim_spec_recursive(
    src_layer: Sdf.Layer,
    dst_layer: Sdf.Layer,
    src_path: Sdf.Path,
    dst_path: Sdf.Path,
) -> int:
    """Copy one prim spec and all descendants. Returns number of prims copied."""
    src_spec = src_layer.GetPrimAtPath(src_path)
    if not src_spec or not src_spec.active:
        return 0
    Sdf.CopySpec(src_layer, src_path, dst_layer, dst_path)
    count = 1
    for child_spec in src_spec.nameChildren:
        name = child_spec.name
        count += _copy_prim_spec_recursive(
            src_layer,
            dst_layer,
            src_path.AppendChild(name),
            dst_path.AppendChild(name),
        )
    return count


def merge_source_world_at_same_paths(
    dest_stage,
    source_stage,
    skip_source_subtrees: Optional[Collection[Sdf.Path]] = None,
) -> int:
    """
    Merge each top-level child of ``/World`` from the source **root** layer onto ``dest_stage``'s
    root layer at the **same path** as the source (e.g. ``/World/rig_trajectories``), using
    ``Sdf.CopySpec``. References and payloads are copied as-authored so sibling layers (e.g.
    ``rig_trajectories.usda``) keep all time samples when those files are bundled unchanged.

    Skips subtrees in ``skip_source_subtrees`` (default: ``/World/Gaussians`` for LightField).
    Skips any path where the destination root layer already has a prim spec (e.g. export's
    ``/World/gaussians`` reference prim).
    """
    skips = tuple(skip_source_subtrees) if skip_source_subtrees is not None else _DEFAULT_SKIP_SUBTREES
    src_layer = source_stage.GetRootLayer()
    dst_layer = dest_stage.GetRootLayer()

    world_spec = src_layer.GetPrimAtPath("/World")
    if not world_spec:
        logger.info("Source USD has no /World prim; nothing to merge")
        return 0

    total = 0
    for child_spec in world_spec.nameChildren:
        name = child_spec.name
        path = Sdf.Path("/World").AppendChild(name)
        if _path_is_under_skipped(path, skips):
            logger.info("Skipping source subtree %s (transcode merge skip list)", path)
            continue
        if dst_layer.GetPrimAtPath(path):
            logger.info("Keeping destination prim %s; not overwriting with source", path)
            continue
        total += _copy_prim_spec_recursive(src_layer, dst_layer, path, path)

    if total == 0:
        logger.info("No source /World prims merged (empty or all skipped / already present)")
    else:
        logger.info("Merged %d source prim subtree(s) at original /World paths", total)

    return total


def merge_source_prim_at_same_path(dest_stage, source_stage, prim_path: str) -> int:
    """
    Copy one source root-layer prim subtree to the destination at the same path.

    This preserves non-geometry export data, such as `/Render`, during USD to
    USD transcode without regenerating renderer state from Python objects.
    """
    src_layer = source_stage.GetRootLayer()
    dst_layer = dest_stage.GetRootLayer()
    path = Sdf.Path(prim_path)

    if not src_layer.GetPrimAtPath(path):
        logger.info("Source USD has no %s prim; nothing to merge", prim_path)
        return 0
    if dst_layer.GetPrimAtPath(path):
        logger.info("Keeping destination prim %s; not overwriting with source", prim_path)
        return 0

    count = _copy_prim_spec_recursive(src_layer, dst_layer, path, path)
    logger.info("Merged source %s subtree with %d prim(s)", prim_path, count)
    return count


def copy_authored_time_settings_from_source(source_stage, dest_stage) -> None:
    """Copy authored time code range and FPS from source to destination stage when set."""
    try:
        source_start = source_stage.GetStartTimeCode()
        source_end = source_stage.GetEndTimeCode()
        if source_start != dest_stage.GetStartTimeCode() or source_end != dest_stage.GetEndTimeCode():
            dest_stage.SetStartTimeCode(source_start)
            dest_stage.SetEndTimeCode(source_end)
        tps = source_stage.GetTimeCodesPerSecond()
        if tps is not None and float(tps) > 0.0:
            dest_stage.SetTimeCodesPerSecond(tps)
    except Exception as ex:
        logger.debug("Could not copy time settings from source stage: %s", ex)


# Filenames commonly authored by USD transcode targets (never pull from the
# source package when collecting copied-source sidecars).
_OUTPUT_AUTHORED_NAMES = frozenset({"gaussians.usdc", "gauss.usda", "default.usda"})


def _basename_packaged_ref(asset_path: str) -> Optional[str]:
    """USDZ-flat basename for a relative layer/asset reference, or None if not packagable."""
    if not asset_path:
        return None
    s = asset_path.strip().strip("@")
    if not s or "://" in s or s.startswith("/"):
        return None
    return Path(s.replace("\\", "/")).name


def _gather_ref_payload_basenames_from_prim_spec(spec: Sdf.PrimSpec) -> Set[str]:
    out: Set[str] = set()
    if not spec:
        return out
    ref_list = spec.referenceList
    for item in list(ref_list.prependedItems) + list(ref_list.appendedItems):
        bn = _basename_packaged_ref(getattr(item, "assetPath", "") or "")
        if bn:
            out.add(bn)
    pay_list = getattr(spec, "payloadList", None)
    if pay_list is not None:
        for item in list(pay_list.prependedItems) + list(pay_list.appendedItems):
            bn = _basename_packaged_ref(getattr(item, "assetPath", "") or "")
            if bn:
                out.add(bn)
    for prop in spec.properties:
        default_value = getattr(prop, "default", None)
        asset_path = getattr(default_value, "path", None) or getattr(
            default_value,
            "assetPath",
            None,
        )
        if asset_path:
            bn = _basename_packaged_ref(asset_path)
            if bn:
                out.add(bn)
    return out


def _companion_sidecar_basenames(basename: str) -> Set[str]:
    """Additional package files implied by a referenced asset."""
    if basename.endswith(".cu"):
        return {f"{basename}.lua"}
    if basename.endswith(".slang"):
        return {f"{basename}.lua"}
    return set()


def append_unique_serialized_file(files: List[NamedSerialized], entry: NamedSerialized) -> None:
    """Append ``entry`` unless a file with the same package basename is already queued."""
    if not any(existing.filename == entry.filename for existing in files):
        files.append(entry)


def save_serialized_files(files: Collection[NamedSerialized], output_dir: Path) -> None:
    """Write serialized sidecars next to a loose USD output."""
    for entry in files:
        entry.save(output_dir)


def merge_source_prims_and_collect_sidecars(
    *,
    dest_stage,
    source_stage,
    res_root: Optional[Path],
    source_stage_path: Path,
    files: List[NamedSerialized],
    skip_source_subtrees: Optional[Collection[Sdf.Path]] = None,
    path_prefixes: Collection[str] = ("/World", "/Render"),
) -> None:
    """Merge source non-Gaussian prims and queue referenced sidecars.

    The merge preserves source ``/World`` support prims (cameras, rigs, etc.)
    and source ``/Render`` prims while avoiding duplicate Gaussian payloads.
    Any relative assets referenced under the merged prefixes are queued in
    ``files`` so packaged USDZ exports can include them and loose USD exports
    can write them next to the target layer.
    """
    # The destination stage is usually anonymous while we are still collecting
    # sidecars, so relative references cannot resolve until packaging finishes.
    # Capture those transient diagnostics; missing files are reported by the
    # explicit sidecar collector below.
    delegate = UsdUtils.CoalescingDiagnosticDelegate()
    merge_source_world_at_same_paths(dest_stage, source_stage, skip_source_subtrees=skip_source_subtrees)
    merge_source_prim_at_same_path(dest_stage, source_stage, "/Render")
    copy_authored_time_settings_from_source(source_stage, dest_stage)
    del delegate

    if res_root is None or not res_root.is_dir():
        return

    for path_prefix in path_prefixes:
        sidecars = collect_transitive_sidecars_for_subtree(
            dest_stage.GetRootLayer(),
            res_root,
            path_prefix=path_prefix,
            extra_skip_names={source_stage_path.name},
        )
        for entry in sidecars:
            append_unique_serialized_file(files, entry)


def stage_has_ppisp_post_processing_effects(stage, render_scope_path: str = "/Render") -> bool:
    """Return True if a stage's render scope contains PPISP SPG shader effects."""
    render_scope = stage.GetPrimAtPath(render_scope_path)
    if not render_scope.IsValid():
        return False

    for prim in Usd.PrimRange(render_scope):
        if prim.GetTypeName() != "Shader":
            continue
        prim_text = str(prim.GetPath()).lower()
        if "ppisp" in prim_text:
            return True
        source_asset = prim.GetAttribute("info:spg:sourceAsset")
        if source_asset.IsValid():
            asset = source_asset.Get()
            asset_path = getattr(asset, "path", asset)
            if asset_path is not None and "ppisp" in str(asset_path).lower():
                return True
        implementation_source = prim.GetAttribute("info:implementationSource")
        if implementation_source.IsValid() and implementation_source.Get() == "sourceAsset" and "ppisp" in prim_text:
            return True
    return False


def _walk_prim_subtree(layer: Sdf.Layer, root_path: Sdf.Path):
    """Depth-first active prims under root_path (inclusive)."""
    spec = layer.GetPrimAtPath(root_path)
    if not spec or not spec.active:
        return
    yield root_path
    for child_spec in spec.nameChildren:
        yield from _walk_prim_subtree(layer, root_path.AppendChild(child_spec.name))


def _gather_refs_from_layer_subtree(layer: Sdf.Layer, path_prefix: str) -> Set[str]:
    """Collect referenced basenames from all prims under path_prefix on this layer."""
    needed: Set[str] = set()
    root = Sdf.Path(path_prefix)
    if not layer.GetPrimAtPath(root):
        return needed
    for path in _walk_prim_subtree(layer, root):
        spec = layer.GetPrimAtPath(path)
        needed |= _gather_ref_payload_basenames_from_prim_spec(spec)
    return needed


def _walk_entire_layer(layer: Sdf.Layer):
    """All active prim paths (excluding absolute root pseudo-prim)."""
    root = Sdf.Path("/")
    spec = layer.GetPrimAtPath(root)
    if not spec:
        return
    for child_spec in spec.nameChildren:
        yield from _walk_prim_subtree(layer, root.AppendChild(child_spec.name))


def collect_transitive_sidecars_for_subtree(
    dest_layer: Sdf.Layer,
    res_root: Path,
    path_prefix: str,
    extra_skip_names: Optional[Collection[str]] = None,
) -> List[NamedSerialized]:
    """
    Resolve layer/asset references under ``path_prefix`` and bundle files from
    ``res_root`` into the output USDZ (flat layout).

    Follows references/payloads transitively through USD layers. Skips names in
    ``_OUTPUT_AUTHORED_NAMES`` and ``extra_skip_names`` (e.g. source root default
    file).
    """
    skip: Set[str] = set(_OUTPUT_AUTHORED_NAMES)
    if extra_skip_names:
        skip.update(extra_skip_names)

    seed = _gather_refs_from_layer_subtree(dest_layer, path_prefix)
    queue: Set[str] = {n for n in seed if n not in skip}
    done: Set[str] = set(skip)
    result: List[NamedSerialized] = []

    while queue:
        name = queue.pop()
        if name in done:
            continue
        done.add(name)
        path = res_root / name
        if not path.is_file():
            logger.warning("Referenced package file missing under %s: %s", res_root, name)
            continue
        try:
            data = path.read_bytes()
        except OSError as e:
            logger.warning("Could not read sidecar %s: %s", path, e)
            continue
        result.append(NamedSerialized(filename=name, serialized=data))
        for companion in _companion_sidecar_basenames(name):
            if companion not in done:
                queue.add(companion)

        suf = path.suffix.lower()
        if suf not in (".usd", ".usda", ".usdc"):
            continue
        sub = Sdf.Layer.FindOrOpen(str(path))
        if not sub:
            logger.warning("Could not open referenced layer for sidecar walk: %s", path)
            continue
        for p in _walk_entire_layer(sub):
            spec = sub.GetPrimAtPath(p)
            for bn in _gather_ref_payload_basenames_from_prim_spec(spec):
                if bn and bn not in done:
                    queue.add(bn)

    if result:
        logger.info(
            "Bundled %d sidecar file(s) from %s for %s references",
            len(result),
            res_root,
            path_prefix,
        )
    return result


def collect_transitive_sidecars_for_world_subtree(
    dest_layer: Sdf.Layer,
    res_root: Path,
    world_prefix: str = "/World",
    extra_skip_names: Optional[Collection[str]] = None,
) -> List[NamedSerialized]:
    return collect_transitive_sidecars_for_subtree(
        dest_layer,
        res_root,
        path_prefix=world_prefix,
        extra_skip_names=extra_skip_names,
    )


@contextmanager
def usd_stage_path_context_for_camera_copy(usd_path: Path) -> Iterator[Optional[UsdStagePathPair]]:
    """
    Yield (root_stage_path, asset_resolution_dir) for opening a USD/USDZ with correct asset paths.

    For USDZ, extracts to a temporary directory (deleted on exit).
    """
    path = usd_path.resolve()
    suffix = path.suffix.lower()
    if suffix not in (".usd", ".usda", ".usdc", ".usdz"):
        yield None
        return

    if suffix == ".usdz":
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmp_path)
            usd_files = list(tmp_path.glob("*.usd*"))
            root_file = None
            for f in usd_files:
                if f.stem == "default":
                    root_file = f
                    break
            if root_file is None and usd_files:
                root_file = usd_files[0]
            if root_file is None:
                logger.warning("USDZ has no USD root for source prim copy: %s", path)
                yield None
                return
            yield (root_file.resolve(), tmp_path.resolve())
        return

    yield (path, path.parent.resolve())
