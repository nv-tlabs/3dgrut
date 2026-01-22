"""
An example for running incremental SfM on 360 spherical panorama images.
Modified to process omni images and generate 16 virtual camera views.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import PIL.ExifTags
import PIL.Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pycolmap
from pycolmap import logging


def create_virtual_camera(
    pano_height: int, fov_deg: float = 90
) -> pycolmap.Camera:
    """Create a virtual perspective camera."""
    image_size = int(pano_height * fov_deg / 180)
    focal = image_size / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    return pycolmap.Camera.create(0, "PINHOLE", focal, image_size, image_size)


def get_virtual_camera_rays(camera: pycolmap.Camera) -> np.ndarray:
    size = (camera.width, camera.height)
    y, x = np.indices(size).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()])
    # The center of the upper left most pixel has coordinate (0.5, 0.5)
    xy += 0.5
    xy_norm = camera.cam_from_img(xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(image_size, rays_in_cam: np.ndarray) -> np.ndarray:
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size


def get_virtual_rotations(
    num_steps_yaw: int = 8, pitches_deg: Sequence[float] = (-35.0, 35.0)
) -> Sequence[np.ndarray]:
    """Get the relative rotations of the virtual cameras w.r.t. the panorama.
    Modified to generate 16 cameras (8 yaw x 2 pitch).
    """
    # Assuming that the panos are approximately upright.
    cams_from_pano_r = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            cam_from_pano_r = Rotation.from_euler(
                "XY", [-pitch_deg, -yaw_deg], degrees=True
            ).as_matrix()
            cams_from_pano_r.append(cam_from_pano_r)
    return cams_from_pano_r


def create_pano_rig_config(
    cams_from_pano_rotation: Sequence[np.ndarray], ref_idx: int = 0
) -> pycolmap.RigConfig:
    """Create a RigConfig for the given virtual rotations."""
    rig_cameras = []
    for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = (
                cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T
            )
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation), np.zeros(3)
            )
        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=idx == ref_idx,
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


def render_perspective_images(
    pano_image_names: Sequence[str],
    pano_image_dir: Path,
    output_image_dir: Path,
    mask_dir: Path,
) -> pycolmap.RigConfig:
    cams_from_pano_rotation = get_virtual_rotations()
    rig_config = create_pano_rig_config(cams_from_pano_rotation)

    # We assign each pano pixel to the virtual camera with the closest center.
    cam_centers_in_pano = np.einsum(
        "nij,i->nj", cams_from_pano_rotation, [0, 0, 1]
    )

    camera = pano_size = rays_in_cam = None
    for pano_name in tqdm(pano_image_names):
        pano_path = pano_image_dir / pano_name
        try:
            pano_image = PIL.Image.open(pano_path)
        except PIL.Image.UnidentifiedImageError:
            logging.info(f"Skipping file {pano_path} as it cannot be read.")
            continue
        pano_exif = pano_image.getexif()
        pano_image = np.asarray(pano_image)
        gpsonly_exif = PIL.Image.Exif()
        gpsonly_exif[PIL.ExifTags.IFD.GPSInfo] = pano_exif.get_ifd(
            PIL.ExifTags.IFD.GPSInfo
        )

        pano_height, pano_width, *_ = pano_image.shape
        if pano_width != pano_height * 2:
            raise ValueError("Only 360° panoramas are supported.")

        if camera is None:  # First image.
            camera = create_virtual_camera(pano_height)
            for rig_camera in rig_config.cameras:
                rig_camera.camera = camera
            pano_size = (pano_width, pano_height)
            rays_in_cam = get_virtual_camera_rays(camera)  # Precompute.
        else:
            if (pano_width, pano_height) != pano_size:
                raise ValueError(
                    "Panoramas of different sizes are not supported."
                )

        for cam_idx, cam_from_pano_r in enumerate(cams_from_pano_rotation):
            rays_in_pano = rays_in_cam @ cam_from_pano_r
            xy_in_pano = spherical_img_from_cam(pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(
                camera.width, camera.height, 2
            ).astype(np.float32)
            xy_in_pano -= 0.5  # COLMAP to OpenCV pixel origin.
            image = cv2.remap(
                pano_image,
                *np.moveaxis(xy_in_pano, -1, 0),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            # We define a mask such that each pixel of the panorama has its
            # features extracted only in a single virtual camera.
            closest_camera = np.argmax(rays_in_pano @ cam_centers_in_pano.T, -1)
            mask = (
                ((closest_camera == cam_idx) * 255)
                .astype(np.uint8)
                .reshape(camera.width, camera.height)
            )

            image_name = rig_config.cameras[cam_idx].image_prefix + pano_name
            mask_name = f"{image_name}.png"

            image_path = output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            PIL.Image.fromarray(image).save(image_path, exif=gpsonly_exif)

            mask_path = mask_dir / mask_name
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            if not pycolmap.Bitmap.from_array(mask).write(mask_path):
                raise RuntimeError(f"Cannot write {mask_path}")

    return rig_config


import concurrent.futures as futures

# Optional: let OpenCV manage threads (or set a fixed number, e.g., 8)
cv2.setNumThreads(0)

def build_remaps_and_masks(camera, pano_size, cams_from_pano_rotation):
    """Precompute xmap/ymap per cam and per-cam masks (independent of image)."""
    rays_in_cam = get_virtual_camera_rays(camera)  # (W*H, 3)
    w, h = camera.width, camera.height

    # camera centers for mask assignment (once)
    cam_centers_in_pano = np.einsum("nij,i->nj", cams_from_pano_rotation, [0, 0, 1])

    xymaps = []
    masks = []

    for cam_idx, cam_from_pano_r in enumerate(cams_from_pano_rotation):
        # Rays rotated into pano
        rays_in_pano = rays_in_cam @ cam_from_pano_r  # (W*H, 3)

        # Pixel coords in pano (float32, OpenCV wants float32)
        xy_in_pano = spherical_img_from_cam(pano_size, rays_in_pano).astype(np.float32)
        xy_in_pano = xy_in_pano.reshape(w, h, 2)
        xy_in_pano -= 0.5  # COLMAP -> OpenCV pixel origin

        # Split to x/y maps once
        xmap, ymap = np.moveaxis(np.ascontiguousarray(xy_in_pano), -1, 0)
        xymaps.append((xmap, ymap))

        # Closest-camera mask (purely geometric; same for all images)
        closest_camera = np.argmax(rays_in_pano @ cam_centers_in_pano.T, -1)
        mask = ((closest_camera == cam_idx) * 255).astype(np.uint8).reshape(w, h)
        masks.append(mask)

    return xymaps, masks


def run(args):
    input_image_dir = args.input_image_path
    output_base_dir = args.output_path

    # Create output directories for each camera
    for i in range(16):
        (output_base_dir / f"pano_camera{i}").mkdir(exist_ok=True, parents=True)
    mask_dir = output_base_dir / "masks"
    mask_dir.mkdir(exist_ok=True, parents=True)

    # Find images
    pano_image_names = [p.name for p in sorted(input_image_dir.glob("frame_*.png"))]
    logging.info(f"Found {len(pano_image_names)} images in {input_image_dir}.")

    # Virtual cameras (do once)
    cams_from_pano_rotation = get_virtual_rotations()
    logging.info(f"Generating {len(cams_from_pano_rotation)} virtual camera views")

    camera = pano_size = None
    xymaps = masks = None

    # Small thread pool for disk writes (encoding is I/O-bound)
    writer_pool = futures.ThreadPoolExecutor(max_workers=4)

    for pano_name in tqdm(pano_image_names, desc="Processing panorama images"):
        pano_path = input_image_dir / pano_name
        # Read once, directly to ndarray, contiguous; enforce 3 channels
        pano_image = np.asarray(PIL.Image.open(pano_path).convert("RGB"))
        pano_image = np.ascontiguousarray(pano_image)
        pano_height, pano_width, _ = pano_image.shape

        if pano_width != pano_height * 2:
            raise ValueError("Only 360° panoramas are supported.")

        if camera is None:
            camera = create_virtual_camera(pano_height)
            pano_size = (pano_width, pano_height)

            # Build maps & masks ONCE for all cams and reuse across images
            xymaps, masks = build_remaps_and_masks(camera, pano_size, cams_from_pano_rotation)

        # Reuse a single output buffer to avoid repeated allocations
        dst = None

        for cam_idx, (maps, mask) in enumerate(zip(xymaps, masks)):
            xmap, ymap = maps

            # cv2.remap lets you pass an existing dst buffer to reuse memory
            dst = cv2.remap(
                pano_image, xmap, ymap,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
                dst=dst  # reuse buffer
            )

            # Save image asynchronously (speed up I/O)
            cam_dir = output_base_dir / f"pano_camera{cam_idx}"
            image_path = cam_dir / pano_name
            mask_path = mask_dir / f"pano_camera{cam_idx}/{pano_name}.png"
            mask_path.parent.mkdir(exist_ok=True, parents=True)

            # Use OpenCV to write (usually faster than PIL); convert RGB->BGR
            writer_pool.submit(cv2.imwrite, str(image_path), dst[..., ::-1])
            writer_pool.submit(cv2.imwrite, str(mask_path), mask)

    logging.info(f"Successfully processed {len(pano_image_names)} panorama images")
    logging.info(f"Generated 16 virtual camera views saved to {output_base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 360° panorama images to 16 virtual camera views")
    parser.add_argument(
        "--input_image_path", 
        type=Path, 
        default=Path("/data/ipek_insta360/3dgut_data/outdoors_panorama/output/image aas"),
        help="Input directory containing panorama images"
    )
    parser.add_argument(
        "--output_path", 
        type=Path, 
        default=Path("/data/ipek_insta360/3dgut_data/outdoors_panorama/output/images_16"),
        help="Output base directory for virtual camera images"
    )
    run(parser.parse_args())