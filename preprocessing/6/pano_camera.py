"""
An example for running incremental SfM on 360 spherical panorama images.
Modified to process omni images and generate 6 virtual camera views.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import PIL.ExifTags
import PIL.Image
from tqdm import tqdm

import pycolmap
from pycolmap import logging

def get_cubemap_face_rays(face_id: int, face_size: int) -> np.ndarray:
    # Pixel centers in [-1, 1]
    yy, xx = np.indices((face_size, face_size), dtype=np.float32)
    a = 2.0 * (xx + 0.5) / face_size - 1.0   # horizontal in [-1,1]
    b = 2.0 * (yy + 0.5) / face_size - 1.0   # vertical in [-1,1]

    if face_id == 0:       # +X (right)
        x = np.ones_like(a)
        y = b
        z = -a
    elif face_id == 1:     # -X (left)
        x = -np.ones_like(a)
        y = b
        z = a
    elif face_id == 2:     # +Z (front)
        x = a
        y = b
        z = np.ones_like(a)
    elif face_id == 3:     # -Z (back)
        x = -a
        y = b
        z = -np.ones_like(a)
    elif face_id == 4:     # -Y (up)
        x = a
        y = -np.ones_like(a)
        z = b
    elif face_id == 5:     # +Y (down)
        x = a
        y = np.ones_like(a)
        z = -b
    else:
        raise ValueError(f"Invalid face_id {face_id}")

    rays = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays

def create_virtual_camera(pano_height: int) -> pycolmap.Camera:
    """Create a virtual perspective camera."""
    image_size = pano_height
    focal = image_size / 2
    return pycolmap.Camera.create(0, "PINHOLE", focal, image_size, image_size)

def spherical_img_from_cam(image_size, rays_in_cam: np.ndarray) -> np.ndarray:
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size

def lookat_rotation(forward, up):
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack([right, up, forward], axis=1)  # camera basis vectors

def run(args):
    input_image_dir = args.input_image_path
    output_base_dir = args.output_path

    # Create output directories for each face
    for i in range(6):
        cam_dir = output_base_dir / f"pano_camera{i}"
        cam_dir.mkdir(exist_ok=True, parents=True)

    pano_image_names = [p.name for p in sorted(input_image_dir.glob("frame_*.png"))]
    logging.info(f"Found {len(pano_image_names)} images in {input_image_dir}.")
    if not pano_image_names:
        logging.error(f"No frame_*.png images found in {input_image_dir}")
        return

    camera = None
    pano_size = None

    for pano_name in tqdm(pano_image_names, desc="Processing panorama images"):
        pano_path = input_image_dir / pano_name
        try:
            pano_image = PIL.Image.open(pano_path)
        except PIL.Image.UnidentifiedImageError:
            logging.info(f"Skipping file {pano_path} as it cannot be read.")
            continue

        pano_image = np.asarray(pano_image)
        pano_height, pano_width, *_ = pano_image.shape

        if pano_width != pano_height * 2:
            logging.warning(
                f"Image {pano_name} is not a 360° panorama (width={pano_width}, "
                f"height={pano_height}). Skipping."
            )
            continue

        if camera is None:
            camera = create_virtual_camera(pano_height)
            pano_size = (pano_width, pano_height)
        else:
            if (pano_width, pano_height) != pano_size:
                logging.warning(f"Image {pano_name} has different size. Skipping.")
                continue

        face_size = pano_height  # standard cubemap: each face is H x H

        for face_id in range(6):
            # Rays already in pano/world coords for this face
            rays_in_pano = get_cubemap_face_rays(face_id, face_size)

            xy_in_pano = spherical_img_from_cam(pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(face_size, face_size, 2).astype(np.float32)
            xy_in_pano -= 0.5  # pixel-center convention for cv2.remap

            face_img = cv2.remap(
                pano_image,
                *np.moveaxis(xy_in_pano, -1, 0),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )

            cam_dir = output_base_dir / f"pano_camera{face_id}"
            image_path = cam_dir / pano_name
            PIL.Image.fromarray(face_img).save(image_path)

    logging.info(f"Successfully processed {len(pano_image_names)} panorama images")
    logging.info(f"Generated 6 cubemap faces per panorama in {output_base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 360° panorama images to 6 virtual camera views")
    parser.add_argument(
        "--input_image_path", 
        type=Path, 
        default=Path("/data/ipek_insta360/3dgut_data/outdoors_panorama/output/images"),
        help="Input directory containing panorama images"
    )
    parser.add_argument(
        "--output_path", 
        type=Path, 
        default=Path("/data/ipek_insta360/3dgut_data/outdoors_panorama/output/images_6"),
        help="Output base directory for virtual camera images"
    )
    run(parser.parse_args())