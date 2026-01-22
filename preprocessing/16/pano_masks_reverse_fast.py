"""
Convert virtual camera SAM masks back to 360° spherical panorama masks.
This processes the SAM masks generated from virtual camera views.
"""

import argparse
from pathlib import Path
from collections.abc import Sequence

import cv2
import numpy as np
import PIL.Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pycolmap
from pycolmap import logging


def create_virtual_camera(pano_height: int, fov_deg: float = 90) -> pycolmap.Camera:
    """Create a virtual perspective camera (same as original)."""
    image_size = int(pano_height * fov_deg / 180)
    focal = image_size / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    return pycolmap.Camera.create(0, "PINHOLE", focal, image_size, image_size)


def get_virtual_rotations(
    num_cameras: int = 16,
    num_steps_yaw: int = 8, 
    pitches_deg: Sequence[float] = (-35.0, 35.0)
) -> Sequence[np.ndarray]:
    """Get the relative rotations of the virtual cameras w.r.t. the panorama.
    
    Args:
        num_cameras: Total number of virtual cameras
        num_steps_yaw: Number of yaw steps per pitch level
        pitches_deg: Pitch angles to use
    """
    cams_from_pano_r = []
    
    # Calculate number of cameras per pitch level
    num_pitch_levels = len(pitches_deg)
    if num_cameras % num_pitch_levels != 0:
        logging.warning(f"Number of cameras ({num_cameras}) is not evenly divisible by number of pitch levels ({num_pitch_levels})")
        logging.warning(f"Some cameras may not be used or some pitch levels may have different numbers of cameras")
    
    cameras_per_pitch = num_cameras // num_pitch_levels
    remaining_cameras = num_cameras % num_pitch_levels
    
    for i, pitch_deg in enumerate(pitches_deg):
        # Distribute remaining cameras across pitch levels
        cams_this_pitch = cameras_per_pitch + (1 if i < remaining_cameras else 0)
        
        if cams_this_pitch == 0:
            continue
            
        yaws = np.linspace(0, 360, cams_this_pitch, endpoint=False)
        yaw_offset = (360 / cams_this_pitch / 2) if pitch_deg > 0 else 0
        
        for yaw_deg in yaws + yaw_offset:
            cam_from_pano_r = Rotation.from_euler(
                "XY", [-pitch_deg, -yaw_deg], degrees=True
            ).as_matrix()
            cams_from_pano_r.append(cam_from_pano_r)
    
    # Ensure we return exactly num_cameras rotations
    if len(cams_from_pano_r) > num_cameras:
        cams_from_pano_r = cams_from_pano_r[:num_cameras]
    elif len(cams_from_pano_r) < num_cameras:
        logging.warning(f"Generated {len(cams_from_pano_r)} camera rotations, expected {num_cameras}")
    
    logging.info(f"Generated {len(cams_from_pano_r)} virtual camera rotations")
    return cams_from_pano_r


def get_panorama_coordinates(pano_size: tuple) -> np.ndarray:
    """Get all pixel coordinates in panorama image as (u, v) normalized coordinates."""
    pano_width, pano_height = pano_size
    
    # Create coordinate grid
    v_coords, u_coords = np.meshgrid(
        np.arange(pano_height), np.arange(pano_width), indexing='ij'
    )
    
    # Normalize to [0, 1]
    u_norm = (u_coords + 0.5) / pano_width
    v_norm = (v_coords + 0.5) / pano_height
    
    return np.stack([u_norm.ravel(), v_norm.ravel()], axis=1)


def spherical_to_direction(uv_coords: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates (u, v) to 3D direction vectors."""
    u, v = uv_coords[:, 0], uv_coords[:, 1]
    
    # Convert to spherical angles
    yaw = (u * 2 - 1) * np.pi
    pitch = -(v * 2 - 1) * np.pi / 2
    
    # Convert to 3D directions
    x = np.sin(yaw) * np.cos(pitch)
    y = -np.sin(pitch)
    z = np.cos(yaw) * np.cos(pitch)
    
    return np.stack([x, y, z], axis=1)


def project_to_camera(directions: np.ndarray, cam_from_pano_r: np.ndarray, 
                     camera: pycolmap.Camera) -> tuple:
    """Project 3D directions to camera image coordinates."""
    # Transform directions to camera coordinate system
    directions_cam = directions @ cam_from_pano_r.T
    
    # Only keep directions that point forward (positive z)
    valid_mask = directions_cam[:, 2] > 0
    
    if not np.any(valid_mask):
        return None, None
    
    # Get valid directions and convert to homogeneous coordinates
    directions_cam_valid = directions_cam[valid_mask]
    
    # Create 3D points in camera frame (at unit distance)
    points_3d = directions_cam_valid
    
    # Convert to pixel coordinates using the updated API
    xy_pixels = camera.img_from_cam(points_3d)
    
    # Check which pixels are within image bounds
    in_bounds = (
        (xy_pixels[:, 0] >= 0) & 
        (xy_pixels[:, 0] < camera.width) &
        (xy_pixels[:, 1] >= 0) & 
        (xy_pixels[:, 1] < camera.height)
    )
    
    if not np.any(in_bounds):
        return None, None
    
    xy_pixels_valid = xy_pixels[in_bounds]
    global_indices = np.where(valid_mask)[0][in_bounds]
    
    return xy_pixels_valid, global_indices


def sample_mask_nearest(mask: np.ndarray, xy_coords: np.ndarray) -> np.ndarray:
    """Sample mask values using nearest neighbor interpolation."""
    h, w = mask.shape[:2]
    x, y = xy_coords[:, 0], xy_coords[:, 1]
    
    # Clamp coordinates to image bounds
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    
    # Round to nearest pixel
    x_int = np.round(x).astype(np.int32)
    y_int = np.round(y).astype(np.int32)
    
    # Sample mask values
    sampled = mask[y_int, x_int]
    
    return sampled


def reconstruct_panorama_mask_from_cameras(
    virtual_camera_masks: list,
    cams_from_pano_rotation: Sequence[np.ndarray],
    pano_size: tuple,
    camera: pycolmap.Camera
) -> np.ndarray:
    """Reconstruct panorama mask from virtual camera masks using OR operation."""
    pano_width, pano_height = pano_size
    
    # Initialize panorama mask (binary)
    pano_mask = np.zeros((pano_height, pano_width), dtype=np.uint8)
    
    # Get all panorama pixel coordinates
    pano_uv = get_panorama_coordinates(pano_size)
    pano_directions = spherical_to_direction(pano_uv)
    
    for cam_idx, (cam_mask, cam_from_pano_r) in enumerate(
        zip(virtual_camera_masks, cams_from_pano_rotation)
    ):
        if cam_mask is None:
            continue
            
        logging.info(f"Processing camera mask {cam_idx + 1}/{len(virtual_camera_masks)}")
        
        # Ensure mask is binary (0 or 255)
        cam_mask_binary = (cam_mask > 127).astype(np.uint8) * 255
        
        # Project panorama pixels to this camera
        xy_pixels, global_indices = project_to_camera(
            pano_directions, cam_from_pano_r, camera
        )
        
        if xy_pixels is None:
            continue
        
        # Sample from camera mask using nearest neighbor
        sampled_mask_values = sample_mask_nearest(cam_mask_binary, xy_pixels)
        
        # Convert global indices back to 2D panorama coordinates
        pano_indices_2d = np.unravel_index(global_indices, (pano_height, pano_width))
        
        # Use OR operation to combine masks (any camera that sees a person will mark that pixel)
        pano_mask[pano_indices_2d] = np.maximum(
            pano_mask[pano_indices_2d], 
            sampled_mask_values
        )
    
    return pano_mask


def find_mask_files(input_base_dir: Path, num_cameras: int) -> list:
    """Find all available mask files across all cameras."""
    mask_files = set()
    
    for cam_idx in range(num_cameras):
        cam_mask_dir = input_base_dir / f"pano_camera{cam_idx}" / "masks"
        if cam_mask_dir.exists():
            for mask_path in cam_mask_dir.glob("frame_*.jpg"):
                mask_files.add(mask_path.name)
            # Also check for PNG files
            for mask_path in cam_mask_dir.glob("frame_*.png"):
                mask_files.add(mask_path.name)
    
    return sorted(list(mask_files))


def auto_detect_num_cameras(input_base_dir: Path) -> int:
    """Auto-detect the number of cameras by counting pano_camera directories."""
    camera_dirs = list(input_base_dir.glob("pano_camera*"))
    if not camera_dirs:
        return 0
    
    # Extract camera numbers and find the maximum
    camera_numbers = []
    for cam_dir in camera_dirs:
        try:
            cam_num = int(cam_dir.name.replace("pano_camera", ""))
            camera_numbers.append(cam_num)
        except ValueError:
            continue
    
    if not camera_numbers:
        return 0
    
    # Return the count (max number + 1, since cameras are 0-indexed)
    detected_num = max(camera_numbers) + 1
    logging.info(f"Auto-detected {detected_num} cameras (pano_camera0 to pano_camera{max(camera_numbers)})")
    return detected_num


def run_mask_reconstruction(args):
    """Main mask reconstruction function."""
    input_base_dir = Path(args.input_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine number of cameras
    if args.num_cameras == -1:
        num_cameras = auto_detect_num_cameras(input_base_dir)
        if num_cameras == 0:
            logging.error(f"No camera directories found in {input_base_dir}")
            return
    else:
        num_cameras = args.num_cameras
    
    logging.info(f"Using {num_cameras} virtual cameras")
    
    # Get virtual camera rotations
    cams_from_pano_rotation = get_virtual_rotations(
        num_cameras=num_cameras,
        num_steps_yaw=args.num_steps_yaw,
        pitches_deg=args.pitches_deg
    )
    # Find all available mask files
    mask_file_names = find_mask_files(input_base_dir, num_cameras)
    
    if not mask_file_names:
        logging.error(f"No mask files found in {input_base_dir}")
        return
    
    logging.info(f"Found {len(mask_file_names)} mask files to reconstruct")
    
    # Determine camera and panorama size from first available mask
    first_mask = None
    for cam_idx in range(num_cameras):
        cam_mask_dir = input_base_dir / f"pano_camera{cam_idx}" / "masks"
        if cam_mask_dir.exists():
            first_mask_path = cam_mask_dir / mask_file_names[0]
            first_mask = np.array(PIL.Image.open(first_mask_path))
            if len(first_mask.shape) == 3:
                first_mask = cv2.cvtColor(first_mask, cv2.COLOR_RGB2GRAY)
            break
        
    # Determine panorama size and create camera
    cam_height, cam_width = first_mask.shape[:2]
    # Estimate original panorama height from camera image size
    estimated_pano_height = int(cam_height * 180 / args.fov_deg)
    pano_size = (estimated_pano_height * 2, estimated_pano_height)
    
    camera = create_virtual_camera(estimated_pano_height, args.fov_deg)
    
    # Process each mask file
    for mask_name in tqdm(mask_file_names, desc="Reconstructing panorama masks"):
        rows = []
        pano_w, pano_h = pano_size
        fx, fy, cx0, cy0 = camera.params

        for cam_idx, R in enumerate(cams_from_pano_rotation):
            centers_path = input_base_dir / f"pano_camera{cam_idx}" / "centers.csv"
            if not centers_path.exists():
                continue

            # find this frame's center
            cx = cy = np.nan
            with open(centers_path, "r") as f:
                _ = f.readline()  # header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != 3:
                        continue
                    fname_c, sx, sy = parts
                    if fname_c == mask_name:
                        cx, cy = float(sx), float(sy)
                        break
            if not np.isfinite(cx) or not np.isfinite(cy):
                continue

            # pixel -> camera ray (PINHOLE)
            xn = (cx - cx0) / fx
            yn = (cy - cy0) / fy
            dir_cam = np.array([xn, yn, 1.0], dtype=np.float64)
            dir_cam /= np.linalg.norm(dir_cam)

            # rotate back to panorama/world: R maps pano->cam, so pano_dir = R^T * cam_dir
            dir_pano = R.T @ dir_cam
            x, y, z = dir_pano

            # 3D direction -> equirect (match spherical_to_direction inverse)
            yaw = np.arctan2(x, z)           # [-pi, pi]
            pitch = -np.arcsin(y)            # [-pi/2, pi/2]
            u = (yaw / np.pi + 1.0) * 0.5    # [0,1]
            v = (-2.0 * pitch / np.pi + 1.0) * 0.5  # [0,1]

            px = int(np.clip(u * pano_w, 0, pano_w - 1))
            py = int(np.clip(v * pano_h, 0, pano_h - 1))
            
            rows.append((mask_name, cam_idx, u, v, int(px), int(py), float(yaw), float(pitch), 1.0))

            # cv2.circle(vis, (px, py), 4, (0, 0, 255), 30)
        
        centers_dir = output_dir / "centers"
        centers_dir.mkdir(exist_ok=True, parents=True)
        csv_path = centers_dir / Path(mask_name).with_suffix(".csv")

        with open(csv_path, "w") as f:
            f.write("filename,cam_idx,u,v,px,py,yaw_rad,pitch_rad,weight\n")
            for r in rows:
                fname, cami, uu, vv, pxx, pyy, yawr, pr, w = r
                f.write(f"{fname},{cami},{uu:.6f},{vv:.6f},{pxx},{pyy},{yawr:.9f},{pr:.9f},{w:.3f}\n")
    logging.info(f"Mask reconstruction complete. Output saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct 360° panorama masks from virtual camera SAM masks"
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("/data/ipek_insta360/3dgut_data/outdoors_panorama/masks_16"),
        help="Input directory containing pano_camera0, pano_camera1, ... mask subdirectories"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("/data/ipek_insta360/3dgut_data/outdoors_panorama/reconstructed_masks"),
        help="Output directory for reconstructed panorama masks"
    )
    parser.add_argument(
        "--num_cameras",
        type=int,
        default=-1,
        help="Number of virtual cameras to use. Use -1 for auto-detection (default: -1)"
    )
    parser.add_argument(
        "--num_steps_yaw",
        type=int,
        default=8,
        help="Number of yaw steps per pitch level (default: 8)"
    )
    parser.add_argument(
        "--pitches_deg",
        type=float,
        nargs='+',
        default=[-35.0, 35.0],
        help="Pitch angles in degrees (default: -35.0 35.0)"
    )
    parser.add_argument(
        "--fov_deg",
        type=float,
        default=90.0,
        help="Field of view of virtual cameras in degrees (default: 90.0)"
    )
    
    run_mask_reconstruction(parser.parse_args())