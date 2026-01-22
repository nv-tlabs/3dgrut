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

def get_cubemap_face_rays(face_id: int, face_size: int) -> np.ndarray:
    yy, xx = np.indices((face_size, face_size), dtype=np.float32)
    a = 2.0 * (xx + 0.5) / face_size - 1.0
    b = 2.0 * (yy + 0.5) / face_size - 1.0

    if face_id == 0:       # +X
        x = np.ones_like(a);  y = b;            z = -a
    elif face_id == 1:     # -X
        x = -1*np.ones_like(a); y = b;          z = a
    elif face_id == 2:     # +Z
        x = a;              y = b;              z = np.ones_like(a)
    elif face_id == 3:     # -Z
        x = -a;             y = b;              z = -np.ones_like(a)
    elif face_id == 4:     # -Y (UP)
        x = a;              y = -np.ones_like(a); z = b
    elif face_id == 5:     # +Y (DOWN)
        x = a;              y = np.ones_like(a);  z = -b
    else:
        raise ValueError("Invalid face_id")

    rays = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays

def spherical_img_from_cam(image_size, rays_in_cam: np.ndarray) -> np.ndarray:
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size

def reconstruct_pano_from_cubemap_masks(cube_masks, pano_size):
    pano_w, pano_h = pano_size
    pano_mask = np.zeros((pano_h, pano_w), dtype=np.uint8)

    for face_id, face_mask in enumerate(cube_masks):
        if face_mask is None:
            continue

        face_h, face_w = face_mask.shape
        assert face_h == face_w
        face_size = face_h

        # 1. Rays for this face
        rays = get_cubemap_face_rays(face_id, face_size)

        # 2. Map rays → equirectangular pixel locations
        xy = spherical_img_from_cam((pano_w, pano_h), rays)

        x = np.clip(np.floor(xy[:, 0]).astype(np.int32), 0, pano_w - 1)
        y = np.clip(np.floor(xy[:, 1]).astype(np.int32), 0, pano_h - 1)

        # 3. Scatter OR into panorama
        pano_mask[y, x] = np.maximum(
            pano_mask[y, x],
            face_mask.reshape(-1)
        )

    return pano_mask

def run(args):
    input_dir = Path(args.input_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    num_faces = 6

    # collect all mask filenames from face 0
    mask_dir0 = input_dir / "pano_camera0" / "masks"
    mask_files = sorted(mask_dir0.glob("frame_*.png"))
    # breakpoint()

    if not mask_files:
        print("No masks found")
        return

    # Determine pano size from cube face size
    sample_mask = np.array(PIL.Image.open(mask_files[0]))
    face_size = sample_mask.shape[0]
    pano_size = (2 * face_size, face_size)  # (W, H)

    print("Reconstructing equirect masks:")
    print(f"  Cubemap face size: {face_size}x{face_size}")
    print(f"  Panorama size:     {pano_size[0]}x{pano_size[1]}")

    for mpath in tqdm(mask_files):
        frame = mpath.name

        # load masks for all 6 faces
        cube_masks = []
        for face_id in range(num_faces):
            fmask = input_dir / f"pano_camera{face_id}" /"masks"/ frame
            if fmask.exists():
                m = np.array(PIL.Image.open(fmask))
                if m.ndim == 3:
                    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
                cube_masks.append(m)
            else:
                cube_masks.append(None)
                breakpoint()

        # reconstruct
        pano_mask = reconstruct_pano_from_cubemap_masks(cube_masks, pano_size)
        
        # write .csv file
        rows = []
        pano_w, pano_h = pano_size
        vis = cv2.cvtColor(pano_mask, cv2.COLOR_GRAY2BGR)

        cx = (face_size - 1) * 0.5
        cy = (face_size - 1) * 0.5
        center_idx = int(round(cy)) * face_size + int(round(cx))

        for cam_idx in range(num_faces):
            cam_mask_for_area = cube_masks[cam_idx]
            if cam_mask_for_area is None:
                continue

            # area = connected component area at face center
            bin_mask = (cam_mask_for_area > 127).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
            x0 = int(round(cx)); y0 = int(round(cy))
            lbl = labels[y0, x0]
            area_px = int(stats[lbl, cv2.CC_STAT_AREA]) if lbl > 0 else 0
            weight = float(area_px)

            # face center ray in pano coordinates
            rays = get_cubemap_face_rays(cam_idx, face_size)
            x, y, z = rays[center_idx]

            # ===== PASTE YOUR OLD CODE EXACTLY HERE =====
            yaw = np.arctan2(x, z)
            pitch = -np.arcsin(y)
            u = (yaw / np.pi + 1.0) * 0.5
            v = (-2.0 * pitch / np.pi + 1.0) * 0.5

            px = int(np.clip(u * pano_w, 0, pano_w - 1))
            py = int(np.clip(v * pano_h, 0, pano_h - 1))

            rows.append((frame, cam_idx, u, v, int(px), int(py), float(yaw), float(pitch), area_px, weight))
            cv2.circle(vis, (px, py), 4, (0, 0, 255), 30)

        # save
        out_path = output_dir / frame
        PIL.Image.fromarray(pano_mask).save(out_path)
        
        centers_dir = output_dir / "centers"
        centers_dir.mkdir(exist_ok=True, parents=True)
        csv_path = centers_dir / Path(frame).with_suffix(".csv")

        with open(csv_path, "w") as f:
            f.write("filename,cam_idx,u,v,px,py,yaw_rad,pitch_rad,area_px,weight\n")
            for r in rows:
                fname, cami, uu, vv, pxx, pyy, yawr, pr, area_px, w = r
                f.write(f"{fname},{cami},{uu:.6f},{vv:.6f},{pxx},{pyy},{yawr:.9f},{pr:.9f},{area_px},{w:.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    run(parser.parse_args())