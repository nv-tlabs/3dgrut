import cv2
import numpy as np
import pycolmap
from lib.cam_utils import *
from lib.cam_utils import radial_mask
import csv
import os
import glob
import sys

### ----- config -----
for i, a in enumerate(sys.argv[1:]):
    if a == "--scene" and i + 2 <= len(sys.argv):
        scene = sys.argv[1:][i + 1].strip("\"'")
    if a == "--project" and i + 2 <= len(sys.argv):
        project = sys.argv[1:][i + 1].strip("\"'")
        
        
data_dir     = f"/home/youlenda/360/omni/3dgrut/data/{project}/{scene}"
omni_dir     = f"{data_dir}/omni"
centers_dir  = f"{data_dir}/pre_masking_6/reconstructed_masks/centers/"
out_fi_dir   = f"{data_dir}/pre_masking_6/fisheye_syn"
# out_fi_dir   = f"fisheye_syn"
out_R_table  = f"{data_dir}/pre_masking_6/synth_R_list.csv"

colmap_dir   = f"/home/youlenda/360/omni/3dgrut/data/{project}/{scene}/sparse/0"
int_rot_dir  = "examples/no_lens_guards/42/out/optimized.txt"
radius_r     = 107
camera_size  = 2880


os.makedirs(out_fi_dir, exist_ok=True)

### ----- load camera & image -----
with open(int_rot_dir, "r") as f:
    lines = f.readlines()

params_r = eval(lines[1].split('params=')[1].split('(')[0].strip())

rec = pycolmap.Reconstruction()
try:
    rec.read(colmap_dir)
except:
    rec.read(f"/home/youlenda/360/omni/3dgrut/data/party_room/sparse/0")

camera_r = pycolmap.Camera(camera_id=1, model=pycolmap.CameraModelId.OPENCV_FISHEYE,
                           width=2880, height=2880, params=params_r)

def fisheye_theta_from_radius(r, f, k1, k2, k3, k4):
    # Newton-Raphson on r_d(θ) - r = 0
    theta = min(np.pi, r / max(f, 1e-6))  # good initial guess
    for _ in range(50):
        t2 = theta*theta
        t4 = t2*t2
        t6 = t4*t2
        t8 = t4*t4
        poly  = 1 + k1*t2 + k2*t4 + k3*t6 + k4*t8
        r_d   = f * theta * poly
        if abs(r_d - r) < 1e-9:
            break
        # derivative dr_d/dθ
        dpoly = 2*k1*theta + 4*k2*t3 if (t3:=t2*theta) else 0.0
        dpoly+= 6*k3*(t5:=t4*theta) + 8*k4*(t7:=t6*theta)
        drd   = f * (poly + theta * dpoly)
        theta = theta - (r_d - r)/max(drd, 1e-12)
    return theta

# Example usage once you know params_r:
# fx, fy, cx, cy, k1, k2, k3, k4 = params_r
# r_eff = 1332.5
# theta_max = fisheye_theta_from_radius(r_eff, fx, k1, k2, k3, k4)
# FOV_degrees = np.degrees(2*theta_max)
# breakpoint()
W_eq, H_eq = camera_r.width * 2, camera_r.height
img_paths = sorted(glob.glob(os.path.join(omni_dir, "*.png")))

# --- precompute once (constant for all frames) ---
W_fi, H_fi = camera_r.width, camera_r.height

dirs_cam = get_cam_ray_dirs(camera_r, mask=True, radius=radius_r).cpu().numpy()  # (H,W,3) or (H*W,3)

r, Rpix = radial_mask(W_fi, H_fi)
lens_mask = (r < (Rpix - (radius_r + 0.5)))

for img_path in img_paths:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    img_omni = cv2.imread(img_path)
    
    centers_csv = os.path.join(centers_dir, f"{stem}.csv")
    
    rows = []
    with open(centers_csv, "r") as f:
        rdict = csv.DictReader(f)
        for row in rdict:
            u = float(row["u"]); v = float(row["v"])
            w = float(row.get("area_px") or row.get("weight", 1.0))
            x_px = u * W_eq; y_px = v * H_eq
            rows.append((x_px, y_px, w))

    uv_pix  = np.array([[x, y] for x, y, _ in rows], dtype=np.float64)
    weights = np.array([w for _, _, w in rows], dtype=np.float64)
    lats, lons = uv_to_latlon(uv_pix, W_eq, H_eq)
    xyzs = latlon_to_xyz(-lats, -lons)
    if np.all(weights <= 0):
        # fallback to unweighted if all weights are zero/non-positive
        xyz_avg = np.mean(xyzs, axis=0)
    else:
        xyz_avg = np.average(xyzs, axis=0, weights=weights)
    
    R_synthetic = look_at_camZ(xyz_avg) 
    
    with open(out_R_table, "a", newline="") as f:
        rowR = [os.path.basename(img_path)] + list(R_synthetic.reshape(-1))
        csv.writer(f).writerow(rowR)


    xyz_world = (R_synthetic @ dirs_cam.T).T
    lat_s, lon_s = xyz_to_latlon(xyz_world)
    u_s, v_s = latlon_to_uv(lat_s, -lon_s, W_eq, H_eq)

    map_x = np.full((H_fi, W_fi), -1, np.float32)
    map_y = np.full((H_fi, W_fi), -1, np.float32)    

    map_x[lens_mask] = u_s.astype(np.float32)
    map_y[lens_mask] = v_s.astype(np.float32)

    fisheye_s = cv2.remap(
        img_omni, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    out_path = os.path.join(out_fi_dir, f"{stem}.png")
    ok = cv2.imwrite(out_path, fisheye_s)
    print(f"[ok] {stem}  | fisheye: {ok}")