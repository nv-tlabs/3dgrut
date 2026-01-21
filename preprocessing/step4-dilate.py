import cv2
import numpy as np
import os
import sys

# ----- configs -----
for i, a in enumerate(sys.argv[1:]):
    if a == "--scene" and i + 2 <= len(sys.argv):
        scene = sys.argv[1:][i + 1].strip("\"'")
    if a == "--method" and i + 2 <= len(sys.argv):
        method = sys.argv[1:][i + 1].strip("\"'")
    if a == "--camera" and i + 2 <= len(sys.argv):
        camera = sys.argv[1:][i + 1].strip("\"'")
    if a == "--iter" and i + 2 <= len(sys.argv):
        dilate_iter = int(sys.argv[1:][i + 1].strip("\"'"))
    if a == "--project" and i + 2 <= len(sys.argv):
        project = sys.argv[1:][i + 1].strip("\"'")



data_dir = f"/home/youlenda/360/omni/3dgrut/data/{project}/{scene}"
folder   = f"{data_dir}/out-{method}/{camera}"

_cam_map = {"rear": "camera2", "front": "camera1"}
out_cam  = _cam_map.get(camera.lower(), camera) 

dilate_kernel = 9
# dilate_iter = 3 # original
# dilate_iter = 20  # extra

out_dir         = f"{data_dir}/out-{method}/masks-{dilate_iter}/{out_cam}"
# out_dir_overlay = f"{data_dir}/out-{method}/masks/{out_cam}_overlay"
images_dir      = f"{data_dir}/images/{out_cam}"

def load_as_binary(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    return bin_img.astype(np.uint8)

# ----- loop -----
os.makedirs(out_dir, exist_ok=True)
# os.makedirs(out_dir_overlay, exist_ok=True)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))

for filename in sorted(os.listdir(folder)):
    p = os.path.join(folder, filename)
    m = load_as_binary(p)
    out = cv2.dilate(m, kernel, iterations=dilate_iter)

    base = os.path.splitext(os.path.basename(p))[0]
    out_name = base + "_mask.png"
    out_path = os.path.join(out_dir, out_name)
    k = cv2.imwrite(out_path, out)
    print(f"{out_path} Done.")

    # img_path = os.path.join(images_dir, os.path.basename(p))
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # color_layer = np.full_like(img, (0, 0, 255), dtype=np.uint8)
    # blended = cv2.addWeighted(img, 1 - 0.5, color_layer, 0.5, 0)

    # overlay = img.copy()
    # mask_bool = out > 0
    # overlay[mask_bool] = blended[mask_bool]

    # overlay_name = base + "_overlay.png"
    # overlay_path = os.path.join(out_dir_overlay, overlay_name)
    # ok2 = cv2.imwrite(overlay_path, overlay)
    # print(f"{overlay_path} Done.")

print("Done.")
