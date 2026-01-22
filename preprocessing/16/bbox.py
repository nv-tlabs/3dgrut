# Wrap-aware bounding box for an omnidirectional (equirectangular) mask.
# Uses a circular interval along the horizontal axis so a single "logical" bbox
# can span the left/right edges. If it wraps, we visualize it as two rectangles.

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Path from your note
path = "data/maria/reconstructed_masks/frame_0012.png"

# Load and make sure it's grayscale (binary mask-like)
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(path)

H, W = img.shape[:2]

# Threshold to binary (in case of anti-aliased edges); white=foreground
_, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Collect foreground pixel coordinates
ys, xs = np.where(mask > 0)

if len(xs) == 0:
    raise ValueError("No foreground (white) pixels found in the mask.")

# Vertical bounds are simple (no wrap vertically)
y_min = int(ys.min())
y_max = int(ys.max())

# Horizontal wrap-aware minimal covering arc:
# 1) unique occupied columns (x), sorted
cols = np.unique(xs)
cols.sort()

# 2) find the largest circular gap between consecutive occupied columns
# compute diffs
diffs = np.diff(cols)
# include wrap gap from last to first + W
wrap_gap = (cols[0] + W) - cols[-1]
diffs_circ = np.concatenate([diffs, [wrap_gap]])

# index of the largest gap; bbox should start after that gap
i_max_gap = int(np.argmax(diffs_circ))

# start column = next occupied column after the max gap (modulo n)
n = len(cols)
start_idx = (i_max_gap + 1) % n
x_start = int(cols[start_idx])

# end column is the one at the gap index
x_end_linear = int(cols[i_max_gap])

# Compute width on circular domain (inclusive bbox)
# width = (x_end - x_start) mod W  + 1 (inclusive)
width = ((x_end_linear - x_start) % W) + 1

# Prepare a visualization: draw rectangle(s) on an RGB version
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Helper to draw a rectangle with a given color/thickness
def draw_rect_wrap(vis, x_start, width, y_min, y_max, color=(0,0,255), thickness=4):
    H, W = vis.shape[:2]
    x_end = (x_start + width - 1) % W
    if x_start + width <= W:
        # single rectangle
        cv2.rectangle(vis, (x_start, y_min), (x_start + width - 1, y_max), color, thickness)
    else:
        # wraps: draw two pieces
        # piece 1: x_start .. W-1
        cv2.rectangle(vis, (x_start, y_min), (W-1, y_max), color, thickness)
        # piece 2: 0 .. x_end
        cv2.rectangle(vis, (0, y_min), (x_end, y_max), color, thickness)

draw_rect_wrap(vis, x_start, width, y_min, y_max, color=(0,0,255), thickness=6)

# Save outputs
out_path = "omni_bbox_visualization.png"
cv2.imwrite(out_path, vis)

# Package the numeric bbox too
logical_bbox = {
    "x_start": int(x_start),
    "width": int(width),
    "y_min": int(y_min),
    "y_max": int(y_max),
    "image_size": (int(W), int(H)),
    "wraps": bool(x_start + width > W)
}
print(logical_bbox)
out_path, logical_bbox
# {'x_start': 4804, 'width': 1602, 'y_min': 1249, 'y_max': 2719, 'image_size': (5760, 2880), 'wraps': True}