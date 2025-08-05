
import numpy as np
import matplotlib.pyplot as plt

dataset_type = 'dl3dv'
scene = 'garden'
id = '41'

ours   = f'/home/ipek/forks/private/3dgrut/runs/undistorted/outdoors_panorama-0108_115016/training_images/renders/render_step_013000.png'
#colmap = f'000{id}_col.png'
gt     = f'/home/ipek/forks/private/3dgrut/runs/undistorted/outdoors_panorama-0108_115016/training_images/gt/gt_step_013000.png'
def load_image(path):
    return plt.imread(path)

def compute_error_map(gt, rendered):
    if gt.shape != rendered.shape:
        raise ValueError("Images must have the same dimensions and channels")
    
    #error = np.abs(gt - rendered)
    error = np.uint8(np.abs(rendered - gt) * 255)

    #error_map = np.dot(error[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Mask the background where gt is black
    #mask = np.all(gt[..., :3] == 0, axis=-1)
    #error_map[mask] = 0  # Set background error to zero
    
    return error

def display_image(image, title="Image", cmap=None):
    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def save_image(image, path, cmap=None):
    plt.imsave(path, image, cmap=cmap)

gt = load_image(gt)
render_ours = load_image(ours)
#render_colmap = load_image(colmap)

error_map_ours = compute_error_map(gt, render_ours)
#error_map_colmap = compute_error_map(gt, render_colmap)

cmap = 'RdPu'

save_image(error_map_ours,   f'error.jpg', cmap=cmap)
#save_image(error_map_colmap, f'{scene}_000{id}_{cmap}_err_col.jpg', cmap=cmap)

# save_image(render_ours, f'{scene}_000{id}_5kd.jpg', cmap=cmap)
# save_image(render_colmap, f'{scene}_000{id}_col.jpg', cmap=cmap)