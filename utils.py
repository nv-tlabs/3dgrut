import torch
import numpy as np
import numpy.typing as npt
from typing import Optional, Callable
from torch.optim.lr_scheduler import ExponentialLR, StepLR

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def to_torch(data: npt.NDArray, device: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Converts a numpy array to a torch tensor on target device with optional type-casting"""
    return torch.from_numpy(data).to(device=device, dtype=dtype)

def to_np(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().cpu().numpy()

def show_image(imgs, figAxes=None):
    """Utility to show a list of image tensors"""
    if not isinstance(imgs, list):
        imgs = [imgs]
    newFigure = not figAxes
    if newFigure:
        _, axes = plt.subplots(ncols=len(imgs), squeeze=False)
        figAxes = []
    for i, img in enumerate(imgs):
        if newFigure:
            figAxes.append(axes[0, i].imshow(img))
            axes[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        else:
            figAxes[i].set_data(img)
    return figAxes    


ACTIVATION_DICT: dict[str, Callable[..., torch.Tensor]] = {
    "sigmoid": torch.sigmoid,
    "exp": torch.exp,
    "normalize": torch.nn.functional.normalize,
    }
         
INVERSE_ACTIVATION_DICT: dict[str, Callable[..., torch.Tensor]] = {
    "sigmoid": inverse_sigmoid,
    "exp": torch.log,
    }


def get_activation_function(activation_function: str, inverse=False) -> Callable:
    if not inverse:
        return ACTIVATION_DICT[activation_function]
    else:
        return INVERSE_ACTIVATION_DICT[activation_function]




def quaternion_to_so3(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def exponential_scheduler(lr_init, lr_final, max_steps=1000000, type=""):
    def helper(step):
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp

    return helper

def skip_scheduler(type=""):
    def helper(step):
        return None
    return helper


SCHEDULER_DICT: dict[str, Callable] = {
    'exp': exponential_scheduler,
    'skip': skip_scheduler
}

def get_scheduler(scheduler: str) -> Callable:
    return SCHEDULER_DICT[scheduler]

