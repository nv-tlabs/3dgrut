import logging
from typing import Optional, Callable, TypeVar

import torch
import numpy as np
import numpy.typing as npt
from typing import Optional, Callable
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


OmegaConf.register_new_resolver("div", lambda a, b: a / b)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

T = TypeVar("T")
def unpack_optional(optional: Optional[T]) -> T:
    """Unpacks the value of an optional, raising if value is missing"""
    if optional is None:
        raise ValueError("Can't unpack empty optional")

    return optional

def to_torch(data: npt.NDArray, device: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Converts a numpy array to a torch tensor on target device with optional type-casting"""
    return torch.from_numpy(data).to(device=device, dtype=dtype)

def to_torch_optional(
    data: Optional[npt.NDArray], device: str, dtype: Optional[torch.dtype] = None
) -> Optional[torch.Tensor]:
    """Wrapper of 'to_torch', simply bypassing tensor creation if input is 'None'"""
    return to_torch(data, device, dtype) if data is not None else None

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

    R = torch.zeros((q.size(0), 3, 3), dtype=r.dtype, device=r.device)

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


def sh_degree_to_specular_dim(degree):
    """ Number of dimensions used by SH of deg [1..degree], inclusive """
    return 3 * ((degree + 1) ** 2 - 1)


def sh_degree_to_num_features(degree):
    """ Number of dimensions used by SH of deg [0..degree], inclusive """
    return sh_degree_to_specular_dim(degree) + 3



# FROM 3DGS

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def get_git_commit():
    import git
    """ Get the version of the code running, useful for tracking history of experiments. """
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except git.InvalidGitRepositoryError as e:
        sha = 'version info n/a'
        logging.warning('Could not fetch current git commit, this is normal for deployed / copied code.\n'
                        'You can get your git information by deploying the .git folder which may be ignored '
                        'by your IDE.')
    except Exception as e:
        sha = 'version info n/a'
    return sha
