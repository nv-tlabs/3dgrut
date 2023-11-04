import torch
import numpy.typing as npt
from typing import Optional, Callable

def inverse_sigmoid(x):
    return torch.log(x/(1-x))



def to_torch(data: npt.NDArray, device: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Converts a numpy array to a torch tensor on target device with optional type-casting"""
    return torch.from_numpy(data).to(device=device, dtype=dtype)


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

   
