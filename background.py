from abc import ABC, abstractmethod
from typing import Union

import torch 
import omegaconf
from omegaconf import OmegaConf
from einops import rearrange


def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)

def make(name:str, config):
    match name:
        case "sky-mlp":
            return SkyMlp(config=config)
        case "background-color":
            return BackgroundColor(config=config)
        case "skip-background":
            return SkipBackground(config=config)
        case _  : 
            raise NotImplementedError(f"background {name} not implemented, choice must be in [sky-mlp, background-color, skip-background]")


DEFAULT_DEVICE = 'cuda'
class BaseBackground(ABC, torch.nn.Module):
    def __init__(self, config: omegaconf.dictconfig.DictConfig, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.device = DEFAULT_DEVICE

        self.setup(**kwargs)

    @abstractmethod
    def setup(self, **kwargs) -> None:
        raise NotImplementedError("Must override in the child class")

    @abstractmethod
    def forward(self, rays_d, rgb, opacity):
        raise NotImplementedError("Must override in the child class")

    def linear_to_srgb(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x**0.41666 - 0.055)

    def srgb_to_linear(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class SkyMlp(BaseBackground):
    def setup(self, **kwargs):
        import tinycudann as tcnn

        self.dir_encoding = tcnn.Encoding(
            n_input_dims=3, encoding_config=config_to_primitive(self.config.dir_encoding_config)
        )

        self.sky_mlp = tcnn.Network(
            n_input_dims=self.config.dir_encoding_config.degree**2,
            n_output_dims=3,
            network_config=config_to_primitive(self.config.mlp_network_config),
        )

    @torch.cuda.nvtx.range("sky_background.forward")
    def forward(self, rays_d, rgb, opacity):
        (b,h,w,c) = rays_d.shape
        rays_d = rearrange(rays_d,'b h w c -> (b h w) c')
        rgb = rearrange(rgb,'b h w c -> (b h w) c')
        opacity = rearrange(opacity,'b h w c -> (b h w) c')

        d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)

        dir_encoding = self.dir_encoding((d + 1) / 2)
        input_tensor = dir_encoding

        background_rgb = self.sky_mlp(input_tensor)

        # Compute the linear combination of the color within the volume and background
        rgb = rgb + background_rgb * (1.0 - opacity)

        return rearrange(rgb,'(b h w) c -> b h w c', b=b, h=h, w=w, c=c), rearrange(opacity,'(b h w) c -> b h w c', b=b, h=h, w=w, c=1)


class BackgroundColor(BaseBackground):
    def setup(self, **kwargs):
        self.background_color_type = self.config.color

        assert self.background_color_type in [
            "white",
            "black",
            "random",
        ], "Background color must be one of 'white', 'black', 'random'"

        if self.background_color_type == "white":
            self.color = torch.ones((3,), dtype=torch.float32).to(self.device)
        elif self.background_color_type == "black":
            self.color = torch.zeros((3,), dtype=torch.float32).to(self.device)

    @torch.cuda.nvtx.range("background_color.forward")
    def forward(self, rays_d, rgb, opacity):
        if self.background_color_type == "random":
            self.color = torch.rand((3,), dtype=torch.float32).to(rays_d)

        rgb = rgb + self.color * (1.0 - opacity)

        return rgb, opacity

class SkipBackground(BaseBackground):
    def setup(self, **kwargs):
        pass

    @torch.cuda.nvtx.range("skip_background.forward")
    def forward(self, rays_d, rgb, opacity) -> None:
        return rgb, opacity
