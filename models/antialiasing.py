import torch

def make(name:str, config):
    match name:
        case 'none':
            pass
        case 'random':
            return RandomRayJitter(
                enabled=False,  # Start jittering from iteration N
                apply_every_n_iterations=config.dataset.train.ray_jittering.apply_every_n_iterations,
                device='cpu'
            )
        case 'stratified':
            return StratifiedRayJitter(
                enabled=False,  # Start jittering from iteration N
                apply_every_n_iterations=config.dataset.train.ray_jittering.apply_every_n_iterations,
                num_samples=config.dataset.train.ray_jittering.num_samples,
                device='cpu'
            )
        case _:
            raise ValueError(f'Unknown ray jitter type: {config.dataset.train.ray_jittering.type}')

# Adapted from InstantNGP / following Peter Shirley's blog:
# https://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
def pixel_to_disc_shirley(seed):
    """ seed is a point on the unit square [0, 1]"""
    a = 2 * seed[:, 0] - 1
    b = 2 * seed[:, 1] - 1
    mask = a * a > b * b
    pi = torch.pi
    r = torch.where(mask, a, b)
    phi = torch.where(mask,
                      (pi / 4.0) * (b / a),
                      (pi / 4.0) * (a / b) + (pi / 2.0))
    disc_coords = r * torch.cos(phi), r * torch.sin(phi)
    return torch.stack(disc_coords)


class StratifiedRayJitter:
    """ Uses informed stratified sampling which relies on perturbing a fixed anti-aliasing pattern"""

    def __init__(self, enabled=True, apply_every_n_iterations=1, num_samples=16, device='cuda'):

        self.enabled = enabled
        self.apply_every_n_iterations = apply_every_n_iterations
        self.device = device

        # Keeps track of how many iterations were skipped
        self.num_iterations_not_jittered = 0

        # Number of subsamples to use in pattern
        self.num_samples = num_samples

        # Subpixel offset values used by DirectX MSAA (Source: Ray Tracing Gems II)
        subpixel_means = dict(
            s1=  [[0.5, 0.5]],
            s2 = [[0.25, 0.25], [0.75, 0.75]],
            s4 = [[0.375, 0.125], [0.875, 0.375], [0.625, 0.875], [0.125, 0.625]],
            s8 = [[0.5625, 0.6875], [0.4375, 0.3125],
                  [0.8125, 0.4375], [0.3125, 0.8125],
                  [0.1875, 0.1875], [0.0625, 0.5625],
                  [0.6875, 0.0625], [0.9375, 0.9375]],
            s16 = [[0.5625, 0.4375], [0.4375, 0.6875],
                   [0.3125, 0.375], [0.75, 0.5625],
                   [0.1875, 0.625], [0.625, 0.1875],
                   [0.1875, 0.3125], [0.6875, 0.8125],
                   [0.375, 0.125], [0.5, 0.9375],
                   [0.25, 0.875], [0.125, 0.25],
                   [0.0, 0.5], [0.9375, 0.75],
                   [0.875, 0.0625], [0.0625, 0.0]]
        )
        self.subpixel_means = {k: torch.tensor(v) for k, v in subpixel_means.items()}

        # Max distance between points in this pattern
        self.subpixel_offset_max = dict(
            s1 = 0.5,
            s2 = 0.3535533905932738,
            s4 = 0.2795084971874737,
            s8 = 0.13975424859373686,
            s16 = 0.04419417382415922
        )

        assert f's{num_samples}' in self.subpixel_means, \
            f'num_samples={num_samples} not supported. Choose a value in: {list(self.subpixel_means.keys())}'
        self.pattern = self.subpixel_means[f's{num_samples}']
        self.relaxation = self.subpixel_offset_max[f's{num_samples}']

        # A generator of jittered sample patterns.
        # Reshuffles every time a new image size is encountered, otherwise will repeat a permuted pattern
        self.samples_generator = self._subsample_gen()

    def _shuffle(self, img_shape):
        """ Change the permuted order of samples """
        # Permute the order of subpixels in the pattern
        cyclic_order = torch.randperm(self.num_samples)
        # Each pixel starts from a different location in cyclic_order
        pixel_indices = torch.randint(low=0, high=self.num_samples, size=img_shape)
        return cyclic_order, pixel_indices

    def _subsample_gen(self):
        cyclic_order, pixel_indices, prev_shape = None, None, None
        while True:
            img_shape = yield   # Take some image_shape to jitter
            if prev_shape != img_shape:
                cyclic_order, pixel_indices = self._shuffle(img_shape)

            # Sample subpixel index
            sample_indices = cyclic_order[pixel_indices]
            # Advance sequence
            pixel_indices = (pixel_indices + 1) % self.num_samples

            # For each pixel: load a subpixel location from the pattern
            jittered_pixels = self.pattern[sample_indices]
            # Perturb a bit to introduce randomization
            perturb = self.relaxation * (torch.rand_like(jittered_pixels) * 2.0 - 1.0)
            jittered_pixels = (jittered_pixels + perturb) % 1.0
            # Cache shape for next call, so we know if to continue using the permutation or reshuffle
            prev_shape = img_shape

            yield jittered_pixels

    def __call__(self, img_shape):
        """ Given an image shape, returns a pattern of pixel values to sample """

        should_apply_jitter = self.num_iterations_not_jittered == 0
        self.num_iterations_not_jittered = (self.num_iterations_not_jittered + 1) % self.apply_every_n_iterations

        if self.enabled and should_apply_jitter:
            next(self.samples_generator)
            return self.samples_generator.send(img_shape)
        else:
            return 0.5 * torch.ones((*img_shape, 2), device=self.device)


class RandomRayJitter:
    def __init__(self, enabled=True, apply_every_n_iterations=1, device='cuda'):
        self.enabled = enabled
        self.apply_every_n_iterations = apply_every_n_iterations
        self.device = device

        # Keeps track of how many iterations were skipped
        self.num_iterations_not_jittered = 0

    def __call__(self, img_shape):
        should_apply_jitter = self.num_iterations_not_jittered == 0
        self.num_iterations_not_jittered = (self.num_iterations_not_jittered + 1) % self.apply_every_n_iterations

        if self.enabled and should_apply_jitter:
            return torch.rand((*img_shape, 2))
        else:
            return 0.5 * torch.ones((*img_shape, 2), device=self.device)
