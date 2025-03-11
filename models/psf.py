from typing import Any
import torch
import scipy

class PSF:
    def __init__(self) -> None:
        raise ValueError('Cannot instantiate abstract class')

    def __call__(self, array: torch.Tensor) -> Any:
        return scipy.signal.convolve(array, self.psf, mode='same')

class GuassionPSF(PSF):
    def __init__(self, std: float) -> None:
        ## Make the size of the kernel sufficiently large 
        ## so that the result of a convulation does not look blocky
        self.size = 6 * (int(std) + 1)
        self.std = std

        x = torch.arange(-self.size // 2 + 1, self.size // 2 + 1)
        y = torch.arange(-self.size // 2 + 1, self.size // 2 + 1)
        x, y = torch.meshgrid(x, y)

        psf = torch.exp(-(x**2 + y**2) / (2 * self.std**2))
        psf /= torch.sum(psf)  

        self.psf = psf


