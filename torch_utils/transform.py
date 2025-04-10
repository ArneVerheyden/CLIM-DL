from typing import Any
import torch
from data_utils.normalization import normalize_data
import torch.nn.functional as F
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeWarning
import warnings

# Specifically suppress the covariance warning
warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")

def gaussian(x, A, mu, sigma, b):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + b


class ZScoreNorm:
    def __init__(self) -> None:
        pass
    
    def __call__(self, data: torch.tensor) -> Any:
        std, mean = torch.std_mean(data, keepdim=False)

        return (data - mean) / std

class NormalizeIntensityTrace:
    def __init__(self) -> None:
        pass

    def __call__(self, data) -> Any:
        return normalize_data(data)
    
class SkipFrames:
    def __init__(self, skip: int) -> None:
        self.skip = skip

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        result_mask = torch.arange(0, data.shape[0]) % self.skip == 0

        return data[result_mask]
    

class UpscaleTransform:
    def __init__(self, scale=2):
        self.scale = scale
    
    def __call__(self, data: torch.Tensor):
        x = data.unsqueeze(0)

        x_upscaled = F.interpolate(
            x,
            scale_factor=(self.scale, self.scale),
            mode='bilinear',
            align_corners=False
        )

        return x_upscaled.squeeze(0).squeeze(0)
    
class BackgroundRemovalNormalize:
    def __init__(self):
        pass

    def __call__(self, data: torch.Tensor):
        n_bins=500
        threshold = 0.2

        data = data.clone()
        # print(data.shape)
        data -= data.min()
        data /= data.max()

        ## Get the value density
        counts, bins = torch.histogram(data[data > threshold].flatten(), bins=n_bins, density=True)
        I_vals = (bins[1:] + bins[:-1])/2

        ## Try to fit a guassion to the peak

        ### Estimate parameters of the peak
        peak_idx = torch.argmax(counts).item()

        peak_mask = torch.arange(max(0, peak_idx - n_bins//7), min(peak_idx + n_bins//7, n_bins - 1))

        A_guess = counts[peak_idx].item()
        mu_guess = I_vals[peak_idx].item()

        ### Fit the guassion
        guess = (A_guess, mu_guess, 0.05, 0.1)
        try:
            popt, _, infodict, errmsg, ier = curve_fit(gaussian, I_vals[peak_mask].numpy(), counts[peak_mask].numpy(), p0=guess, full_output=True)
        except:
            return NormalizeIntensityTrace()(data)
        
        mu = popt[1]
        sigma = popt[2]

        cutoff_point = mu/2

        background_mask = data < cutoff_point
        
        
        data = NormalizeIntensityTrace()(data)
        data[background_mask] = data.min()

        return data