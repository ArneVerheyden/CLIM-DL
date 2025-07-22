import numpy as np
import torch
import math

def simulate_ccd_noise(size, 
                       mean_dark_current=100, 
                       hot_pixel_prob=0.04, 
                       hot_pixel_multiplier=5,
                       use_torch: bool = False):
    
    mean_dark_current = int(mean_dark_current)
    hot_pixel_multiplier = int(hot_pixel_multiplier)
    
    # Base dark current (Poisson distributed)
    dark_frame = np.random.poisson(mean_dark_current, size) + np.random.normal(size=size) * math.sqrt(mean_dark_current) * math.sqrt(12)
    # dark_frame = np.random.random(size=size) * mean_dark_current

    noise_std = int(math.sqrt(mean_dark_current))

    # Hot pixels (salt noise)
    hot_pixels = np.random.random(size) < hot_pixel_prob
    hot_pixel_mean = noise_std * hot_pixel_multiplier
    dark_frame[hot_pixels] += np.random.poisson(hot_pixel_mean, size=len(dark_frame[hot_pixels]))
    hot_pixels = np.random.random(size) < hot_pixel_prob
    dark_frame[hot_pixels] -= np.random.poisson(hot_pixel_mean, size=len(dark_frame[hot_pixels]))

    result = np.round(dark_frame)
    
    if use_torch:
        return torch.Tensor(result)
    else:
        return result