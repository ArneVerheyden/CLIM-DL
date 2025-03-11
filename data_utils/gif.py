import numpy as np
import torch
import os

from PIL import Image

def write_gif(filename: str, frames: np.ndarray | torch.Tensor, framerate: float, overwrite: bool = True):
    """
        Writes image frames to a .gif file

        Arguments: 
            filename(str):
            frames(tensor): tensor containing frames, expected shape: (frame, width, height)
    """
    if not overwrite and os.path.exists(filename):
        raise ValueError(f'Filename {filename} already exists, use overwrite=True to write anyways.')

    frames = frames - frames.min()
    frames *= 255 / frames.max()

    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    frames = frames.astype(np.uint8)
    
    images = [Image.fromarray(frames[i, :, :]) for i in range(frames.shape[0])]

    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1/framerate,
        loop=0,             
        optimize=True,      
    )

