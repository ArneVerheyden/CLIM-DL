import torch
import glob, os

from torch.utils.data import Dataset

from data_utils.tiff import tiff_page_count, tiff_to_array
from models.psf import PSF
from simulation.grain_PL_simulation import TrainingDataSimulationOptions, generate_training_data, generate_noise_data

import random

class GeneratedPLOutlineDataset(Dataset):
    def __init__(self, 
                 length: int, 
                 sim_options: TrainingDataSimulationOptions,
                 transforms = None,
                 empty_chance = None,
                 noise_data: torch.Tensor | None = None):
    
        self.num_samples = length
        self.options = sim_options
        self.transforms = transforms
        
        self.empty_chance = empty_chance

        self.noise_data = noise_data

        if not noise_data is None:
            noise_width = noise_data.shape[1]
            noise_height = noise_data.shape[2]

            if noise_height < sim_options.grid_size or noise_width < sim_options.grid_size:
                raise ValueError(f"Noise data has wrong dimensions, each spation dimension should be at least: {sim_options.grid_size} pixels")



    def __len__(self):
        return self.num_samples
    
    def _get_noise_data(self, frames, shape):
        outline = torch.zeros(shape)

        n_noise_frames = self.noise_data.shape[0]
        frames = min(n_noise_frames, frames)

        frames_start = int(random.random() * (n_noise_frames - frames))

        # Randomize position
        grid_width = shape[0]
        grid_height = shape[1]

        noise_width = self.noise_data.shape[1]
        noise_height = self.noise_data.shape[2]

        start_x = int(random.random() * (noise_width - grid_width))
        start_y = int(random.random() * (noise_height - grid_height))

        return self.noise_data[frames_start:frames_start + frames, start_x:start_x + grid_width, start_y:start_y + grid_height], outline

    def __getitem__(self, index):
        if self.empty_chance and random.random() < self.empty_chance:
            # Have a 50% chance of using inputted noise for training
            if not self.noise_data is None and random.random() < 0.5:
                frames = int(self.options.min_seconds + random.random() * (self.options.max_seconds - self.options.min_seconds)) * self.options.sample_rate
                video, outline = self._get_noise_data(frames, (self.options.grid_size, self.options.grid_size))
            else:
                video, outline = generate_noise_data(self.options)
        else:
            video, outline = generate_training_data(self.options)
        
        if self.transforms:
            video = self.transforms(video)
        
        return video, outline



class GrainPLOutlineDataset(Dataset):
    def __init__(self, data_dir: str, frames_per_sample: int, transforms = None):
        available_data_files = []
        
        ## Get all PL.tiff files in the data_dir
        pl_tiff_files = glob.glob(
            os.path.join(data_dir, '**', 'PL.tiff')
        )

        total_samples = 0

        ## Check if there also exist an outline.tiff file in the same dir
        for tiff_file in pl_tiff_files:
            directory = os.path.dirname(tiff_file)

            outline_file = os.path.join(directory, 'outline.tiff')
            if not os.path.exists(outline_file):
                continue

            n_frames = tiff_page_count(tiff_file)
            total_samples += n_frames // frames_per_sample

            details = {
                'n_frames': n_frames,
                'dir': directory,
                'n_samples': n_frames // frames_per_sample,
                'outline': None,
                'frames': None,
            }
            available_data_files.append(details)

        self.total_samples = total_samples
        self.available_data_files = available_data_files
        self.frames_per_sample = frames_per_sample
        self.transforms = transforms

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int):
        for data_file in self.available_data_files:
            if index >= data_file['n_samples']:
                index -= data_file['n_samples']
                continue
            
            ## Load data if it has not been loaded yet
            if data_file['outline'] is None or data_file['frames'] is None:
                outline_file = os.path.join(data_file['dir'], 'outline.tiff')
                PL_file = os.path.join(data_file['dir'], 'PL.tiff')
                
                data_file['outline'] = tiff_to_array(outline_file)
                data_file['frames'] = tiff_to_array(PL_file)

            label = data_file['outline']
            data = data_file['frames'][index * self.frames_per_sample:(index + 1) * self.frames_per_sample]
            
            if self.transforms:
                data = self.transforms(data)

            return data, label



        return None, None

            
