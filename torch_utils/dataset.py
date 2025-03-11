import torch
import glob, os

from torch.utils.data import Dataset

from data_utils.tiff import tiff_page_count, tiff_to_array
from models.psf import PSF
from simulation.grain_PL_simulation import TrainingDataSimulationOptions, generate_training_data

class GeneratedPLOutlineDataset(Dataset):
    def __init__(self, 
                 length: int, 
                 sim_options: TrainingDataSimulationOptions,
                 transforms = None,):
        self.num_samples = length
        self.options = sim_options
        self.transforms = transforms

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
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

            
