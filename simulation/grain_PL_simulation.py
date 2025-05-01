import numpy as np
from scipy.spatial import Voronoi
from matplotlib.path import Path

import random
import lloyd
import torch

from data_utils.ccd_noise import simulate_ccd_noise
from data_utils.gif import write_gif
from data_utils.tiff import array_to_tiff
from models.psf import PSF
from simulation.sim_intensity import SimulationParameters, SimulationType, simulate_intensity
import os

def _get_grain_mask(grid_size: int, 
                    vertices: torch.Tensor, 
                    grid_points: torch.Tensor) -> torch.Tensor:
    polygon = vertices
    
    path = Path(polygon)
    
    mask = path.contains_points(grid_points)
    return mask.reshape((grid_size, grid_size))

def _get_line_mask(grid_size: int, v0, v1) -> torch.Tensor:
    num_points = max(5 * grid_size, int(max(abs(v1[0] - v0[0]), abs(v1[1] - v0[1]))))

    x_values = torch.linspace(v0[0], v1[0], num_points * 2).int()
    y_values = torch.linspace(v0[1], v1[1], num_points * 2).int()

    filter = torch.logical_and(
        torch.logical_and(x_values >= 0, y_values >= 0),
        torch.logical_and(x_values < grid_size, y_values < grid_size)
    )

    x_values = x_values[filter]
    y_values = y_values[filter]

    mask = torch.zeros((grid_size, grid_size), dtype=torch.bool)

    mask[x_values, y_values] = True

    return mask

def _get_grain_outline(grid_size: int, 
                        vertices: torch.Tensor,
                        scaling: int) -> torch.Tensor:
    
    size = grid_size * scaling
    result = torch.zeros((size, size), dtype=torch.bool)
    for i in range(len(vertices)):
        v0 = vertices[i] * scaling
        v1 = vertices[(i + 1) % len(vertices)] * scaling

        line_mask = _get_line_mask(size, v0, v1)
        result = torch.logical_or(result, line_mask)

    return result

class TrainingDataSimulationOptions:
    def __init__(self, grid_size: int, min_grains: int, max_grains: int, 
                 min_noise: float, max_noise: float, sample_rate: int, seconds: int, 
                 min_blinker_transition: float, max_blinker_transition: float, 
                 min_base_counts: int, max_base_counts: int, min_hole_chance: float, 
                 max_hole_chance: float, min_boundary_dimish: float, max_boundary_dimish: float, 
                 min_blinker_strength: float, max_blinker_strength: float,
                 min_blinkers_average: int, max_blinkers_average: int,
                 static_prob: float,
                 psf: PSF, label_scaling: int=1):
        self.grid_size = grid_size
        self.min_grains = min_grains
        self.max_grains = max_grains
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.min_blinker_transition = min_blinker_transition
        self.max_blinker_transition = max_blinker_transition

        self.min_base_counts = min_base_counts
        self.max_base_counts = max_base_counts
        self.min_hole_chance = min_hole_chance
        self.max_hole_chance = max_hole_chance

        self.min_boundary_dimish = min_boundary_dimish
        self.max_boundary_dimish = max_boundary_dimish

        self.min_blinkers_average = min_blinkers_average
        self.max_blinkers_average = max_blinkers_average

        self.psf = psf
        self.label_scaling = label_scaling

        self.min_blinker_strength = min_blinker_strength
        self.max_blinker_strength = max_blinker_strength

        self.static_prob = static_prob

    def __repr__(self):
        return (f"Options(output_dir={self.output_dir}, n_data={self.n_data}, "
                f"grid_size={self.grid_size}, min_grains={self.min_grains}, "
                f"max_grains={self.max_grains}, min_noise={self.min_noise}, "
                f"max_noise={self.max_noise}, sample_rate={self.sample_rate}, "
                f"seconds={self.seconds}, min_blinker_transition={self.min_blinker_transition}, "
                f"max_blinker_transition={self.max_blinker_transition}, "
                f"min_base_counts={self.min_base_counts}, max_base_counts={self.max_base_counts}, "
                f"min_hole_chance={self.min_hole_chance}, max_hole_chance={self.max_hole_chance}, "
                f"min_boundary_dimish={self.min_boundary_dimish}, max_boundary_dimish={self.max_boundary_dimish}, "
                f"min_blinker_strength={self.min_blinker_strength}, max_blinker_strength={self.max_blinker_strength}, "
                f"min_blinker_average={self.max_blinkers_average}, max_blinkers_average={self.max_blinkers_average}, "
                f"psf={self.psf}, overwrite={self.overwrite})"
                f"label_scaling={self.label_scaling}")


def generate_training_data_to_file(n_data: int,
                                   output_dir: str,
                                   sim_options: TrainingDataSimulationOptions,
                                   overwrite: bool = False):
    for i in range(n_data):
        print(f'Generating data {i + 1}...')

        video, outline = generate_training_data(sim_options)
        n_frames = int(sim_options.sample_rate * sim_options.seconds)

        dir_name = f'sim_{i}_{int(sim_options.sample_rate)}hz_{sim_options.seconds}s_{sim_options.grid_size}x{sim_options.grid_size}'
        directory = os.path.join(output_dir, dir_name)
        os.makedirs(directory, exist_ok=True)

        video_filename = os.path.join(directory, 'PL.tiff')
        outline_name = os.path.join(directory, 'outline.tiff')
        gif_name = os.path.join(directory, 'PL.gif')

        array_to_tiff(video_filename, video, overwrite=overwrite)
        array_to_tiff(outline_name, outline, overwrite=overwrite)
        write_gif(gif_name, video[:n_frames//10, :, :], sim_options.sample_rate, overwrite=overwrite)

        print(f'Done generating data {i + 1}\n')

def generate_noise_data(sim_options: TrainingDataSimulationOptions):
    video_len = sim_options.seconds * sim_options.sample_rate
    grid_dim = sim_options.grid_size

    label_scale = sim_options.label_scaling

    ## Randomize the noise values a little bit
    mean = 1 + random.random() * 2
    hot_multiplier = int(1 + random.random() * 80)
    video = simulate_ccd_noise(
        (video_len, grid_dim, grid_dim),
        mean_dark_current=mean,
        hot_pixel_multiplier=hot_multiplier,
        hot_pixel_prob=0.004,
        use_torch=True)
        
    outline = torch.zeros((grid_dim * label_scale, grid_dim * label_scale))

    return video, outline

def generate_training_data(sim_options: TrainingDataSimulationOptions):
    grid_size = sim_options.grid_size
    min_grains = sim_options.min_grains
    max_grains = sim_options.max_grains
    min_noise = sim_options.min_noise
    max_noise = sim_options.max_noise
    sample_rate = sim_options.sample_rate
    seconds = sim_options.seconds
    
    min_blinker_transition = sim_options.min_blinker_transition
    max_blinker_transition = sim_options.max_blinker_transition
    
    min_base_counts = sim_options.min_base_counts
    max_base_counts = sim_options.max_base_counts
    
    min_hole_chance = sim_options.min_hole_chance
    max_hole_chance = sim_options.max_hole_chance
    
    min_boundary_dimish = sim_options.min_boundary_dimish
    max_boundary_dimish = sim_options.max_boundary_dimish
    psf = sim_options.psf

    min_blinker_strength = sim_options.min_blinker_strength
    max_blinker_strength = sim_options.max_blinker_strength

    min_blinkers_av = sim_options.min_blinkers_average
    max_blinkers_av = sim_options.max_blinkers_average

    assert min_grains <= max_grains, "Minimum amount of grains should be smaller or equal than the maximum amount of grains"
    assert min_noise <= max_noise, "Minimum amount of noise should be smaller than the maximum amount of noise"
    assert min_blinker_transition <= max_blinker_transition, "Minimum of blinker transition should be smaller than the maximum"
    assert min_base_counts <= max_base_counts, "Minimum amount of base counts should be smaller than the maximum amount"
    assert min_hole_chance <= min_hole_chance, "Minimum hole chance should be smaller than the maximum "
    assert min_blinker_strength <= max_blinker_strength, "Minimum blinker strength should be smaller than maximum blinker strength"
    assert isinstance(psf, PSF), "'psf' should be an instance of the class PSF" 

    n_frames = int(sample_rate * seconds)

    random_nums = torch.rand(7)
    
    ## Generate all the random parameters
    n_grains = int(min_grains + random_nums[0] * (max_grains - min_grains))
    noise = float(min_noise + random_nums[1] * (max_noise - min_noise))
    blinker_transition = float(min_blinker_transition + random_nums[2] * (max_blinker_transition - min_blinker_transition))
    base_counts = int(min_base_counts + random_nums[3] * (max_base_counts - min_base_counts))
    hole_chance = float(min_hole_chance + random_nums[4] * (max_hole_chance - min_hole_chance))
    boundary_dimish = float(min_boundary_dimish + random_nums[4] * (max_boundary_dimish - min_hole_chance))
    blinker_strength = float(min_blinker_strength + random_nums[5] * (max_blinker_strength - min_blinker_strength))
    blinkers_average = int(min_blinkers_av + random_nums[6] * (max_blinkers_av - min_blinkers_av))

    sim_params = SimulationParameters()

    sim_params.base_counts = base_counts
    sim_params.int_mod = 1.25
    sim_params.base_prob = blinker_transition
    sim_params.n_frames = n_frames
    sim_params.n_blinkers = 20
    sim_params.quencher_strength = blinker_strength

    ## Generate simulation
    video, outline = generate_simulated_grains(
        grid_size=grid_size,
        n_grains=n_grains,
        hole_chance=hole_chance,
        static_chance=sim_options.static_prob,
        noise_level=noise,
        simulation_params=sim_params,
        boundary_diminish=boundary_dimish,
        av_blinker_count=blinkers_average,
        psf=psf,
        label_scaling=sim_options.label_scaling,

        min_base_counts=min_base_counts,
        max_base_counts=max_base_counts,
    )
    
    return video, outline


def generate_simulated_grains(
        grid_size: int,
        n_grains: int, 
        hole_chance: float,
        static_chance: float,
        noise_level: float,
        simulation_params: SimulationParameters,
        boundary_diminish: float,
        av_blinker_count: int,
        min_base_counts: int,
        max_base_counts: int,
        psf: PSF,
        label_scaling: int=1):
    
    outline_size = grid_size * label_scaling
    X, Y = torch.meshgrid(
            torch.arange(grid_size),
            torch.arange(grid_size))
    grid_points = torch.vstack((X.flatten(), Y.flatten())).T
    
    points = torch.rand(n_grains, 2) * grid_size
    
    ## Relax the points so that we get a more even distribution of grain sizes
    field = lloyd.Field(points.numpy())
    field.relax()
    field.relax()

    vor = Voronoi(torch.from_numpy(field.get_points()))

    ## Get a mask of all regions of the Voronoi map so that we can set the intensity of 
    ## each grain individually
    outline_image = torch.zeros((outline_size, outline_size), dtype=torch.float32)
    grain_masks = []

    for region_idx in vor.regions:
        if not region_idx or -1 in region_idx:
            continue
        
        ## Check if the grain is to be a hole
        random = np.random.random()
        if random < hole_chance:
            continue

        vertices = vor.vertices[region_idx]
        ## Check for vertices that fall too for outside the grid
        ## This is too prevent OOM error because something seems to rarely go wrong with the lloyd relaxation causing a vertex to have a value of 10**10
        if (np.any(vertices > 11 * grid_size) or np.any(vertices < -10 * grid_size)):
            continue
        

        is_static = np.random.random() < static_chance

        grain_mask = _get_grain_mask(grid_size, vertices, grid_points)
        outline_mask = _get_grain_outline(grid_size, vertices, label_scaling)

        grain_masks.append((grain_mask, is_static))
        if not is_static:
            outline_image[outline_mask] = 1

    ## Get a 'mask' for the outline, so that we can diminish the intensity around the border according to the parameter 
    broadened_outline = torch.from_numpy(psf(outline_image))
    broadened_outline[broadened_outline < 0.2 * broadened_outline.max()] = 0
    broadened_outline = np.ceil(broadened_outline)

    broadened_outline = 1 - boundary_diminish * broadened_outline
    ## Simulate blinking of the grains

    simulation_params.n_particles = n_grains

    ## Generate random number of blinkers according to a poisson distribution
    blinker_counts = np.random.poisson(av_blinker_count, n_grains)

    base_intensities =  min_base_counts + np.random.random(n_grains) * (max_base_counts - min_base_counts) 
    simulated_intensity = torch.from_numpy( 
        simulate_intensity(simulation_params, base_intensities, blinker_counts)
    )

    simulated_video = torch.zeros((simulation_params.n_frames, grid_size, grid_size))

    ## Fill the different grains with their simulated intensity
    for i, mask_tuple in enumerate(grain_masks):
        mask, is_static = mask_tuple
        if not is_static:
            simulated_video[:, mask] = simulated_intensity[i].unsqueeze(1)
        else:
            ## If it is static give the grain a solid appearance with a random intensity
            simulated_video[:, mask] = base_intensities[i]

    ## Apply point spread function on all frames 
    for i in range(simulated_video.shape[0]):
        ## todo add broadened outline back in:  * broadened_outline
        simulated_video[i, :, :] = torch.from_numpy(psf(simulated_video[i, :, :]))

    ## Add proportional noise to the video
    hot_multiplier = int(np.random.random() * 5)
    hot_pixel_occurence = np.random.random() * 0.004
    noise_level = noise_level * torch.mean(simulated_video)
    
    simulated_video += simulate_ccd_noise(simulated_video.shape, 
                                          mean_dark_current=noise_level,
                                          hot_pixel_multiplier=hot_multiplier,
                                          hot_pixel_prob=hot_pixel_occurence,
                                          use_torch=True,) 
    

    return simulated_video, outline_image

    